import os
import re
import numpy as np
import pandas as pd
import math
import copy
import ast
from datetime import datetime
from datetime import timedelta
from itertools import product
import pickle
from skimage import io, filters, exposure, measure, feature
from skimage.transform import downscale_local_mean,resize,rescale
from scipy.ndimage import gaussian_filter,zoom
from skimage.color import label2rgb
from skimage.morphology import disk, dilation, erosion,remove_small_objects
from skimage.segmentation import clear_border
from scipy.ndimage.filters import generic_filter
from sklearn.cluster import DBSCAN
import cv2 as cv
from scyjava import jimport
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import ipywidgets as widgets
from IPython.display import display
from basicpy import BaSiC
from aicsimageio import AICSImage
import btrack

import tifffile

from . import generalFunctions as genF
from . import thresh_utils as tu
from .zProj import maxProj,avProj,minProj,signalProj,findSliceSelection,takeSlicesSelection,signalF
from .FindCentres import findCentres,findCentres2
from .noMergeStitch import noMergeStitch,noMergeStitch2,cenList2Size
from .LabelVideo import moment2Str,addTimeLabel
from .AlignExtract import findRegion,extractRegion
from .AlignExtract import rotCoord_NoCrop,rotate_image
from .exceptions import UnknownTifMeta,ChannelException
from . import definitions as defs
from . import findMeta as fMeta
from . import errorMessages as EM


class XFold:
    """
    This class represents an 'experiment folder'. That is a folder where you
    put all the data that you want the code to analyse and consider as one
    experiment. The object therefore holds information that concerns the
    global structure of the data.

    Attributes
    ----------
    XPath : str or list
        The path of the experiment folder. Can also be just one file. In this 
        case the XPath is set as the files parent directory and all other 
        files+directories in the directory as put in the Filters.
    XPathP : str
        The path of the parent directory of the experiment folder. 
    XFoldName : str
        The name of the experiment folder. I.e. os.path.split(XPath)[1].
    FieldIDMapList : list of list of str or 'get_FieldIDMapList_from_tags'
        Each element represents a session and is a list of fieldIDs (str),
        one for each field in that session. These IDs define which fields
        correspond to each other through different Sessions. I.e. field i in
        Session j may not correspond to field i in the next Session.
        Put 'get_FieldIDMapList_from_tags' if you want to put the discovered 
        F-tags (i.e. from file names) in as the FieldIDs.
    AllFieldIDs : list of str
        Each unique field ID.
    SessionsList : list of XFold.Session
        All the Sessions of the XFold, ordered by time from metadata.
    StartTimes : dict
        The dictionary keys are the field IDs and the values are the datetime
        object of specifing the moment considered as 'time zero' for that
        field.
    sizeDic : dict
        Is a dictionary of {fieldID:[maxYSize,maxXSize]}.
    blankTPs : list of list of ints
        Each element is the list [s,t,f] which uniquely identifies a data
        point which is blank, i.e. all pixels are 0 for all montage tiles,
        channels, and z-slices. s gives index of the data point's session
        within the SessionsList, t is the index of the time point within that
        Session and f the index of the field. This attribute is filled
        lazily(?) by the methods ExpTime2ST and ConcatFrame2ST, so you can't
        rely on them being complete, but they may save lots of processing time.
    nonBlankTPs : list of list of ints
        same as blankTPs but identiying any data point that is not all 0s.
    metaDic : dict
        A general dictionary of key : value pairs stored in the meta file as
        'key : value'.
    CustomTags : dict
        See __init__
    SegmentationXFolds : dict
        The values are ms.XFolds - they are the XFolds for segmentation masks 
        of the XFold. The values are the str name XFold.XPath. These XFolds 
        are made as needed, not during XFold init, e.g. in 
        makeTData(SegmentationMask...).
    silenceWarnings : bool 
        Whether warnings were silenced.
    """

    UINT16MAX = 65535
    RECOGNISED_EXTENSIONS = ['.tif','tiff','.lif','.czi','.nd2','.npz']
    FieldDir = 'Exp'
    chanOrder = {'BF':0,
                 'YFP':1,
                 'GFP':2,
                 'RFP':3,
                 'CFP':4,
                 'FR':5,
                 'Label':6,
                 'DAPI':7}

    def __init__(
        self,
        XPath,
        FieldIDMapList=None,
        MustNOTContain=[],
        MustContainAND=[''],
        MustContainOR=[''],        
        ignoreDirs=[],
        StartTimes = None,
        CustomTags = {},
        makeTFiles = True,
        makeTFilesVerbose=None,
        assumeConstantDims=False,
        SaveXFoldPickle=True,
        LoadFromPickle=False,
        OriginalXFold=None,
        silenceWarnings=False
    ):
        """
        Parameters
        ----------
        FieldIDMapList : None or str or list of str or list of list of str or 'get_FieldIDMapList_from_tags'
            User provided source to build the FieldIDMapList from.
            
            If None then the Nth field of each session if given ID 'N'.
            
            If str then it must be a filepath to a .txt file containing the
            information in format:
            A,B,C,D
            D,E
            C,F
            where each line represents a session and each comma-separated
            string is the ID for each field.
            Filepaths which are just one level deep are interpreted to just be
            the name of the file in the XFold parent directory.
            In cases where the file format has separated fields into separate files... ?
            If a list of str is provided then all sessions must have the same 
            number of fields and this provides the ID of each field.
            List of list of str format seems to not be supported yet.
            Put 'get_FieldIDMapList_from_tags' if you want to put the discovered 
            F-tags (i.e. from file names) in as the FieldIDs.            
        MustNOTContain : list of str
            Any files containing any of the strings in this list are ignored.
        MustContainAND : list of str   
            Only files containing all of the strings in this list are used. 
        MustContainOR : list of str
            Only files containing at least one of the strings in this list are 
            used. 
        ignoreDirs : list of str
            Names of directories which you ignore. Only works for directories 
            located directly in the XPath. 
        StartTimes : None or str or dict or datetime.datetime
            User provided source to build StartTimes dictionary from. The 
            final StartTimes attribute is a dictionary where the keys are the 
            fieldIDs and the values are the datetime objects defining time 
            zero for that field.
            
            If None then each field will be assigned the time it first appears
            in the data.
            
            If str then it must be a path to a .txt with the information.
            Filepaths which are just one level deep are interpreted to just be
            the name of the file in the XFold parent directory. The file must
            be in one of 2 formats:
            Format1:
            fieldID1: dd/mm/yyyy hh:mm:ss
            fieldID2: dd/mm/yyyy hh:mm:ss
            Where there is a line for every fieldID.
            Format2:
            dd/mm/yyyy hh:mm:ss
            Meaning every field started at the same time.
            
            If dict then we assume you have made the whole dictionary yourself.
        CustomTags : dict
            The format of this dictionary is {'Q':regex}. For each file, if 
            regex is found in the file name, it will be removed before it is 
            decided which Session it belongs to. I.e. it will be grouped with 
            any other files that have the same name once all tags are removed. 
            Also, the Q in the key of the discovered tag assigns which other 
            discovered tags it is grouped with in the Session and is used in 
            its counting and ordering. For now we are only allowing this when 
            the discovered madeBy is aicsimageio compatible. It probably 
            doesn't make sense to use it if software is known since that is 
            the point of loading all the metadata properly. I.e. would only 
            be useful for a software that doesn loads of unpredictable things 
            with filename tags.
        makeTFiles : bool
            Making the TFiles requires more processing since it looks into the
            files to see what is stored. You need to make them if you want to
            make a TData but otherwise you can set to False to save time.
        makeTFilesVerbose : None or ...
            ...
        assumeConstantDims : bool
            Will speed things up with loading xfold if you don't check files 
            for dimensions all the time but assume they are always the same as 
            the first file it looks at.   
        SaveXFoldPickle : bool or str
            If not False then it will save a pickle of the xfold that can be 
            easily reloaded next time. 
            - If True then the filepath of this pickle file will be 
            self.XPath+'.pkl'. 
            - If you provide an absolute path then it will save the file there. 
            If it ends in .pkl then this is the full file path but if it 
            doesn't then this is assumed to be a directory and the filename 
            XFoldName+'.pkl' will be added.
            -If you provide a string that is not an absolute path then this is 
            assumed to be relative to self.XPathP (i.e. NOT the location from 
            which you are running the code!) - so it is added to XPathP. 
            Again, if it doesn't end with .pkl then filename XFoldName+'.pkl' 
            is added. 
            * Note how this all means that to name it something other than 
            XFoldName+'.pkl' you have to give something ending in '.pkl'.
            * Note also that to save in the location of the notebook you can't 
            use '.' - that will save in self.XPathP! You have to pass e.g. 
            os.path.abspath('./Data.pkl'). It seems counterintuitive since it 
            seems like we are changing the '.' definition but we decided the 
            default should always be saving the .pkl next to the data. 
        LoadFromPickle : bool or str
            If not False then it will look for a .pkl file and if it finds it, 
            it loads the object and sets the state of self to that object's 
            state and exits this __init__.
            The filepath it tries to load follows the same logic as 
            SaveXFoldPickle.
            - If True then the filepath it tries is self.XPath+'.pkl'. 
            - If you provide an absolute path then it will try to load this 
            filepath.... if it ends in .pkl then this is the full file path 
            but if it doesn't then this is assumed to be a directory and the 
            filename XFoldName+'.pkl' will be added.
            - If you provide a string that is not an absolute path then this 
            is assumed to be relative to self.XPathP (i.e. NOT the location 
            from which you are running the code!) - so it is added to XPathP. 
            Again, if it doesn't end with .pkl then filename XFoldName+'.pkl' 
            is added. 
            * Note how this all means that if the name is something other than 
            XFoldName+'.pkl' you have to give it that something ending in 
            '.pkl'.
            * Note also that to load from the location of the notebook you 
            can't use '.' - that will try to load from self.XPathP! You have 
            to pass e.g. os.path.abspath('.'). It seems counterintuitive since 
            it seems like we are changing the '.' definition but we decided 
            multisesh filepath assumptions should always be that it is 
            relative to XPathP. 
        Chan/Chan2 : list of list of str
            The Chan of the Sessions.
        OriginalXFold : None or str of ms.XFold
            Allows building an XFold from an image dataset with no metadata 
            (other than SeshQ specified by filename tags) by taking the 
            metadata from the XFold specified by the str (i.e. filepath) or 
            ms.XFold provided here. NOTE: for str it loads xfold in simplest 
            way, it doesn't currently do anything with assumeConstantDims etc 
            - so could be slow. To avoid that provide a str ending in .pkl 
            here. This is used for .npz compressed segmentations currently.
        silenceWarnings : bool
            Whether warnings should be silenced.
        """
        self.silenceWarnings = silenceWarnings
        
        # dealing with if XPath is a file, see __doc__
        if os.path.isfile(XPath):
            XPath2,XFile = os.path.split(XPath)
            ffs = [f for f in os.listdir(XPath2) if f!=XFile]
            MustNOTContain += [os.path.join(XPath2,f) for f in ffs]
            XPath = XPath2
        
        self.XPath = XPath
        assert os.path.exists(XPath), 'The XPath you provided doesn\'t exist'
        assert os.path.split(self.XPath)[0]!='',EM.xf1
        self.XPathP = os.path.split(self.XPath)[0]
        self.XFoldName = os.path.split(self.XPath)[1]
        
        self.OriginalXFold = OriginalXFold
        if isinstance(OriginalXFold,str):
            if OriginalXFold[-4:]=='.pkl':
                self.OriginalXFold = XFold(XPath,LoadFromPickle=OriginalXFold)
            else:
                self.OriginalXFold = XFold(OriginalXFold)        

        self.SaveXFoldPickle = SaveXFoldPickle
        if SaveXFoldPickle:
            if isinstance(SaveXFoldPickle,str):
                absQ = os.path.isabs(SaveXFoldPickle)
                pklQ = SaveXFoldPickle[-4:]=='.pkl'

                if not absQ:
                    SaveXFoldPickle = os.path.join(self.XPathP,SaveXFoldPickle)
                if not pklQ:
                    SaveXFoldPickle = os.path.join(SaveXFoldPickle,
                                                   self.XFoldName+'.pkl')

            else:
                SaveXFoldPickle = self.XPath+'.pkl'
            if os.path.exists(SaveXFoldPickle):
                print(EM.xf5)  
            if not genF.can_write_to_folder(os.path.split(SaveXFoldPickle)[0]):
                print(EM.xf4)
        
        self.LoadFromPickle = LoadFromPickle
        if LoadFromPickle:
            if isinstance(LoadFromPickle,str):

                absQ = os.path.isabs(LoadFromPickle)
                pklQ = LoadFromPickle[-4:]=='.pkl'

                if not absQ:
                    LoadFromPickle = os.path.join(self.XPathP,LoadFromPickle)
                if not pklQ:
                    LoadFromPickle = os.path.join(LoadFromPickle,
                                                  self.XFoldName+'.pkl')

            else:
                LoadFromPickle = self.XPath+'.pkl'

            if not os.path.exists(LoadFromPickle):
                print(EM.xf6)
            else:
                with open(LoadFromPickle, 'rb') as file:
                    temp_xfold = pickle.load(file)
                self.__dict__ = temp_xfold.__dict__
                return

        assert os.listdir(XPath)!=[],EM.xf2        
        
        self.MustNOTContain = MustNOTContain
        self.MustContainAND = MustContainAND
        self.MustContainOR = MustContainOR
        self.ignoreDirs = ignoreDirs
        self.ignorePaths = [os.path.join(self.XPath,d) for d in ignoreDirs]
        self.CustomTags = CustomTags
        self.Warnings = []
        self.assumeConstantDims = assumeConstantDims
        
        # _seshData is temporary storage of session data since there are
        # various moments where it could be retrieved
        self._seshData = None

        self.FieldIDMapListIn = FieldIDMapList
        self.FieldIDMapList = self.buildFieldIDMapList(FieldIDMapList)
    
        self.AllFieldIDs = [fid for xv in self.FieldIDMapList for fid in xv]
        self.AllFieldIDs = list(set(self.AllFieldIDs))

        self.StartTimesIn = StartTimes
        self.StartTimes = self.buildStartTimes(StartTimes)
    
        self.SessionsList = [] # set in makeSessions()
        self.makeTFiles = makeTFiles
        self.makeTFilesVerbose = makeTFilesVerbose
        self.makeSessions(makeTFs=self.makeTFiles,
                          makeTFilesVerbose=self.makeTFilesVerbose,
                          assumeConstantDims=self.assumeConstantDims,
                          OriginalXFold=self.OriginalXFold,
                          silenceWarnings=self.silenceWarnings)

        self.HomogFiltDic = {} # see XFold.buildHomogFilts()
        self.AlignDic = {} # see XFold.buildStoredAlignments()
        self.TemplateDic = {} # see XFold.buildTemplateDic()
        self.WindowDic = {} # see XFold.buildWindowDic()
        self.MaskDic = {} # see XFold.buildMaskDic()

        self.ExtractedSizeDic = {} # see XFold.buildExtractedSizeDic()
        # see TData.ExtractRings() 
        # structure is {mask:{(FID,ringWidth,NSeg):[[ring,rlab,[ringSegs,],[alabs,]],],},}
        self.RingDic = {} 

        self.SavedFilePaths = []
        self.StitchCounts = [0,0]
        self.HomogOverflow = [0,0] #[no. of ims overf'ed,no. of ims treated]

        self.Assertions()
        
        self.blankTPs = []
        self.nonBlankTPs = []
        self.MetaDic = {}
        self.loadMeta()

        self.SegmentationXFolds = {}

        self.Chan = [s.Chan for s in self.SessionsList]
        self.Chan2 = [s.Chan2 for s in self.SessionsList]
           
        if SaveXFoldPickle:
            if not os.path.exists(SaveXFoldPickle):
                with open(SaveXFoldPickle, 'wb') as file:
                    pickle.dump(self,file)

    def Assertions(self):
        """Put any final checks during XFold initiation here."""
        assert isinstance(self.MustNOTContain,list),'MustNOTContain must be a list.'
        assert isinstance(self.MustContainAND,list),'MustContainAND must be a list.'
        assert isinstance(self.MustContainOR,list),'MustContainOR must be a list.'


    def buildFieldIDMapList(self,FieldIDMapList):
        """
        Interprets user input for FieldIDMapList source to build the true list.
        """

        if type(FieldIDMapList)==list:
            if not self._seshData:
                self.buildSessionData()
            seshLens = [s[1]['NF'] for s in self._seshData]
            # this assertion is assuming the list is a list of str because list
            # of list of str not supported yet
            assert all([n==len(FieldIDMapList) for n in seshLens]),EM.bfml
            FieldIDMapList = [FieldIDMapList for s in self._seshData]
        elif FieldIDMapList=='get_FieldIDMapList_from_tags':
            FieldIDMapList = []
            if not self._seshData:
                self.buildSessionData()
            for iis,sesh in enumerate(self._seshData):
                tags = [genF.getTagDic(tp,sesh[1]['madeBy']) for tp in sesh[0]]
                if 'F' in tags[0].keys():
                    tags = [tag['F'] for tag in tags]
                    FieldIDMap = list(dict.fromkeys(tags))
                else:
                    FieldIDMap = self.get_auto_FieldIDMap(iis)
                FieldIDMapList.append(FieldIDMap)
        elif isinstance(FieldIDMapList,str) and FieldIDMapList != '':
            if not os.path.split(FieldIDMapList)[0]:
                FieldIDMapList = os.path.join(self.XPathP,FieldIDMapList)
            with open(FieldIDMapList,'rt') as theFile:
                FM = theFile.read()
            FieldIDMapList = [sesh.split(',') for sesh in FM.split('\n')]
        elif not FieldIDMapList:
            if not self._seshData:
                self.buildSessionData()            
            FieldIDMapList = []
            for iis,sesh in enumerate(self._seshData):
                FieldIDMapList.append(self.get_auto_FieldIDMap(iis))
        else:
            raise Exception('FieldIDMapList format not recognised.')
        return FieldIDMapList

    
    def get_auto_FieldIDMap(self,seshN):
        """
        This gets a default FieldIDMap for the Session with index seshN. This 
        is taken from _seshData so it can be used before Sessions are built. 
        The default FieldIDMap list is just zero-padded numbers from 1 to NF. 
        Except if made by multisesh then it gets it from the filename tags.

        Parameters
        ----------
        seshN : int
            The Session index that you want to get the FiedlIDMap for.
        """
        if not self._seshData:
            self.buildSessionData()
            
        sesh = self._seshData[seshN]
        if sesh[1]['madeBy']=='multisesh' and not 'NFOriginal' in sesh[1].keys():
            FieldIDMap = sesh[1]['FieldIDMap']
        elif sesh[1]['madeBy']=='multisesh':
            FIDs = [genF.getTagDic(tp,'multisesh')['F'] for tp in sesh[0]]
            FieldIDMap = [fid[len(defs.FieldDir):] for fid in FIDs]
        else:
            NF = sesh[1]['NF']
            ndig = len(str(NF))
            FieldIDMap = [str(x+1).zfill(ndig) for x in range(NF)]

        return FieldIDMap
        

    def buildStartTimes(self,StartTimes):
        """
        Interprets user input for StartTimes to create StartTimes dictionary.
        
        Note
        -----
        If nothing provided by the user and it wasn't made by multisesh then 
        it is going to work out the first time each field appears and assign 
        this BUT in this function that process is just started, it will be 
        completed in XFold.makeSessions().

        It nothing provided and it was made by multisesh then it will load the 
        StartTimes that multisesh saved. Note how it loads it from one Session 
        but it is for the whole XFold... could be a problem.
        """
        
        if isinstance(StartTimes,dict):
            pass
        elif isinstance(StartTimes,datetime):
            XVFields = set([y for x in self.FieldIDMapList for y in x])
            StartTimes = {ID:StartTimes for ID in XVFields}
        elif isinstance(StartTimes,str):
            if os.path.split(StartTimes)[0] == '':
                StartTimes = os.path.join(self.XPathP,StartTimes)
            assert os.path.exists(StartTimes),EM.st1%StartTimes

            with open(StartTimes,'rt') as theFile:
                labData = theFile.read()
            # this gets rid of any spurious whitespace at end of the .txt file
            endWSpaceReg = r'(\s*)$'
            labData = re.sub(endWSpaceReg,'',labData)
            labData = labData.split('\n')
            # get rid of spurious white spaces at the end of lines:
            labData = [re.sub(endWSpaceReg,'',s) for s in labData]

            StartTimes = [line.split(': ') for line in labData]
            XVFields = set([y for x in self.FieldIDMapList for y in x])

            if len(StartTimes)==1 and len(StartTimes[0])==1:
                # format2 (see __init__):
                StartTimes = {k:labData[0] for k in XVFields}
            elif len(StartTimes)>1 and all([len(s)==2 for s in StartTimes]):
                # format1 (see __init__):
                StartTimes = {k:v for k,v in StartTimes}
            else:
                raise Exception(EM.st2)

            # now just do checks that everything is good:
            labRegs = set(list(StartTimes.keys()))
            regMom = r'\d{2}/\d{2}/\d{4} \d\d:\d\d:\d\d'
            matchQ = all([re.match(regMom,x) for x in StartTimes.values()])
            TX = '%d/%m/%Y %H:%M:%S'

            if labRegs==XVFields and matchQ:
                StartTimes = {k:datetime.strptime(v,TX)
                              for k,v in StartTimes.items()}
            elif not matchQ:
                raise Exception(EM.st2)
            else:
                raise Exception(EM.st3)

        elif not StartTimes:
            if not self._seshData:
                self.buildSessionData()      
            found = False
            for sesh in self._seshData:
                if sesh[1]['madeBy']=='multisesh' and 'StartTimes' in sesh[1].keys():
                    StartTimes = sesh[1]['StartTimes']
                    found = True
                    break
            if not found:
                self.Warnings.append(EM.stW)
                XVFields = set([y for x in self.FieldIDMapList for y in x])
                StartTimes = {}
                for field in XVFields:
                    for i,XV in enumerate(self.FieldIDMapList):
                        if field in XV:
                            StartTimes.update({field:i})
                            break
                # for now we don't know the metadata of the sessions so we just
                # leave it like this and update during makeSessions()
        else:
            raise Exception('StartTimes not in recognised format.')
        return StartTimes


    def buildSessionData(self):
        """
        This looks in XPath and retrieves all data required to make sessions.
        It groups image file paths into sessions and extracts metadata.
        
        Result
        ------
        self._seshData : list
            Of form [sesh0, sesh1,...seshN]. Where seshi = [seshTPs,seshMeta]. 
            Where seshTPs is a list of all filepaths of image files in that 
            session. They are normally ordered alphabetically and it is 
            important that this corresponds to correct time/field etc ordering 
            (i.e. SeshQ - in fact in makeTFiles SeshQ will be assigned 
            according to this ordering). Usually it will due to tag numbering 
            but, for example, multisesh output files need reordering to get 
            fields well ordered. seshMeta is the dictionary with all session 
            metadata.
        """

        # get all the image file paths you can find and process them
        walk = [x for x in os.walk(self.XPath) 
                if not any([os.path.commonpath([id,x[0]])==id 
                            for id in self.ignorePaths])]
        
        fps = [os.path.join(x[0],fn) for x in walk for fn in x[2]]
        # this prevents wrong ordering when no. of digits in a number changes
        fps2 = [re.sub(r'\d+', lambda x: x.group().zfill(4), fp) for fp in fps]
        fps = [fp for fp2,fp in sorted(zip(fps2,fps))]
        fps = [fp for fp in fps if not os.path.split(fp)[1][0]=='.']
        fps = [fp for fp in fps if not any(fl in fp for fl in self.MustNOTContain)]
        fps = [fp for fp in fps if any(mc in fp for mc in self.MustContainOR)]
        fps = [fp for fp in fps if all(mc in fp for mc in self.MustContainAND)]
            
        tps = [t for t in fps if t[-4:] in XFold.RECOGNISED_EXTENSIONS]

        tps2 = tps.copy()

        # for each session make: [all sesh paths,sesh metadata dictionary]
        sesh = []
        for tp in tps:
            if tp not in tps2:
                continue
            mb = fMeta.madeBy(tp)
            
            strippedPath = genF.stripTags(tp,mb,self.CustomTags)
            
            tps2b = [genF.stripTags(T,mb,self.CustomTags) for T in tps2]
            
            seshTPs = [T for T,Tb in zip(tps2,tps2b) if Tb==strippedPath]
            
            for T in seshTPs:
                tps2.remove(T)

            meta = fMeta.allSeshMeta(tp,seshTPs,
                                     self.CustomTags,
                                    silenceWarnings=self.silenceWarnings)
            
            if meta['madeBy']=='multisesh_compressed':
                if not self.OriginalXFold:
                    raise Exception(EM.bs1)
                meta2 = self.seshMetaFromOriginalXFold(tp)
                for k,v in meta2.items():
                    if k not in meta.keys():
                        meta[k] = v   
                
            # make sure fields are in good ordering since in multisesh the 
            # 'FTag' is the FieldID in the directory name and these might not 
            # be in alphabetical order. So orders by metadata SeshF found in 
            # tiff metadata.
            seshTPs = genF.orderFilePathsByField(seshTPs,mb,self.OriginalXFold)
            
            if isinstance(meta,dict):
                sesh.append([seshTPs,meta])
            elif isinstance(meta,list):
                for met in meta:
                    sesh.append([seshTPs,met])  

        # order the sessions by their startMom. multisesh saved datasets have 
        # a SessionN value so we sort by this second. Last we sort by 
        # alphabetical ordering of the TFiles.
        seshSM = [s[1]['startMom'] for s in sesh]
        seshN = [s[1]['SessionN'] for s in sesh]
        self._seshData = [S for _,_,S in sorted(zip(seshSM,seshN,sesh))]


    def makeSessions(self,
                     makeTFs=True,
                     makeTFilesVerbose=None,
                     assumeConstantDims=False,
                     OriginalXFold=None,
                     silenceWarnings=False):
        """
        Makes all the sessions found in the XPath.
        It applies the MustNOTContain, MustContainAND and MustContainOR to 
        filter out unwanted files. All Sessions are saved within the 
        XFold.SessionsList and also are returned in a list so you could save 
        them separately if you want.
        """

        self.SessionsList = []
        if not self._seshData:
            self.buildSessionData()
        assert len(self._seshData)==len(self.FieldIDMapList),EM.fm1

        # make the sessions
        allSesh = []
        for i,s in enumerate(self._seshData):

            # here we put in SessionN from metadata if it has been assigned, 
            # i.e. not -1. It is only assigned for multisesh saved datasets. 
            # This is all about allowing blank sessions to be added for 
            # missing data when you want to make multisesh processed data 
            # align with original data (e.g. for aligned segmentation masks) 
            # but not all Sessions were analysed.
            if s[1]['SessionN']==-1:
                seshN = i
            else:
                seshN = s[1]['SessionN']
                
            allSesh.append(
                Session(
                    self,
                    seshN,
                    self.FieldIDMapList[i],
                    s[1]['metadata'],
                    s[0],
                    allMeta = s[1],
                    makeTFiles = makeTFs,
                    makeTFilesVerbose=makeTFilesVerbose,
                    assumeConstantDims=assumeConstantDims, 
                    silenceWarnings=silenceWarnings
                ))

        # update StartTimes if we didn't complete it before
        # (i.e. if there wasn't a txt file provided we use metadata to guess)
        for k,v in self.StartTimes.items():
            if isinstance(v,int):
                self.StartTimes.update({k:allSesh[v].StartMom})
        # now you can update the Session.Times
        for s in allSesh:
            if s.Times is None:
                s.Times = s.getTimes()

        self.SessionsList = allSesh
        return allSesh

    def seshMetaFromOriginalXFold(self,tp):
        """
        This is probably only used for meadeBy==multisesh_compressed datasets. 
        There is no metadata attached to the image files so we take it from an 
        XFold that has been stored in the self xfold. We just need to get the 
        SessionN from the file name.

        Parameters
        ----------
        tp : str
            The filepath of one of the files in this Session.
        """
        
        tagRegex = re.compile(r'_s(\d{4})')    
        if re.search(tagRegex,tp).group(1):
            seshN = int(re.search(tagRegex,tp).group(1))
        else:
            raise Exception('no SessionN found')     

        allMeta = self.OriginalXFold.SessionsList[seshN].allMeta
        allMeta['Shape'] = list(allMeta['Shape'])
        allMeta['Shape'][3] = 1 # only works for Z=1 (usually z-projected)
        allMeta['Shape'][4] = 1 # only 1 channel for compressed Segmentations
        allMeta['Shape'] = tuple(allMeta['Shape'])
        
        return allMeta
        

    def buildHomogFilts(self,HomogFilts,matchsize=False):
        """
        This loads homogenisation filters (see TData.Homogensise) that are
        saved to the XFold so they don't have to be loaded multiple times.
        They are saved to XFold.HomogFiltDic which is a dict of channel names
        for keys and dicts of {'XF':array} for values. It overwrites any old
        filters in XFold.HomogFiltDic if equivalent is found in HomogFilts
        (but leaves old ones if not).

        Parameters
        ----------
        HomogFilts : str or dict
            If str then it is a path to a directory containing the filters.
            They must be named chan_XX.tif where chan is the channel name and
            XX if FF or DF for flat/dark field (see TData.Homogenise). Path
            with one level is taken to be in the XFold parent folder. If
            dict then keys are channel names and values are dicts with key =
            FF or DF (for flat field or dark field) and values are path to
            filter or the image itself.
        matchsize : bool
            If True then it will resize to the size of the images in the
            xfold. Will only do that if all the sizes are the same.
        """

        if isinstance(HomogFilts,str):
            if not os.path.split(HomogFilts)[0]:
                HomogFilts = os.path.join(self.XPathP,HomogFilts)
            fps = os.listdir(HomogFilts)
            HF = ['_FF.tif','_DF.tif']
            hNames = [c+f for c in XFold.chanOrder.keys() for f in HF]
            for h in hNames:
                if h in fps:
                    if h[:-7] not in self.HomogFiltDic.keys():
                        self.HomogFiltDic[h[:-7]] = {}
                    im = io.imread(os.path.join(HomogFilts,h)).astype('float32')
                    self.HomogFiltDic[h[:-7]][h[-6:-4]] = im
        elif isinstance(HomogFilts,dict):
            for k,v in HomogFilts.items():
                if k not in self.HomogFiltDic.keys():
                    self.HomogFiltDic[k] = {}
                for k2,v2 in HomogFilts[k].items():
                    if isinstance(v2,str):
                        im = io.imread(v2).astype('float32')
                        self.HomogFiltDic[k][k2] = im
                    elif isinstance(v2,np.ndarray):
                        self.HomogFiltDic[k][k2] = v2
                    else:
                        raise Exception('HomogFilts format not correct.')
        else:
            raise Exception('HomogFilts must be a str or dict')

        if matchsize:
            allSizes = [(s.NY,s.NX) for s in self.SessionsList]
            assert len(set(allSizes))==1, EM.bh1
            size = allSizes[0]
            for ch,ff in self.HomogFiltDic.items():
                for f,im in ff.items():
                    self.HomogFiltDic[ch][f] = resize(im,size)


    def buildStoredAlignments(self,storedAlignments,refresh=False):
        """
        Stored alignments is a .txt file that can be created when you run
        TData.AlignExtract and stores the alignments found so that they can be
        used again later to save time. This function loads a given .txt file
        to the XFold so it is ready to use.

        Parameters
        ---------
        storedAlignments : str
            The path to the stored alignments. If it is a path of one level
            then it assumes this is in the parent directory of the XFold.
        refresh : bool
            Whether to reload the provided storedAlignments if XFold.AlignDic
            already contains a dictionary under the key 'storedAlignments'.

        Builds
        ------
        XFold.AlignDic : dict
            Each key is the str storedAlignments (so that multiple
            storedAlignments can be stored) and each value is another
            dictionary. That 2nd dictionary have keys that are the codes
            that identify each session-time-field and the values are (ang,
            [shifty,shiftx], see TData.AlignExtract).
        """
        assert isinstance(storedAlignments,str),EM.ae7
        if storedAlignments in self.AlignDic.keys() and not refresh:
            return
        else:
            self.AlignDic[storedAlignments] = {}
        storedAlignmentsP = storedAlignments
        if os.path.split(storedAlignmentsP)[0]=='':
            storedAlignmentsP = os.path.join(self.XPathP,storedAlignmentsP)
        if not os.path.exists(storedAlignmentsP):
            open(storedAlignmentsP,'w').close()
        with open(storedAlignmentsP,'r') as file:
            aligns = file.read()
        aligns = aligns.split('\n')
        codeReg = r'(S\d+T\d+F\d+) : '
        codeReg += r'([+-]?\d*.?\d*) ([+-]?\d*.?\d*) ([+-]?\d*.?\d*)'
        for a in aligns:
            search = re.search(codeReg,a)
            if search:
                kk = search.group(1)
                ang = float(search.group(2))
                shift = [float(search.group(3)),float(search.group(4))]
                self.AlignDic[storedAlignments].update({kk:(ang,shift)})


    def buildTemplateDic(self,templates,refresh=False):
        """
        The templates are the images, resized and rotated to the good
        orientation, that you want to match and extract from the data using
        e.g. TData.AlignExtract. This function loads them into a dictionary so
        that they don't have to be loaded each time.

        Parameters
        ---------
        templates : str
            A path to a folder containing the template images saved as tifs.
            String with only one level of path is interpreted as a directory
            within the parent of XPath. The images must be in subdirectories
            for each field with name 'Exp'+fieldID.
        refresh : bool
            Whether to reload the provided templates if XFold.TemplateDic
            already contains a dict under the key 'templates'.

        Builds
        ------
        XFold.TemplateDic : dict
            Each key is the str templates (so that multiple sets of
            templates can be stored) and each value is another
            dictionary. That 2nd dictionary have keys that are the field IDs
            and the values are the images.
        """
        assert isinstance(templates,str),'Templates must be a string.'
        if templates in self.TemplateDic.keys() and not refresh:
            return
        else:
            self.TemplateDic[templates] = {}
        templatesP = templates
        if os.path.split(templatesP)[0]=='':
            templatesP = os.path.join(self.XPathP,templatesP)
        assert os.path.exists(templatesP),EM.ae1

        # now make the dictionary from FieldIDMapList
        temps = [os.path.join(templatesP,d) for d in os.listdir(templatesP)]
        temps = [t for t in temps if os.path.isdir(t)]
        check = [len(os.listdir(t))==1 or len(os.listdir(t))==0 for t in temps]
        assert all(check),EM.ae2
        fs = [os.path.split(t)[1][len(defs.FieldDir):] for t in temps]

        temps = [[os.path.join(t,f) for f in os.listdir(t)] for t in temps]
        temps = [t[0] if len(t)==1 else None for t in temps]

        for i,temp in enumerate(temps):
            if temp:
                with tifffile.TiffFile(temp) as tif:
                    temps[i] = tif.asarray().astype('float32')
        for f,t in zip(fs,temps):
            if isinstance(t,np.ndarray):
                self.TemplateDic[templates].update({f:t})


    def buildExtractedSizeDic(self,templates,refresh=False,templateDic=None):
        """
        When you use TData.alignExtract() it will pad the extraction to make
        sure all fields are the same size (so that a TData with multiple
        fields is not jagged). This gives the size they are padded to for each
        set of templates. It is essentially just the maximum dimensions of the
        templates (there is one for each field and they may be different
        sizes.)

        Parameters
        ----------
        templates : str
            A path to a folder containing the template images saved as tifs.
            String with only one level of path is interpreted as a directory
            within the parent of XPath. The images must be in subdirectories
            for each field with name 'Exp'+fieldID.
        refresh : bool
            Whether to recalculate if XFold.ExtractedSizeDic
            already contains a sizes under the key 'templates'.
        templateDic : dict
            This dict is of form {fieldID : image}, i.e. {str : array-like}.
            If you provide this then it will calculate sizes from the images
            of the dict. If not then it takes the images of the parent XFold's
            TemplateDic.

        Builds
        -------
        XFold.ExtractedSizeDic : dict
            Each key is the str templates (so that multiple sets of
            templates can be stored) and each value is the (ysize,xsize) of
            the images that alignExtract will create.
        """
        if templates in self.ExtractedSizeDic.keys() and not refresh:
            return
        if not templateDic:
            templateDic = self.TemplateDic
        if templates not in templateDic.keys():
            self.buildTemplateDic(templates)
            templateDic = self.TemplateDic
        templateDic = templateDic[templates]
        maxYSize = []
        maxXSize = []
        for k,tem in templateDic.items():
            ysizeT,xsizeT = tem.shape
            maxYSize.append(ysizeT)
            maxXSize.append(xsizeT)
        maxYSize = max(maxYSize)
        maxXSize = max(maxXSize)

        self.ExtractedSizeDic[templates] = (maxYSize,maxXSize)


    def buildWindows(self,windows,refresh=False):
        """
        The windows are a .txt file with the four corners of a rectangle
        defining where on a template image you find the region you want to
        extract data with using TData.ExtractProfileRectangle(). They are in
        format made with image j rectangle roi -> Save As -> XY Coordinates.
        This loads them for all fields so they don't have to be built each
        time.

        Parameters
        ---------
        windows : str
            A path to a folder containing the window files saved as .txt.
            String with only one level of path is interpreted as a directory
            within the parent of XPath. The files must be in subdirectories
            for each field with name 'Exp'+fieldID.
        refresh : bool
            Whether to reload the provided windows if XFold.WindowDic
            already contains a dict under the key 'windows'.

        Builds
        ------
        XFold.WindowDic : dict
            Each key is the str windows (so that multiple sets of
            windows can be stored) and each value is another
            dictionary. That 2nd dictionary have keys that are the field IDs
            and the values are the window data.
        """

        if windows in self.WindowDic.keys() and not refresh:
            return
        else:
            self.WindowDic[windows] = {}
        windowsP = windows
        if os.path.split(windowsP)[0]=='':
            windowsP = os.path.join(self.XPathP,windowsP)
        assert os.path.exists(windowsP),'Windows file not found.'

        # now make the dictionary from FieldIDMapList
        ws = [os.path.join(windowsP,w) for w in os.listdir(windowsP)]
        ws = [w for w in ws if os.path.isdir(w)]
        fs = [os.path.split(w)[1][len(defs.FieldDir):] for w in ws]

        ws = [[os.path.join(w,f) for f in os.listdir(w)] for w in ws]
        ws = [[t for t in w if '.txt' in t] for w in ws]
        assert all([len(w)==1 or len(w)==0 for w in ws]),EM.bw1
        ws = [w[0] if len(w)==1 else None for w in ws]

        for f,w in zip(fs,ws):
            if w:
                with open(w,'rt') as roi:
                    w = roi.read()
                w = [p.split('\t') for p in w.split('\n') if p!='']
                w = [[int(float(p[1])),int(float(p[0]))] for p in w]
                # you can give it trapezium window and we convert to rectangle
                w = genF.trap2rect(w)
                self.WindowDic[windows].update({f:w})


    def buildMasks(self,masks,refresh=False):
        """
        The masks are .tif files the same as the templates (see
        XFold.buildTemplateDic()) but with a white line drawn around a region
        that you want to take values from using TData.ExtractProfileRectangle.
        Pixels outside this region will be ignored. This loads them for all
        fields so they don't have to be imported and processed each time.

        Parameters
        ---------
        masks : str
            A path to a folder containing the mask tifs. String with only one
            level of path is interpreted as a directory within the parent of
            XPath. The files must be in subdirectories for each field with
            name 'Exp'+fieldID.
        refresh : bool
            Whether to reload the provided masks if XFold.MaskDic
            already contains a dict under the key 'masks'.

        Builds
        ------
        XFold.MaskDic : dict
            Each key is the str masks (so that multiple sets of
            masks can be stored) and each value is another
            dictionary. That 2nd dictionary have keys that are the field IDs
            and the values are the processed masks as numpy arrays.
        """

        if masks in self.MaskDic.keys() and not refresh:
            return
        else:
            self.MaskDic[masks] = {}
        masksP = masks
        if os.path.split(masksP)[0]=='':
            masksP = os.path.join(self.XPathP,masksP)
        assert os.path.exists(masksP),'Masks directory not found.'

        # now make the dictionary from FieldIDMapList
        ms = [os.path.join(masksP,m) for m in os.listdir(masksP)]
        ms = [m for m in ms if os.path.isdir(m)]
        fs = [os.path.split(m)[1][len(defs.FieldDir):] for m in ms]

        ms = [[os.path.join(m,f) for f in os.listdir(m)] for m in ms]
        ms = [[t for t in m if '.tif' in t] for m in ms]
        assert all([len(m)==1 or len(m)==0 for m in ms]),EM.bm1
        ms = [m[0] if len(m)==1 else None for m in ms]
        for f,m in zip(fs,ms):
            if m:
                m = genF.maskFromOutlinePath(m)
                self.MaskDic[masks].update({f:m})


    def BuildSummary(self):
        """
        Builds a string containing global information about
        the xfold. Stuff like what channels, how much data etc... print it to
        see for yourself!
        """

        if len(self.SessionsList)==0:
            return 'No Sessions in XFold'
        
        summary = ''

        if not self.SessionsList:
            self.makeSessions(makeTFs=self.makeTFiles,
                              makeTFilesVerbose=self.makeTFilesVerbose,
                              assumeConstantDims=self.assumeConstantDims,
                              OriginalXFold=self.OriginalXFold,
                              silenceWarnings=self.silenceWarnings)
        allSesh = self.SessionsList
        allTFiles = [TP for sesh in allSesh for TP in sesh.tFilePaths]

        summary += 'Total no. of sessions: ' + str(len(allSesh)) + '\n'
        summary += 'Total no. of tiff files: ' + str(len(allTFiles)) + '\n'

        # size in memory
        totSize = sum([os.stat(tp).st_size for tp in allTFiles])/1000000
        summary += 'Total memory of tiff files: ' + str(totSize) + ' MB\n'

        totalNT = str(sum([s.NT for s in allSesh]))
        summary += 'Total no. of time points (according to metadata): '
        summary += totalNT + '\n'
        uniqueF = str(len(set([y for x in self.FieldIDMapList for y in x])))
        summary += 'Total no. of fields (no. of unique ID): ' + uniqueF + '\n'

        # total duration of experiment
        firstStart = allSesh[0].StartMom
        lastStart = allSesh[-1].StartMom
        timeDelta = allSesh[-1].TStep
        totT = lastStart - firstStart + timeDelta*allSesh[-1].NT
        totD = str(totT.days)
        totH = str(totT.seconds//3600)
        totM = str(totT.seconds%3600//60)
        totT = totD + ' days, ' + totH + ' hours, ' + totM + ' minutes.'
        summary += 'Total time span: ' + totT + '\n'

        # NM,NZ and NC in 'set-like' form:
        summary += '\nThe following shows only the value of the given '\
                'attribute \nwhen it changes from one session to the next: \n'
        setT = str(genF.onlyKeepChanges([s.NT for s in allSesh]))
        setF = str(genF.onlyKeepChanges([s.NF for s in allSesh]))
        setM = str(genF.onlyKeepChanges([s.NM for s in allSesh]))
        setZ = str(genF.onlyKeepChanges([s.NZ for s in allSesh]))
        setC = str(genF.onlyKeepChanges([s.NC for s in allSesh]))
        setDZ = str(genF.onlyKeepChanges([s.ZStep for s in allSesh]))
        summary += 'Time points: ' + setT + '\n'
        summary += 'Fields: ' + setF + '\n'
        summary += 'Montage tiles: ' + setM + '\n'
        summary += 'z-Slices: ' + setZ + '\n'
        summary += 'number of channels: ' + setC + '\n'
        summary += 'ZStep: ' + setDZ + ' um\n'

        # channel names
        setCNames = str(genF.onlyKeepChanges([s.Chan for s in allSesh]))
        summary += 'names of channels: ' + setCNames + '\n'

        # session names:
        sNames = ''.join([s.Name+'\n' for s in allSesh])
        summary += '\nThe names of the sessions: \n' + str(sNames) + '\n'

        summary += 'Number of missing SeshQ (total in all Sessions): \n'
        summary += str(sum([len(s.MissingSeshQ) for s in self.SessionsList]))
        summary += '\n'
        
        return summary


    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)


    def printProcessingStats(self):
        """ 
        Some of the methods of other classes (especially TData) will
        report statistics on the things they have done which get saved
        into the parent xfold's attributes. This method prints them,
        typically to give a report at the end of a processing session.
        """
        # saved files
        print('Files saved from analysis of your xfold:')
        [print(e) for e in self.SavedFilePaths]
        # blank time points removed
        print('Number of auto-alinments during stitching '\
              'due to low signal: ',self.StitchCounts[0])
        print('Number of auto-alignments during stitching due '\
              'to large calculated shifts: ',self.StitchCounts[1])
        if self.HomogOverflow[0]>0:
            print('Fraction of images that had unit16 overflow during '\
                  'division by filter during field of view homogensiation: ',
                  self.HomogOverflow[1]/self.HomogOverflow[0])
        warnings = set(self.Warnings)
        for w in warnings:
            print(w)


    def checkTFiles(self,verbose=False):
        """
        This checks whether the number of images in the TFiles corresponds to
        the dimensions that it thinks it has. I.e. whether the file is
        corrupted somehow. It returns the paths of all corrupted files.
        """

        badFiles = []
        for s in self.SessionsList:
            if s.MadeBy=='Leica':
                print('checkTFiles not implemented yet for Leica')
                pass
            for tf in s.TFilesList:
                if verbose:
                    print('Checking ',os.path.split(tf.TPath)[1])
                try:
                    fMeta.file2Dims(tf.TPath)
                except AssertionError:
                    badFiles.append(tf.TPath)
        return badFiles


    def ExpTime2ST(self,
                   time,
                   FID,
                   returnTData=False,
                   saveMeta='.sessionstitcher'):
        """
        You give a time and fieldID and it returns the session number and
        session time point index for the non-zero-valued data point whose
        'experiment time' is closest to that time (for the specified field).

        Parameters
        ----------
        time : int
            The experiment time in mins that you want to find the closest
            data point to.
        FID : str
            The fieldID of the field that you are referencing.
        returnTData : bool
            Whether to return the TData of the required time point.
        saveMeta : bool or str
            If not False then it will append blank time point information to
            the provided file name which is in the XPath parent directory.

        Returns
        -------
        S : int
            The session index of the frame you are looking for. I.e. the
            position within  XFold.SessionsList.
        T : int
            The time point index within the session of the data point you are
            looking for.
        tdata : SessionStitcher.TData (optional)
            The TData of the time point you were looking for.
        """
        time = timedelta(minutes=time)
        startT = self.StartTimes[FID]
        data = []
        for iS,s in enumerate(self.SessionsList):
            for iT,t in enumerate(range(s.NT)):
                data.append([abs(s.StartMom+s.TStep*t-startT-time),iS,iT])
        data = sorted(data)

        for d in data:
            code = [d[1],d[2],self.FieldIDMapList[d[1]].index(FID)]
            if code in self.blankTPs:
                continue
            elif code in self.nonBlankTPs:
                if returnTData:
                    tdata = self.SessionsList[d[1]].makeTData(T=d[2],F=FID)
                    return [d[1],d[2],tdata]
                else:
                    return [d[1],d[2]]
            else:
                tdata = self.SessionsList[d[1]].makeTData(T=d[2],F=FID)
                if tdata.EmptyQ():
                    if saveMeta:
                        line = '\nblankTP ' + str(code)
                        fp = os.path.join(self.XPathP,saveMeta)
                        with open(fp,'a') as theFile:
                            theFile.write(line)
                    continue
                else:
                    if saveMeta:
                        line = '\nnonBlankTP ' + str(code)
                        fp = os.path.join(self.XPathP,saveMeta)
                        with open(fp,'a') as theFile:
                            theFile.write(line)
                    if returnTData:
                        return [d[1],d[2],tdata]
                    else:
                        return [d[1],d[2]]


    def ConcatFrame2ST(self,TP,FID,
                       returnTData=False,
                       saveMeta='.sessionstitcher',
                       quick=False,
                       very_quick=False):
        """
        This returns the session number and time point number of the image
        specified by its frame number in a concatenated video of non-blank
        time points.

        Parameters
        ----------
        TP : int
            The time point in the concatenated video, starts at 1 to match
            imagej!
        FID : str
            The field ID of the field you are interested in.
        returnTData : bool
            Whether to return the TData once you have reached the correct one.
        saveMeta : bool or str
            If not False then it will append blank time point information to
            the provided file name which is in the XPath parent directory.
        quick : bool
            If True it will only load the first Z and M of the TData and see
            if that is empty.
        very_quick : bool
            Here we would not even open the files, we assume it is nonBlank.
            Don't allow saving to .sessionstitcher here though.

        Returns
        -------
        S : int
            The session index of the frame you are looking for. I.e. the
            position within the SessionsList.
        T : int
            The time point index within the session of the data point you are
            looking for.
        tdata : SessionStitcher.TData (optional)
            The TData of the tim epoint you were looking for.
        """
        assert TP!=0,'TP should start at 1 (chosen to match imageJ!'
        assert isinstance(TP,int),'TP must be an int.'
        if very_quick:
            assert not saveMeta,'you cant saveMeta in very_quick mode because we dont check the tdata!'
        tp = 0
        for i,s in enumerate(self.SessionsList):
            if not FID in s.FieldIDMap:
                continue
            iF = s.FieldIDMap.index(FID)
            for t in range(s.NT):
                if [i,t,iF] in self.blankTPs:
                    continue
                elif [i,t,iF] in self.nonBlankTPs:
                    tp += 1
                else:
                    if quick:
                        tdata = s.makeTData(T=t,F=FID,Z=0,M=0)
                    elif very_quick:
                        pass
                    else:
                        tdata = s.makeTData(T=t,F=FID)
                    if very_quick:
                        tp += 1
                    elif tdata.EmptyQ():
                        if saveMeta:
                            line = '\nblankTP ['
                            line += str(i)+','+str(t)+','+str(iF)+']'
                            fp = os.path.join(self.XPathP,saveMeta)
                            with open(fp,'a') as theFile:
                                theFile.write(line)
                        continue
                    else:
                        tp += 1
                        if saveMeta:
                            line = '\nnonBlankTP ['
                            line += str(i)+','+str(t)+','+str(iF)+']'
                            fp = os.path.join(self.XPathP,saveMeta)
                            with open(fp,'a') as theFile:
                                theFile.write(line)
                if tp==TP:
                    if returnTData:
                        tdata = s.makeTData(T=t,F=FID)
                        return [i,t,tdata]
                    else:
                        return [i,t]

        assert False, EM.co1

        
    def loadMeta(self,file='.sessionstitcher'):
        """
        Some functions will save info to a meta file. This loads that info.
        See blankTPs.
        """
        fp = os.path.join(self.XPathP,file)
        data = []
        if os.path.exists(fp):
            with open(fp,'rt') as theFile:
                data = theFile.read().split('\n')
        for d in data:
            if d[:8]=='blankTP ':
                self.blankTPs.append([int(a) for a in d[9:-1].split(',')])
            elif d[:11]=='nonBlankTP ':
                self.nonBlankTPs.append([int(a) for a in d[12:-1].split(',')])
            elif re.search(r'(.*) : (.*)',d):
                ser = re.search(r'(.*) : (.*)',d)
                self.MetaDic.update({ser.group(1):ser.group(2)})   


    def Cellpose(self,
                 diameter,
                 seg_chan,
                 nuc_chan=None,
                 sessions=None,
                 T='all',
                 F='all',
                 M='all',
                 Z='all',
                 model_type='nuclei',
                 flow_threshold=0.4,
                 cellprob_threshold=0,
                 normalise='auto',
                 blur_sig=None,
                 clear_borderQ=False,
                 remove_small=None,
                 zProject=False,
                 saveSegmentations=None,
                 verbose=False,
                 printWarnings=False,
                 compress=False,
                 allowMissing=False):
        """
        This segments all the data in the XFold using cellpose. Note that it 
        loads every individual T,F,M,Z and segments each, to avoid this (e.g. 
        if you first want to zProject or Stitch) you will for now have to do 
        that yourself using the TData method CellPose.
        
        Parameters
        ----------
        diameter : float or array of floats
            The diameter of the thing being segmented in um. If array then the 
            shape must match the shape of the array minus the last 2 'XY' 
            dimensions.
        seg_chan : int or str
            The channel index or name you want to segment. 
        nuc_chan : int or str
            If you are doing a cytoplasm segmentation you can also 
            provide a nucleus channel to help. Leave this as None if you don't 
            have one.
        sessions : None or list or int
            If None then it processes all sessions. If a list of int then the 
            int are the session indices.    
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for C you can request one channel or a list of channels by
            name(s) and for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent.               
        model_type : 'cyto' or 'nuclei'
            What are you segmenting?
        flow_threshold : float
            A parameter in the segmentation algorithm. Increase to get more 
            masks.
        cellprob_threshold : float
            A parameter in the segmentation algorithm. Decrease to get more 
            masks.
        normalise : None or bool or (float,float)
            By default cellpose does normalisation - which is clipping at 1st 
            and 99th percentiles. I found this is ruining some seemingly 
            simple segmentations that have few nuclei, i.e. so the real signal 
            covers fewer than 1% of pixels. So here we have the option to do 
            no normlaiseation (False), their normalisation (True) or normalise 
            by provided percentiles (min,max).
        blur_sig : int or None
            Sometimes useful to blur the image a bit before segmentation 
            because cellpose does too detailed segmentation for high 
            resolution images.
        clear_borderQ : bool
            Whether to remover the objects in the output labelled which 
            are connected to the border.
        remove_small : None or int or 'auto'
            If int then remove mask objects that have an area smaller than 
            this in um^2. If 'auto' then it removes objects with an effective 
            diameter mroe than 4 time smaller than the diameter you have 
            provided to cellpose.
        zProject : bool
            Whether to do maximum projection of your data first.
        saveSegmentations : None or str
            Whether to save the segmentation masks. Str will be the name of 
            the folder they are saved into.
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.    
        compress : bool
            Whether to save compressed files or not.
        allowMissing : bool
            If True then missing data will be loaded as black files and 
            segmentation will be run as normal on that.            

        Returns
        --------
        allSegs : list of numpy array
            List, each element is session and contains an array which has 
            shape of data except for the channels axis removed.
        """
                     
        if isinstance(sessions,list):
            sesh = [self.SessionsList[s] for s in sessions]
        elif isinstance(sessions,int):
            sesh = [self.SessionsList[sessions]]
        else:
            sesh = self.SessionsList                     
            
        for i,s in enumerate(sesh):
            if verbose:
                print('Session: ',i)

            TT,FF,MM,ZZ,sc2 = s.parseTFMZC(T=T,F=F,M=M,Z=Z,C=seg_chan)
            if zProject:
                ZZ2 = ZZ
                ZZ = [0]
            sc2 = sc2[0]
            if not nuc_chan is None:
                 _,_,_,_,nc2 = s.parseTFMZC(T=T,F=F,M=M,Z=Z,C=nuc_chan)
                 nc2 = nc2[0]
    
            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m,' z: ',z)

                if zProject:
                    tdata = s.makeTData(T=t,F=f,M=m,Z=ZZ2,
                                        C=seg_chan,
                                        allowMissing=allowMissing)
                    tdata.zProject()
                else:
                    tdata = s.makeTData(T=t,F=f,M=m,Z=z,
                                        C=seg_chan,
                                        allowMissing=allowMissing)
                
                if not nuc_chan is None:
                    if zProject:
                        tdata2 = s.makeTData(T=t,F=f,M=m,Z=ZZ2,
                                             C=nc2,
                                             allowMissing=allowMissing)
                        tdata2.zProject()
                    else:
                        tdata2 = s.makeTData(T=t,F=f,M=m,Z=z,
                                             C=nc2,
                                             allowMissing=allowMissing)
                    tdata.AddArray(tdata2.data[:,:,:,:,[0]],s.Chan2[nc2])
                
                tdata.Cellpose(diameter,
                                 seg_chan=0,
                                 nuc_chan= None if nuc_chan is None else 1,
                                 model_type=model_type,
                                 flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold,
                                 normalise=normalise,
                                 blur_sig=blur_sig,
                                 clear_borderQ=clear_borderQ,
                                 remove_small=remove_small,
                                 addSeg2TData=True,
                                 printWarnings=printWarnings,
                                 compress=compress)

                if saveSegmentations:
                    tdata.TakeSubStack(C='Segmentation',updateNewSeshNQ=True)
                    tdata.SaveData(saveSegmentations,compress=compress)

    
    def YOLOSAM(self,
                 diameter,
                 nuc_chan,
                 yolo_model=None,
                 sam_predictor=None,  
                 DEVICE=None,
                 sessions=None,
                 T='all',
                 F='all',
                 M='all',
                 Z='all',
                 clear_borderQ=False,
                 saveSegmentations=None,
                 verbose=False,
                 printWarnings=False):
        """
        This segments all the data in the XFold using YOLOSAM. Note that it 
        loads every individual T,F,M,Z and segments each, to avoid this (e.g. 
        if you first want to zProject or Stitch) you will for now have to do 
        that yourself using the TData method YOLOSAM.
        
        Parameters
        ----------
        diameter : float or array of floats
            The diameter of the thing being segmented in um. If array then the 
            shape must match the shape of the array minus the last 2 'XY' 
            dimensions.
        nuc_channel : int
            The channel index of the nuclei signal that you're segmenting.
        clear_borderQ : bool
            Whether to remover the objects in the output labelled which 
            are connected to the border.
        saveSegmentations : None or str
            Whether to save the segmentation masks. Str will be the name of 
            the folder they are saved into.
        sessions : None or list
            If None then it processes all sessions. If a list of int then the 
            int are the session indices.  
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for C you can request one channel or a list of channels by
            name(s) and for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent. 
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.            

        Returns
        --------
        allSegs : list of numpy array
            If returnSegmentations then: List, each element is session and 
            contains an array which has shape of data except for the channels 
            axis removed.

        """
        import torch
        
        if not DEVICE:
            DEVICE = 'cuda:3' if torch.cuda.is_available else 'cpu'
            
        if not yolo_model:
            from ultralytics import YOLO
            YOLO_CKPT = r'/weka/kgao/projects/jump-nuclei-detection/ckpts/best_model.pt'
            yolo_model = YOLO(YOLO_CKPT).to(DEVICE)
            
        if not sam_predictor:
            from segment_anything import SamPredictor,sam_model_registry
            SAM_CKPT = r'/weka/kgao/public-repos/segment-anything/sam-ckpts/sam_vit_b_01ec64.pth'
            SAM_MODEL_TYPE = 'vit_b'
            sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT).to(device=DEVICE) 
            sam_predictor = SamPredictor(sam_model)  
        
        if isinstance(sessions,list):
            sesh = [self.SessionsList[s] for s in sessions]
        else:
            sesh = self.SessionsList                     
            
        for i,s in enumerate(sesh):
            if verbose:
                print('Session: ',i)

            TT,FF,MM,ZZ,_ = s.parseTFMZC(T=T,F=F,M=M,Z=Z)
    
            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('s: ',i,'t: ',t,' f: ',f,' m: ',m,' z: ',z)

                tdata = s.makeTData(T=t,F=f,M=m,Z=z,C=nuc_chan)
                tdata.YOLOSAM(diameter,
                                 0,
                                 yolo_model,
                                 sam_predictor,
                                 DEVICE,
                                 clear_borderQ=clear_borderQ,
                                 addSeg2TData=True,
                                 printWarnings=printWarnings)

                if saveSegmentations:
                    tdata.TakeSubStack(C='Segmentation',updateNewSeshNQ=True)
                    tdata.SaveData(saveSegmentations)
    
    
    def TrackSegmentations(self,
                           out_name,
                           sessions=None,
                           T='all',
                           F='all',
                           M='all',
                           Z='all',
                           C='all'):
        """
        !! NOTE: this is a draft version that only works when NF,NM,NZ,NC are 
        constant throughout sessions. And TFMZC don't work yet, it processes 
        everything. !!
        
        This does tracking of segmentation masks. Your XFold here must be a 
        segmentation mask. (i.e. you've reloaded a previous analysis which 
        saved just segmentations. It saves new tracked segmentations so that 
        the objects in the masks are labelled with their tracked label.

        Every tracked object is put into the new masks. Objects in the 
        original masks that don't appear in the tracks are not put in the new 
        masks. (Just false positives are removed)
                
        Parameters
        ----------        
        out_name : str
            The name of the directory where the tracks (h5 files) and tracked 
            masks (if saveMask is True) will be stored. It is assumed to be in 
            the parent directory of the XFold.
        T,F,M,Z : 'all' or list of int
            !! NOT implemented yet !!
            Which session times, fields, tiles and slices to load.          
        saveMasks : bool 
            Whether to save masks. Otherwise it just saves the track files.
        sessions : None or list
            If None then it loads all sessions. If a list of int then the int 
            are the session indices.
        """
                         
        fp = os.path.join(self.XPathP,out_name)
                         
        FEATURES = [
            "area", 
            "major_axis_length", 
            "minor_axis_length", 
            "orientation", 
            "solidity"
        ]
                
        if isinstance(sessions,list):
            sesh = [self.SessionsList[s] for s in sessions]
        else:
            sesh = self.SessionsList                         

        NTs = [s.NT for s in sesh]
        FS = list(range(sesh[0].NF))
        MS = list(range(sesh[0].NM))
        ZS = list(range(sesh[0].NZ))
        CS = list(range(sesh[0].NC))
        NY,NX = (sesh[0].NY,sesh[0].NX)
                  
        for f,m,z,c in product(FS,MS,ZS,CS):
            masks = np.zeros((sum(NTs),NY,NX),dtype='uint16')
            TTT = 0
            for s in sesh:
                mask1 = s.makeTData(F=f,M=m,Z=z,C=c).data[:,0,0,0,0].copy()
                                
                masks[TTT:TTT+s.NT] = mask1
                TTT += s.NT
            
            # create btrack objects (with properties) from the segmentation data
            # (you can also calculate properties, 
            # based on scikit-image regionprops)
            objects = btrack.utils.segmentation_to_objects(masks,
                                                    properties=tuple(FEATURES))
            
            with btrack.BayesianTracker() as tracker:
            
                # configure the tracker using a config file
                dir_path = os.path.dirname(os.path.realpath(__file__))            
                tracker.configure(os.path.join(dir_path,'btrack_config.json'))
                tracker.max_search_radius = 200
                tracker.tracking_updates = ["MOTION","VISUAL"]
                
                # append the objects to be tracked
                tracker.append(objects)           
                
                # track them (in interactive mode)
                # step size is just how often you print summary stats
                #tracker.track_interactive(step_size=100)
                tracker.track(step_size=100)
                
                # generate hypotheses and run the global optimizer
                tracker.optimise()
                
                # store the data in an HDF5 file
                if not os.path.isdir(fp):
                    os.mkdir(fp)
                fp2 = os.path.join(fp,'tracks_F'+str(f)
                                   +'_M'+str(m)
                                   +'_Z'+str(z)
                                   +'_C'+str(c)+'.h5')
                tracker.export(fp2, obj_type='obj_type_1')
                # get the tracks as a python list
                
                tracks = tracker.tracks
                
            if saveMasks:
                tracked_masks = np.zeros_like(masks)
                for track in tracks:
                    if not str(track['fate'])=='Fates.FALSE_POSITIVE':
                        for i in range(len(track)):
                            if not track['dummy'][i]:
                                x = int(track['x'][i])
                                y = int(track['y'][i])
                                t = track['t'][i]
                                old_label = masks[t,y,x]
                                new_label = track['ID']
                                tracked_masks[t][masks[t]==old_label] = new_label  
                TTT2 = 0
                for s in sesh:
                    tdata = s.makeTData(F=f,M=m,Z=z,C=c)
                    tdata.data[:,0,0,0,0,:,:] = tracked_masks[TTT2:TTT2+s.NT]
                    TTT2 += s.NT
                    tdata.SaveData(out_name)


    def makeTData(self,S,T='all',F='all',M='all',Z='all',C='all',
                  allowMissing=False,MY=False,MX=False,Segmentation=False):
        """
        S is the index of Session you are loading from, otherwise see 
        Session.makeTData.
        """
        sesh = self.SessionsList[S]
        return sesh.makeTData(T,F,M,Z,C,allowMissing,MY,MX,Segmentation)           
        

    def SaveDataFrameCrops(self,
                           df,
                           chan,
                           outDir,
                           Segmentation=False,
                           maskOutput=False,
                           returnDFWithPaths=False,
                           saveDFWithPaths=False,                           
                           equalSize=False,
                           verbose=False):
        """
        This saves cropped images from your xfold. The crops are specified by 
        a dataframe with xfold coordinates specified in columns 
        (SessionN,T,F,M,Z) and the crop region specified by column 'slice'. The 
        channels to save are specified by the parameter chan and each channel 
        is saved separately. The directory to save to is specified by outDir 
        and there are sub-directories for Session,F,T,M,Z. It can 
        return the dataframe with columns added for the paths of these saved 
        images (one column for each channel) which can be used by 
        PlotDataFrame() to plot them interactively. Note how we don't aim to 
        be able to make an XFold out of these images because the number of 
        crops per image is not constant so the array would be jagged.
        
        Parameters
        ----------
        df : pandas dataframe
            Dataframe must include columns 'SessionN','T','F','M','Z' and 
            'slice'.
        chan : False or int or str or list of int or str
            The channels of the XFold that you want to crop and save, in any 
            formats accepted by Session.parseTFMZC(). Remember to only put 
            channels here that definitely appear in the XFold's Sessions. To 
            include datasets outside this XFold add them to Segmentation (see 
            below). Can put False if you are only doing Segmentation ones.
        outDir : str 
            The path to save the cropped images too. If it is only a one level 
            path then the directory is put in XPathP.
        Segmentation : False or str or list
            Provide paths to other datasets (that exactly match shape of self) 
            that you want to also crop and save. If you give a string with no 
            directory divisors it looks for this directory in the XFold.XPathP.
        maskOutput : False or True or str
            Whether to set all image outside a specified mask to zero. If True 
            then there must be a channel called 'Segmentation' in your TData. 
            If a string is provided then it should match one key in 
            self.Segmentations (it will attempt to load it also if not 
            there).
        returnDFWithPaths : bool
            Whether to return df with columns added that contain the paths that 
            rows were saved to.       
        saveDFWithPaths : bool or str
            Whether to save df with columns added that contain the paths that 
            rows were saved to. If str then this string is the name or path of 
            the csv file, give string with no directory divisors to save in 
            XPathP. Put True to save in self.XPath+'.csv'.
        equalSize : bool
            Whether to pad the cropped images so they are all the same size 
            as the largest image in the dataframe.
        """
        df2 = df.copy()
        if isinstance(chan,int) or isinstance(chan,str):
            chan = [chan]
        if isinstance(Segmentation,str):
            Segmentation = [Segmentation]            
            
        if equalSize:
            max_y = df2['bbox-width-y'].max()
            max_x = df2['bbox-width-x'].max()

        df2['xfold coord'] = df2.apply(lambda row: str(row['SessionN']).zfill(4) 
                                     + str(row['T']).zfill(4) 
                                     + str(row['F']).zfill(4) 
                                     + str(row['M']).zfill(4) 
                                     + str(row['Z']).zfill(4), axis=1)
        
        all_coords = df2['xfold coord'].unique()

        if os.path.split(outDir)[0]=='':
            outDir2 = os.path.join(self.XPathP,outDir)
        else:
            outDir2 = outDir 

        all_chan = []
        if chan:
            all_chan += chan
        if Segmentation:
            all_chan += Segmentation

        for cc in all_chan:
            cc2 = str(cc).zfill(4).replace('/','->').replace('.','p')
            df2['out path ch_'+str(cc)] = df2.apply(lambda row: os.path.join(outDir2,
                                            'S'+str(row['SessionN']).zfill(4),
                                            'T'+str(row['T']).zfill(4),
                                            'F'+str(row['F']).zfill(4),
                                            'M'+str(row['M']).zfill(4),
                                            'Z'+str(row['Z']).zfill(4),
                                            'L'+str(row['label']).zfill(4)
                                              +'_C'+cc2
                                              +'.tif'),axis=1)

        for coord in all_coords:   

            if verbose:
                print('coord: ',coord)
                
            sub_df = df2[df2['xfold coord']==coord]

            sn = sub_df.iloc[0]['Session_Name']
            ss = sub_df.iloc[0]['SessionN']
            tt = sub_df.iloc[0]['T']
            ff = sub_df.iloc[0]['F']
            mm = sub_df.iloc[0]['M']
            zz = sub_df.iloc[0]['Z']

            tdata = self.makeTData(S=ss,T=tt,F=ff,M=mm,Z=zz,C=chan,
                                   Segmentation=Segmentation)
         
            if isinstance(maskOutput,str):
                xf2 = self.get_segmentation_xfold(maskOutput)              
                seg = xf2.makeTData(S=ss,T=tt,F=ff,M=mm,Z=zz,C='Segmentation')
                seg = seg.data[0,0,0,0,0].copy()
            elif maskOutput:
                assert 'Segmentation' in tdata.Chan,EM.sf2
                ccc = tdata.chan_index('Segmentation')
                seg = tdata.data[0,0,0,0,ccc].copy()

            _,_,_,_,chan_int = tdata.parseTFMZC(C=all_chan)

            for index,row in sub_df.iterrows():
                for cci,cc in zip(chan_int,all_chan):
                    im = tdata.data[0,0,0,0,cci,
                                row['slice'][0],row['slice'][1]].copy()
                    if maskOutput:
                        seg1 = seg[row['slice'][0],row['slice'][1]].copy()
                        im[seg1==0] = 0
                    if equalSize:
                        im = genF.pad2Size(im,(max_y,max_x))
                    impath = row['out path ch_'+str(cc)]
                    os.makedirs(os.path.split(impath)[0], exist_ok=True)
                    tifffile.imwrite(row['out path ch_'+str(cc)],im)

        if isinstance(saveDFWithPaths,str):
            if os.path.split(saveDFWithPaths)[0]=='':
                saveDFWithPaths = os.path.join(self.XPathP,saveDFWithPaths)
            df2.to_csv(saveDFWithPaths)
        elif saveDFWithPaths:
            out_csv = self.XPath+'.csv'
            df2.to_csv(out_csv)
                    
        if returnDFWithPaths:
            return df2

    
    def RegionProps(self,
                    Segmentation='Segmentation',                    
                    fun=[],
                    returnDF=True,
                    saveDF=False,
                    verbose=False,
                    tracking=False,
                    T='all',
                    F='all',
                    M='all',
                    Z='all',
                    allowMissing=False
                   ):
        """
        This does the equivalent to TData.RegionProps: it labels regions and 
        returns a pandas dataframe of measurements of each region in a mask, 
        a la skimage regionprops. But it does this to the whole XFold, 
        returning just one big dataframe. It also lets you specify functions 
        that are applied to each of the cropped images made from each region 
        to add further measurement columns to each row.

        Parameters
        ----------
        Segmentation : str
            Provide a string to specify the channel to do the crops and all 
            first basic regionProps measurements on. If the string isn't found 
            in self.Chan or self.Chan2 then it will attempt to load the 
            external Segmentation dataset specified by the string. In that 
            case it can be provided as absolute or relative path or just 
            folder name if it is in the XPathP.   
            !! Important note !! - this can be a labelled segmentation and it 
            shouldn't mess label ordering up by trying to relabel... but the 
            decision of whether it is labelled or not is quite basic (see 
            genF.isLabelledMaskQ)            
            !!Important note if using extra functions!! - The image send to 
            the function is cropped from the raw data according to the limits 
            of each region in this segmentation. So all regions you want to 
            use in functions must be within this segmentation. E.g. if you are 
            measuring nuclear and cytoplasmic signals, put the cytoplasmic 
            Segmentations here and the nuclear segmentations in the function.    
            !!Important note 2 if using extra functions!! - the label from 
            this mask will be passed to the function and it may be assumed 
            (depending on the function) that segmentations passed directly to 
            the function have corresponding labels.  
        fun : list
            See TData.RegionProps() info.
        returnDF : bool
            Whether to return the DF
        saveDF : bool or str
            If you want to save the dataframe as a csv then this should be the 
            name of the csv. It will be saved in the root of the XFold.
        verbose : bool
            Whether to print t,f,m,z,c progress.
        tracking : bool
            Whether to consider this a tracking dataset so most sgape stuff 
            will be thrown away and some new columns will be calculated.
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            and for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session
            but in future it should allow lists of selections so each 
            Session's selection can be independent.    
        allowMissing : bool
            If True then missing data will be loaded as black files and 
            RegionProperties will be run as normal on that.                
            
        Returns
        -------
        df : pandas dataframe
            One dataframe is returned where there is a row for every region 
            in the whole tdata. The following are column headings of the 
            dataframe:
            
           XPath : str
               The XPath of the raw data.
           SessionN,T,F,M,Z,C : int
               The index along each axis that this region comes from.
           Session_Name : str
               The name of the parent session.
        """
        
        df = pd.DataFrame()
        all_chan = []
        for f1,ch1,p1 in fun:
            all_chan += ch1
        all_chan += [Segmentation]
        all_chan = list(set(all_chan))
        
        for iss,s in enumerate(self.SessionsList):
            
            all_chan2 = [s.Chan[c] if isinstance(c,int) else c for c in all_chan]
            
            all_seg_chan = []
            all_data_chan = []
            for cmbc in all_chan2:
                if (cmbc not in s.Chan) and (cmbc not in s.Chan2):
                    all_seg_chan.append(cmbc)
                else:
                    all_data_chan.append(cmbc)

            # you have to sort the data chan because makeTData is fussy
            TT,FF,MM,ZZ,cc = s.parseTFMZC(T=T,F=F,M=M,Z=Z,C=all_data_chan)
            all_data_chan = [x for _,x in sorted(zip(cc, all_data_chan))]    

            if tracking:
                TT = ['all']

            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('s: ',iss,' t: ',t,' f: ',f,' m: ',m,' z: ',z)

                tdata = s.makeTData(T=t,F=f,M=m,Z=z,
                                    C=all_data_chan,
                                    Segmentation=all_seg_chan,
                                    allowMissing=allowMissing)

                df_sub = tdata.RegionProps(Segmentation=Segmentation,
                                       fun=fun,
                                       returnDF=True,
                                       saveDF=False,
                                       verbose=False,
                                       tracking=tracking)
                
                df = pd.concat([df,df_sub])

        if saveDF:
            if os.path.split(saveDF)[0]=="":
                saveDF = os.path.join(self.XPathP,saveDF)
            if not saveDF[-4:]=='.csv':
                saveDF = saveDF+'.csv'
            df_full.to_csv(saveDF)
        if returnDF:
            return df_full


    def FillInMissingSessions(self,xfold=False):
        """
        Put blank Sessions in self.SessionsList when missing ones are detected 
        by missing SessionN before the maximum SessionN discovered. Or, if 
        xfold is provided, blank Sessions are inserted so the SessionN in 
        self.SessionsList match that in xfold.

        xfold : False or multisesh.XFold
            If not False then we compare self.SessionsList with 
            xfold.SessionsList and enter blanks in self.SessionsList where 
            SessionN are missing.
        """
        SNs0 = [s.SessionN for s in self.SessionsList]
        
        if not xfold:
            SNsF = range(max(SNs0))
        else:
            SNsF = [i for i,s in enumerate(xfold.SessionsList)]
        
        for i in SNsF:
            if not i in SNs0:
                self.SessionsList.insert(i,Session(0,0,0,0,0,BlankSession=True))
                if len(self.FieldIDMapList)!=len(SNsF):
                    self.FieldIDMapList.insert(i,[])

            
    def AlignSegmentationLabels(self,
                                nuc_seg,
                                cyto_seg,
                                out_nuc_seg,
                                out_cyto_seg,
                                T='all',
                                F='all',
                                M='all',
                                Z='all',
                                verbose=True,
                                compress=False,
                                allowMissing=False):
        """
        You give this two labelled Segmentation datasets (that are aligned, 
        i.e. were calculated from the same raw dataset, ideally self) and this 
        saves a new version of each such that each label of the second 
        (cyto_seg) is changed to the label in the first that is most 
        overlapping. Regions in both for which a partner is not found are 
        deleted - ultimately each label is guaranteed to be exactly one pair. 
        I.e. it matches nuclei and cytoplasm segmentations.
        
        Parameters
        ----------
        nuc_seg,cyto_seg : str or ms.XFold
            The two Segmentation sets you want to match up. str must be a 
            filepath to root folder of dataset or if no file divisors found it 
            is assumed to be in self's parent directory. Or directly give the 
            XFold of the dataset.
        out_nuc_seg,out_cyto_seg : str
            The filepath to save the new segmentation datasets. If no file 
            divisors here it is assumed to be in self's parent directory.
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent. 
        verbose : bool
            Whether to print progress
        compress : bool 
            Whether to compress saved masks.
        allowMissing : bool
            If True then missing data will be loaded as black files and 
            alignment will be run as normal on that.    
            
        """
        if isinstance(nuc_seg,str):
            xf_n = self.get_segmentation_xfold(nuc_seg)
        elif isinstance(nuc_seg,XFold):
            xf_n = nuc_seg
        else:
            raise Exception('nuc_seg must be str or XFold')
            
        if isinstance(cyto_seg,str):
            xf_c = self.get_segmentation_xfold(cyto_seg)
        elif isinstance(cyto_seg,XFold):
            xf_c = cyto_seg
        else:
            raise Exception('cyto_seg must be str or XFold') 

        for i,s in enumerate(self.SessionsList):
            TT,FF,MM,ZZ,_ = s.parseTFMZC(T=T,F=F,M=M,Z=Z)
            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m,' z: ',z)
                tdata_N = xf_n.makeTData(i,t,f,m,z,allowMissing=allowMissing)
                tdata_C = xf_c.makeTData(i,t,f,m,z,allowMissing=allowMissing)
                ns2,cs2 = genF.align_nuc_cyto_labels(tdata_N.data[0,0,0,0,0],
                                                     tdata_C.data[0,0,0,0,0])
                tdata_N.data[0,0,0,0,0] = ns2
                tdata_C.data[0,0,0,0,0] = cs2
                tdata_N.SaveData(out_nuc_seg,compress=compress)
                tdata_C.SaveData(out_cyto_seg,compress=compress)


    def SubtractMasks(self,
                    mask1,
                    mask_sub,
                    out_mask,
                    T='all',
                    F='all',
                    M='all',
                    Z='all',
                    verbose=True,
                    compress=False,
                    allowMissing=False):
        """
        You give this two labelled Segmentation datasets (that are aligned, 
        i.e. were calculated from the same raw dataset, ideally self) and this 
        saves a new version of each such that each region of the second 
        (mask_sub) has been removed from the correspondingly labelled region 
        of mask1. 
        
        Parameters
        ----------
        mask1,mask_sub : str or ms.XFold
            The two Segmentation sets you want to subtract. str must be a 
            filepath to root folder of dataset or if no file divisors found it 
            is assumed to be in self's parent directory. Or directly give the 
            XFold of the dataset.
        out_mask : str
            The filepath to save the new segmentation dataset. If no file 
            divisors here it is assumed to be in self's parent directory.
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent. 
        verbose : bool
            Whether to print progress
        compress : bool
            Whether to compress the masks.
        allowMissing : bool
            If True then missing data will be loaded as black files and 
            mask subtraction will be run as normal on that.               
        """
        if isinstance(mask1,str):
            xf_c = self.get_segmentation_xfold(mask1)
        elif isinstance(mask1,XFold):
            xf_c = mask1
        else:
            raise Exception('mask1 must be str or XFold')
            
        if isinstance(mask_sub,str):
            xf_n = self.get_segmentation_xfold(mask_sub)
        elif isinstance(mask_sub,XFold):
            xf_n = mask_sub
        else:
            raise Exception('mask_sub must be str or XFold') 

        for i,s in enumerate(self.SessionsList):
            TT,FF,MM,ZZ,_ = s.parseTFMZC(T=T,F=F,M=M,Z=Z)
            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m,' z: ',z)
                tdata_C = xf_c.makeTData(i,t,f,m,z,allowMissing=allowMissing)
                tdata_N = xf_n.makeTData(i,t,f,m,z,allowMissing=allowMissing)
                cs2 = genF.subtract_masks(tdata_C.data[0,0,0,0,0],
                                                     tdata_N.data[0,0,0,0,0])
                tdata_C.data[0,0,0,0,0] = cs2
                tdata_C.SaveData(out_mask,compress=compress)



    def CleanMasks(self,
                   in_mask,
                   out_mask,
                   areaThreshold=None,
                   minAreaUnit='um^2',                   
                   circThreshold=None,
                   clearBorder=False,                    
                   printWarnings=True,
                   compress=True,
                   T='all',
                   F='all',
                   M='all',
                   Z='all',):
        """
        You give this a labelled Segmentation dataset (that is derived from 
        self) and this removes border objects, objects smaller than a threshold
        area and/or objects with a circularity smaller than a threshold from 
        the entire dataset. 

        Note: it assumes this is a labelled mask. Should add option to label 
        if needed.

        Parameters
        ----------
        in_mask : str or ms.XFold
            The Segmentation set you want clean. If str the it must be a 
            filepath to root folder of dataset or if no file divisors found it 
            is assumed to be in self's parent directory. Or directly give the 
            XFold of the dataset.
        areaThreshold : None or int or float
            If you want to remove objects smaller than a threshold then enter 
            it here. It is assumed to be in um^2.
        minAreaUnit : {'pixels','um^2'}
            The unit that minArea is taken to be in.              
        circThreshold : None or int or float
            If you want to remove objects with circularity smaller than a 
            threshold then enter it here.
        clearBorder : bool
            Whether to remove objects touching the image border.
        addAsNew : bool
            If True then it makes a new mask and adds as a new channel with 
            the same name as before but with 
            '_borderCleared_areaThresholdXX_circThresholdXX' added according 
            to what you chose.
        saveSegmentationsAndDelete : None or str
            Whether to save the cleaned segmentation masks. Str will be the 
            name of the folder they are saved into. This deletes everything 
            except for the segmentation just before saving because SaveData 
            doesn't yet have one channel only option.              
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.    
        compress : bool
            Whether the saved files should be compressed or not.   
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent.             
        """
        if isinstance(in_mask,str):
            xf_c = self.get_segmentation_xfold(in_mask)
        elif isinstance(in_mask,XFold):
            xf_c = in_mask
        else:
            raise Exception('in_mask must be str or XFold')

        for i,s in enumerate(self.SessionsList):
            TT,FF,MM,ZZ,_ = s.parseTFMZC(T=T,F=F,M=M,Z=Z)
            for t,f,m,z in product(TT,FF,MM,ZZ):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m,' z: ',z)
                td_S = xf_c.makeTData(i,t,f,m,z,allowMissing=allowMissing)

                td_S.CleanMasks(C='Segmentation',
                               areaThreshold=areaThreshold,
                               minAreaUnit=minAreaUnit,
                               circThreshold=circThreshold,
                               clearBorder=clearBorder,    
                               editDirectly=False,
                               addAsNew=False,
                               saveSegmentationsAndDelete=None,                     
                               printWarnings=printWarnings,
                               compress=compress)  
    
    
    def get_segmentation_xfold(self,
                               Segmentation,
                               assumeConstantDims='auto',
                               FieldIDMapList='auto',
                               SaveXFoldPickle=False,
                               LoadFromPickle=False,                               
                               returnXFold=True):
        """
        This makes the XFold of the Segmentation dataset specified by 
        Segmentation. It checks if it is already in self.SegmentationXFolds 
        and loads it with if not and returns it if requested.
        
        Parameters
        ----------
        Segmentation : str
            Ultimately this must be the absolute path to the Segmentation 
            dataset but if you provide a string with no directory divisors it 
            will prepend the path of the parent directory to self's root and 
            it will convert relative paths to absolute paths.
        assumeConstantDims : bool or 'auto'       
        FieldIDMapList : None or str or list of str or list of list of str 
                         or 'get_FieldIDMapList_from_tags' or 'auto' 
        SaveXFoldPickle : bool or str
        LoadFromPickle : bool or str
            All of these get passed to XFold __init__ when the new XFold is 
            made. If you choose 'auto' for assumeConstantDims or 
            FieldIDMapList then it sends the value from self to the new 
            __init__.     
        returnXFold : bool
            Whether to return the XFold that was created. I.e. otherwise it is 
            just loaded to self.SegmentationXFolds.
        """
        assert isinstance(Segmentation,str),'Segmentation must be a str'
        
        if os.path.split(Segmentation)[0]=='':
            Segmentation = os.path.join(self.XPathP,Segmentation)
            
        Segmentation = os.path.abspath(Segmentation)
        
        if Segmentation not in self.SegmentationXFolds.keys():
            if assumeConstantDims=='auto':
                assumeConstantDims = self.assumeConstantDims
            if FieldIDMapList=='auto':
                FieldIDMapList = self.FieldIDMapListIn            
                
            self.SegmentationXFolds[Segmentation] = XFold(Segmentation,
                                        assumeConstantDims=assumeConstantDims,
                                        OriginalXFold=self,
                                        FieldIDMapList=FieldIDMapList,
                                        SaveXFoldPickle=SaveXFoldPickle,
                                        LoadFromPickle=LoadFromPickle)

        if returnXFold:
            return self.SegmentationXFolds[Segmentation]


    def zProject(self,  
                 outDir,
                 sessions=None,
                 T='all',
                 F='all',
                 M='all',
                 Z='all',
                 C='all',
                 meth='maxProject',
                 downscale=None,
                 slices=1,
                 fur=False,
                 chan=None,
                 proj_best=True,
                 sliceBySlice=False,
                 meth_fun=signalF,
                 *args,
                 verbose=False):
        """
        z-projection of all data in the XFold.

        Parameters
        ----------
        outDir : str
            The directory to which to save the zProjections.
        sessions : None or list or int
            If None then it processes all sessions. If a list of int then the 
            int are the session indices.    
        T,F,M,Z,C : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for C you can request one channel or a list of channels by
            name(s) and for F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent.
        verbose : bool
            Whether to print progress.
        see TData.zProject() for other parameters.
        """
        if isinstance(sessions,list):
            sesh = [self.SessionsList[s] for s in sessions]
        elif isinstance(sessions,int):
            sesh = [self.SessionsList[sessions]]
        else:
            sesh = self.SessionsList 

        for i,s in enumerate(sesh):
            if verbose:
                print('Session: ',i)

            TT,FF,MM,ZZ,CC = s.parseTFMZC(T=T,F=F,M=M,Z=Z,C=C)
    
            for t,f,m,c in product(TT,FF,MM,CC):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m)
                    
                tdata = s.makeTData(T=t,F=f,M=m,Z=ZZ,C=c)
                tdata.zProject(meth=meth,
                               downscale=downscale,
                               slices=slices,
                               fur=fur,
                               chan=chan,
                               proj_best=proj_best,
                               sliceBySlice=sliceBySlice,
                               meth_fun=meth_fun,
                               *args,
                               verbose=False)
                tdata.SaveData(outDir)

    
    def Label2Edges(self,
                    outDir,
                    thickness=2,
                    outwards=False,
                    Segmentation=False,
                    sessions=None,
                    T='all',
                    F='all',
                    M='all',
                    Z='all',      
                    verbose=False,
                    compress=False
                   ):
        """
        This converts all masks in the XFold to just their edge with a 
        specified thickness. Note how you can go outwards for a kind of halo 
        or inwards for an edge that stays within original boundaries. The 
        outwards one does NOT include the original edge pixels so is a real 
        halo.

        Parameters
        ----------

        outDir : str
            Where to save the new XFold.
        thickness : int
            The thickness of the edges in pixels.
        outwards : bool
            Whether to expand the mask outwards to form the edge or inwards 
            (i.e. dilate or erode).
        Segmentation : False or str
            If False then it just works on any channels in self that are 
            called Segmentation. Otherwise it does nothing to self but does it 
            all to the full external XFold specified by Segmentation - in that 
            case Segmentation can be the absolute or relative path or just the 
            folder name if it is in self.XPathP.
        sessions : None or list or int
            If None then it processes all sessions. If a list of int then the 
            int are the session indices.    
        T,F,M,Z : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices.
            For F you can request one field or list of fields by
            its ID(s). Currently the same points are selected in each Session 
            but in future it should allow lists of selections so each 
            Session's selection can be independent.    
        verbose : bool
            Whether to print progress. 
        compress : bool
            Whether to compress the saved output.
        """
        if Segmentation:
            xf = self.get_segmentation_xfold(Segmentation)
        else:
            xf = self

        if isinstance(sessions,list):
            sesh = [xf.SessionsList[s] for s in sessions]
        elif isinstance(sessions,int):
            sesh = [xf.SessionsList[sessions]]
        else:
            sesh = xf.SessionsList 
            
        for i,s in enumerate(sesh):
            if verbose:
                print('Session: ',i)

            CC = [c1 for c1,c2 in zip(s.Chan,s.Chan2) if c2=='Segmentation']

            TT,FF,MM,ZZ,_ = s.parseTFMZC(T=T,F=F,M=M,Z=Z)
    
            for t,f,m,z,c in product(TT,FF,MM,ZZ,CC):
                if verbose:
                    print('t: ',t,' f: ',f,' m: ',m,' z: ',z)
                    
                tdata = s.makeTData(T=t,F=f,M=m,Z=z,C=c)
                tdata.data[0,0,0,0,0] = genF.labelMask2Edges(
                                                        tdata.data[0,0,0,0,0],
                                                        outwards=outwards,
                                                        thickness=thickness)
                tdata.SaveData(outDir,compress=compress)

        

class Session:
    """ 
    A Session object represents one imaging session, i.e. one run of an
    imaging protocol with your microscope software.
    It holds all details on what that protocol was and where files are stored.

    Attributes
    ----------
    Name : str
        A unique name for the Session. First take name from its files, with 
        'tags' removed (note there may be some 'madeBy' specific things done 
        in findMeta too). But this isn't necessarily unique, you might have 
        named things the same. So then add the relative path from XPath to 
        the data files, i.e. because the unique part might be in folder names. 
        One small problem with this is that the Name looks like a path but not 
        all the images actually have to be in this path... e.g. if it was 
        saved by multisesh then the field directory of the first TFile of the 
        Session (e.g. 'Exp01') will appear in the Name but TFiles from other 
        Fields will be found in other directories. We don't actually use this 
        Name like a path though so shouldn't matter.
    ShortName : str
        The same as Name but without the file path parts added so might not be 
        unique.
    StrippedFileName : str
        This is a name that will be used for saving TDatas of this Session. It 
        is made by stripping all tags from any of the TFIles of the Session. 
        It should be the same for any file in the Session. (this is asserted 
        in fact).        
    MadeBy : {'Andor','MicroManager','Zeiss','Nikon','Leica','multisesh',
                'multisesh_compressed','aicsimageio_compatible'}
        The software that made the data.
    MadeByOriginal : {same as MadeBy}
        If this has data has been analysed, MadeBy will show multisesh and 
        this shows what microscopy software made the original data.
    SessionN : int
        The position where you'll find this Session in the parent
        XFold.SessionsList.        
    FileSessionN : int
        When there are multiple Sessions in a file, the ordering in the 
        original file might not be the same as in XFold.SessionList because 
        we sort SessionList in order of time created. In Leica for example 
        they seem to be ordered alphabetically. We remember the original file 
        position here because it is needed for data retrieval from the file.
    imType : 
        
    TFilesList : list
        All TFile objects for that session.
    FieldIDMap : list
        The list of the parent XFold.FieldIDMapList that corresponds to this
        Session. Element i if this list is the name we give to the ith
        field of the image data of this session.
    Chan : list of str
        The names of channels taken directly from the metadata.
    Chan2 : list of str
        Chan but having been regularised to standard channel names by
        genF.chanDic().
    StartMom : datetime.datetime
        The datetime object representing the moment that the Session
        started. Note that this is different to Times and StartTimes!! This is 
        actually when it started rather than relative to some 
        experimentally-relevant imposed time.
    TStep : datetime.timedelta
        The time between time points in the session.
    ZStep : float
        The distance between z-slices in um.        
    pixSizeY,pixSizeX : float
        The size of the pixel, see self.pixUnit for the unit.
    pixUnit : str
        The unit of the number in pixSizeY,pixSizeX.
    pixSizeY_um,pixSizeX_um : float
        The size of the pixels in um.
    Times : Dic
        For each field and time-point in the data, this provides a time,
        defined as time since XFold.StartTimes[field]. The keys are fields and
        the values are a list of int times in min, one for each time point in
        the Session. See To Do below!
    TilePosX,Y : list
        All of the centres of montage tiles. The lists align with the SeshM 
        tile numbering. The numbers are given in meters! (Since this is what 
        Opera provides and we started with the Opera)
    BlankSession : bool
        Whether this is a blank Session.
    FoundMatrix : numpy array (dtype(bool)))
        A boolean array of dimensions (NT,NF,NM,NZ,NC) where each element 
        indicates whether the data for that Session index was found.
    AllFoundQ : bool
        Whether all data was found for this Session.
    MissingSeshQ : list of 5-tuples of int
        Each tuple in the list is the indices (T,F,M,Z,C) where the data is 
        missing. The list contains such a tuple for every missing bit of data.   
    MissingSeshF,T,M,Z,C : set of int
        Each int means there are at least some missing images which have that 
        SeshT,F,M,Z or C.
    silenceWarnings : bool
        Whether various warnings were silences during __init__.

    Methods
    -------
    makeTData - this is the most important method. It extracts specified image
                data from the session.

    To Do
    ------
    self.Times should be turned from int to datetime.timedeltas no??
    """

    def __init__(
        self,
        ParentXFold,
        SessionN,
        FieldIDMap,
        Metadata,
        tFilePaths,
        allMeta = None,
        makeTFiles = True,
        makeTFilesVerbose=None,
        assumeConstantDims=False,
        BlankSession=False,
        OriginalXFold=None,
        silenceWarnings = False,
    ):
        """
        We extract all metadata during intialisation to save in attributes.

        Parameters
        -----------
        ParentXFold : XFold object
            The parent XFold of the Session.
        SessionN : int
            The position where you'll find this Session in the parent
            XFold.SessionsList.
        FieldIDMap : list
            The list of the parent XFold.FieldIDMapList that corresponds to this
            Session. Element i if this list is the name we give to the ith
            field of the image data of this session.
        Metadata : str
            Any metadata that isn't in the image files. Extracted as a raw
            string. Doesn't seem to be used so far.
        tFilePaths : list
            List of file paths of all image files associated with this session.
        allMeta : dict
            The dictionary conataining all the main metadata that is retrieved 
            by multisesh.findMeta.allSeshMeta().
        makeTFiles : bool
            Whether to make the TFiles for the Session now.
        BlankSession : bool
            Whether tyo set this as a blank Session.
        OriginalXFold : None or str or ms.XFold
            Probably always just passed from ms.XFold, see there.
        silenceWarnings : bool
            Whether to silence various warnings during __init__.
        """
        self.BlankSession = BlankSession
        if BlankSession:
            return

        self.silenceWarnings = silenceWarnings
        
        self.ParentXFold = ParentXFold
        self.SessionN = SessionN
        self.FieldIDMap = FieldIDMap
        self.Metadata = Metadata
        self.tFilePaths = tFilePaths

        if allMeta:
            self.allMeta = allMeta
        else:
            self.allMeta = fMeta.allSeshMeta(self.tFilePaths[0],
                                             self.tFilePaths,
                                             self.ParentXFold.CustomTags)
        
        name1 = os.path.split(self.tFilePaths[0])[0]
        name1 = os.path.relpath(name1, self.ParentXFold.XPath)
        name1 = name1.replace('/','->') + '->'
        self.ShortName = self.allMeta['Name']
        self.Name = name1 + self.allMeta['Name']
        
        self.MadeBy = self.allMeta['madeBy']
        self.FileSessionN = self.allMeta['FileSessionN']

        self.imType = self.allMeta['imType']

        self.Chan = self.allMeta['Chan']
        self.Chan2 = [genF.chanDic(c) for c in self.Chan]
        self.NF = self.allMeta['NF']
        self.NT = self.allMeta['NT']
        self.NM = self.allMeta['NM']
        self.NZ = self.allMeta['NZ']
        self.NC = self.allMeta['NC']
        self.NY = self.allMeta['NY']
        self.NX = self.allMeta['NX']
        self.Shape = self.allMeta['Shape']

        self.StartMom = self.allMeta['startMom']
        self.TStep = self.allMeta['TStep']
        self.ZStep = self.allMeta['ZStep']        
        

        STs = [ParentXFold.StartTimes[F] for F in FieldIDMap]
        if not any([isinstance(s,int) for s in STs]):
            self.Times = self.getTimes()
        else:
            self.Times = None
        
        self.MOlap = self.allMeta['MOlap']
        self.NMY = self.allMeta['NMY']
        self.NMX = self.allMeta['NMX']
        self.FieldNMYs = self.allMeta['FieldNMYs']
        self.FieldNMXs = self.allMeta['FieldNMXs']
        
        self.TilePosX = self.allMeta['TilePosX']
        self.TilePosY = self.allMeta['TilePosY']    

        self.MontageOrder = self.allMeta['MontageOrder']
        
        if self.TilePosX and self.TilePosY:
            self.LRUPOrdering = np.lexsort((self.TilePosX,self.TilePosY))
        elif self.MontageOrder == 'LRUD':
            self.LRUPOrdering = list(range(self.NM))
        else:
            self.LRUPOrdering = None
        
        self.pixSizeY = self.allMeta['pixSizeY']
        self.pixSizeX = self.allMeta['pixSizeX']
        self.pixUnit = self.allMeta['pixUnit']  
        
        if self.pixUnit=='m':
            convert = 1000000
        elif self.pixUnit=='micron':
            convert = 1
        elif self.pixUnit=='um':
            convert = 1
        else:
            convert = 1
            print('Warning: pixUnit not in handled format so pixSizeY/X_um may be wrong')

        self.pixSizeY_um = self.pixSizeY*convert
        self.pixSizeX_um = self.pixSizeX*convert
        
        # this is for interpreting the tif files of the session -> what Q where
        self.FileTag2Len = {'T':{},'F':{},'M':{},'Z':{},'C':{}}

        self.tFilesMadeFlag = False
        if makeTFiles:
            self.makeTFiles(verbose=makeTFilesVerbose,
                            assumeConstantDims=assumeConstantDims)         
        else:
            self.TFilesList = []

        TNames = [os.path.split(tf.TPath)[1] for tf in self.TFilesList]
        CTags = self.ParentXFold.CustomTags
        TNames = [genF.stripTags(tn,self.MadeBy,CTags)for tn in TNames]
        assert all([tn==TNames[0] for tn in TNames]),EM.cs1
        self.StrippedFileName = TNames[0]
        
        # now we check what SeshQ we have found in the TFiles. Mostly only 
        # relevant when MadeBy=multisesh becaue we might not have processed 
        #all files. But do it for all madeBys anyway  
        self.FoundMatrix = np.zeros((self.Shape[:5]),dtype='bool')
        for tf in self.TFilesList:
            SeshQ = [tf.SeshT,tf.SeshF,tf.SeshM,tf.SeshZ,tf.SeshC]
            for t,f,m,z,c in product(*SeshQ):
                self.FoundMatrix[t,f,m,z,c] = True
        self.AllFoundQ = np.all(self.FoundMatrix)
        
        self.MissingSeshQ = list(zip(*np.where(self.FoundMatrix==False)))
        self.MissingSeshT = set([tup[0] for tup in self.MissingSeshQ])
        self.MissingSeshF = set([tup[1] for tup in self.MissingSeshQ])
        self.MissingSeshM = set([tup[2] for tup in self.MissingSeshQ])
        self.MissingSeshZ = set([tup[3] for tup in self.MissingSeshQ])
        self.MissingSeshC = set([tup[4] for tup in self.MissingSeshQ])
        
        if not self.AllFoundQ:
            if not silenceWarnings:
                print('Warning: these SeshQ not found: ',self.MissingSeshQ)
        

    def getTimes(self):
        """
        Gets the 'experiment time' of each field and time-point of the Session.
        I.e. the time from the xfold-defined start times (XFold.StartTimes).

        Returns
        -------
        times : dict
            The keys are fields and the values are a list of int times in min
            from the fields start time, one for each time-point in the Session.
        """
        times = {}
        for f in self.FieldIDMap:
            ts = [self.TStep*t for t in range(self.NT)]
            ts = [self.StartMom + t for t in ts]
            ts = [t - self.ParentXFold.StartTimes[f] for t in ts]
            times[f] = [int(t.total_seconds()/60) for t in ts]
        return times


    def makeTFiles(self,
                   setTFileList=True,
                   checkTF=False,
                   SeshQFromFile=False,
                   verbose=None,
                   assumeConstantDims=False):
        """
        This creates all TFiles in tFilePaths of the Session. It returns them
        in a list and saves them to the Session in self.TFilesList. Unless you
        set setTFileList=False.

        Parameters
        ----------
        setTFileList : bool
            Whether to set the Session's TFileList with the results.
        checkTF : bool
            Whether to raise exception if any TF is corrupt.
        SeshQFromFile : bool
            When the Session has been made from multisesh analysed data, it 
            may not contain all the data-points expected from the Session 
            metadata. E.g. some time points may not have been saved. In this 
            case you may want to get the SeshQ from the file's specially 
            saved tdata_metadata so that the SeshQ at least align with the 
            original Session. BUT M,Z,C often have None in their SeshQ, if 
            they were Stitched,zProjected,Labelled. So we only do this to T,F. 
            It is hard to imagine why you would have None in T,F. But note 
            that this causes problems if e.g. you only saved one F and it was 
            the 3rd one... then when loading tdatas it won't find the TFile 
            with SeshF=0. So don't use this one often.

        Note
        ----
        This is where the SeshQ of TFiles are definied! Therefore, for cases 
        where axis Q is spread across multiple files, this is where the 
        ordering of SeshQ is defined. Since TFilesList is ordered 
        alphabetically, the SeshQ will be ordered alphabetically according to 
        their tags.
        
        One big assumption, certainly in Andor, is that when the data along a
        certain dimension Q is split between different TFiles, we assume there
        is an ordering and the position ith TFile will always contain L data
        points along Q (remember there may be multiple ith TFiles when files
        are split in multiple axes). See the attribute Session.FileTag2Len
        """
        if verbose:
            print('entered makeTFiles')
        if setTFileList:
            self.TFilesList = []

        allTFiles = []
        
        if assumeConstantDims:
            nt,nf,nm,nz,nc,ny,nx = fMeta.file2Dims(self.tFilePaths[0],checkCorrupt=checkTF)
            lensDic = {'T':nt,'F':nf,'M':nm,'Z':nz,'C':nc}        

        for i,tf in enumerate(self.tFilePaths):
            if not assumeConstantDims:            
                if tf[-4:]=='.lif':
                    nt,nf,nm,nz,nc,ny,nx = self.Shape
                else:
                    nt,nf,nm,nz,nc,ny,nx = fMeta.file2Dims(tf,checkCorrupt=checkTF)
                lensDic = {'T':nt,'F':nf,'M':nm,'Z':nz,'C':nc}

            tagDic = genF.getTagDic(tf,self.MadeBy,self.ParentXFold.CustomTags)

            # update self.FileTag2Len
            for k,v in tagDic.items():
                if v not in self.FileTag2Len[k].keys():
                    self.FileTag2Len[k].update({v:lensDic[k]})

            # find TFile position in session from self.FileTag2Len
            seshQ = []
            for QQ,lenQ in lensDic.items():
                
                if QQ in tagDic.keys():
                    Qstart = 0
                    # adds dimension length to start until you reach your tag
                    for k2,v2 in self.FileTag2Len[QQ].items():
                        if tagDic[QQ]==k2:
                            break
                        else:
                            Qstart += v2
                    seshQ.append([i+Qstart for i in range(lenQ)])
                else:
                    seshQ.append(list(range(lenQ)))
            if self.MadeBy=='multisesh' and SeshQFromFile:
                with tifffile.TiffFile(tf) as tif:
                    ITD = tif.imagej_metadata['tdata_meta_data']
                seshT = genF.meta_str_2_dict(ITD)['SeshT']
                seshF = genF.meta_str_2_dict(ITD)['SeshF']
                seshQ[0] = seshT
                seshQ[1] = seshF
            allTFiles.append(TFile(self,i,tf,*seshQ,nt,nf,nm,nz,nc,ny,nx))
            if verbose:
                if i%verbose==0:
                    print("created TFile: "+str(i))
        if setTFileList:
            self.TFilesList = allTFiles

        # Andor leaves wrong NT in session metadata .txt file if session is
        # stopped before end. So check if this is the case and update to what
        # the files actually contain if needed.
        if self.FileTag2Len['T']:
            self.NT = sum(self.FileTag2Len['T'].values())
            self.Shape = tuple([self.NT,*self.Shape[1:]])

        self.tFilesMadeFlag = True

        return allTFiles


    def makeTData(self,T='all',F='all',M='all',Z='all',C='all',
                  allowMissing=False,MY=False,MX=False,Segmentation=False):
        """
        This builds a user specified TData from a session.

        Parameters
        ----------
        T,F,M,Z,C : {'all',int,list,str}
            The indices of the data points wrt to the session data.
            'all' for all, int for one index, list for list of indices
            or for C you can request one channel or a list of channels by
            name(s) and for F you can request one field or list of fields by
            its ID(s).
        allowMissing : bool
            If True then will leave image data black if metadata specified 
            files aren\'t found. 
        MY,X : {'all',int,list}
            This allows you to specify the montage section via Y,X (i.e. 
            columns and rows) rather than by montage tile index. You enter 
            the rows and columns according to LRUP ordering and it does the 
            rest. Have only made it work for opera where you have the 
            positions so far. 
        Segmentation : False or str or list
            If you have a corresponding dataset of segmentations that you can 
            make an exactly corresponding XFold of, then this adds the matching
            segmentation to the TData as a final channel. You can provide 
            Segmentation as a relative or absolute path, or if you provide a 
            string with no directory divisors it prepends the ParentXFold's 
            XPathP. The channel self.Chan2 name will be 'Segmentation_' + 
            exactly what you provided here as Segmentation (i.e. no conversion 
            to absolute paths etc) - the prepending of 'Segmentation_' means 
            the self.Chan name will automatically be 'Segmentation'. You can 
            provide a list of strings to add multiple new Segmenation channels.
        
        Returns
        --------
        tdata : TData
            A TData containing the requested data points.

        Notes
        ------
        Your request must be ordered, you can't change ordering here. This
        ultimately is because allowing that would require data from all the
        different TFiles to be put back together with a much slower and more
        complex numpy method.
        This is related to an implicit assumption that data extracted from a
        TFile will form a continuous sequence in the output data along any Q.
        (Note the data may not, however, originate from continuous sequences
        of data points within the TFile).
        """

        assert self.tFilesMadeFlag,EM.mt5

        T,F,M,Z,C = self.parseTFMZC(T,F,M,Z,C)
                    
        if MY is not False and MX is not False:
            if isinstance(MY,int):
                MY = [MY]
            if isinstance(MX,int):
                MX = [MX]                
            nmx = self.NMX
            nmy = self.NMY
            ind0 = list(range(nmx*nmy))
            M = [i for i in ind0 if i%nmx in MX and i//nmx in MY]
            if self.MontageOrder=='LRUD':
                pass
            elif self.MontageOrder=='TilePos':
                M = sorted([self.LRUPOrdering[i] for i in M])
            else:
                raise Exception('not known montage ordering for MY,MX input')
                
        # usersel is eventually list of indices for all channels that the user has chosen
        userSel = [T,F,M,Z,C]
                      
        dimsU = tuple([len(x) for x in userSel]+[self.NY]+[self.NX])
        assert all([Q==sorted(Q) for Q in userSel]),EM.mt2
        
        if 'aics_RGB' in self.allMeta.keys() and self.allMeta['aics_RGB']:
            aics_RGB = True
            userSel[4] = [0]
        else:
            aics_RGB = False

        if not all(userSel):
            # this is a special case where there is no data to load (but there 
            # may be Segmentation data to load)
            fullData = np.zeros(dimsU,dtype='uint16')
            pTFs = []
        elif self.MadeBy=='Leica':  
            # Leica has multiple sessions in one file
            # you have to retrieve the fields one by one because they're in
            # different 'scenes'
            img = AICSImage(self.TFilesList[0].TPath)
            pTFs = self.TFilesList
            
            # making sure you only count previous sessions from same lif file:
            fileSesh = [s for s in self.ParentXFold.SessionsList 
                        if s.TFilesList[0].TPath==self.TFilesList[0].TPath]
            fileSesh = [s for s in fileSesh if s.FileSessionN<self.FileSessionN]
            seshF0 = sum([s.NF for s in fileSesh])

            fullData = np.zeros(dimsU,dtype='uint16')
            for i,f in enumerate(userSel[1]):
                scene = seshF0+f
                img.set_scene(scene)
                ar = img.get_image_data("TZCYX",T=userSel[0],Z=userSel[3],C=userSel[4])
                ar = ar[:,np.newaxis,np.newaxis,:,:,:,:]
                fullData[:,i] = ar.copy()
        elif self.MadeBy=='Zeiss':       
            # Zeiss also needs images selection fetched by AICS, 
            # but we don't have the multiple sessions per file thing
            # will only work for one F per file so far!            
            img = AICSImage(self.TFilesList[0].TPath)
            pTFs = self.TFilesList
            fullData = np.zeros(dimsU,dtype='uint16')
            ar = img.get_image_data("TZCYX",T=userSel[0],Z=userSel[3],C=userSel[4])
            ar = ar[:,np.newaxis,np.newaxis,:,:,:,:]
            fullData = ar.copy()
        elif self.MadeBy=='Nikon': 
            # Nikon is one file per session, everything in there
            # haven't done montages yet but have done FTZC            
            img = AICSImage(self.TFilesList[0].TPath)
            pTFs = self.TFilesList
            fullData = np.zeros(dimsU,dtype='uint16')
            for i,f in enumerate(userSel[1]):
                img.set_scene(f)
                ar = img.get_image_data("TZCYX",T=userSel[0],Z=userSel[3],C=userSel[4])
                ar = ar[:,np.newaxis,np.newaxis,:,:,:,:]
                fullData[:,i] = ar[:,0].copy()                       
        else: 
            # from each file load the required data
            dataList = []
            pTFs = []
            for tf in self.TFilesList:
                seshQ = [tf.SeshT,tf.SeshF,tf.SeshM,tf.SeshZ,tf.SeshC]
                # what of user's selection are in this file:
                tfSel = [[q for q in Q if q in S] for Q,S in zip(userSel,seshQ)]
                
                if not all(tfSel):
                    continue
                pTFs.append(tf)
                # where along each seshQ of this file are these user selections:
                tfSelInd = [[S.index(q) for q in Q] for Q,S in zip(tfSel,seshQ)]
                
                dimsTF = tuple([len(x) for x in tfSel]+[self.NY]+[self.NX])
                prod = product(*tfSelInd)
                pageIndices = [genF.unravel(*X,*tf.Shape[0:5]) for X in prod]

                if self.MadeBy=='multisesh_compressed':
                    with np.load(tf.TPath) as tif:
                        _data = tif['mask'][*tfSelInd].reshape(dimsTF)
                else:
                    with tifffile.TiffFile(tf.TPath) as tif:
                        if aics_RGB:
                            _data = tif.asarray(key=pageIndices)
                        else:
                            _data = tif.asarray(key=pageIndices).reshape(dimsTF)
                        
                if aics_RGB:
                    dimsTF = list(dimsTF)
                    dimsTF[4] = 3                      
                    if len(_data.shape)==4:
                        _data = np.transpose(_data,(0,3,1,2)).reshape(dimsTF)
                    else:
                        _data = np.transpose(_data,(2,0,1)).reshape(dimsTF) 
                        
                if np.issubdtype(_data.dtype, np.floating):
                    _data = ((_data/np.max(_data))*65535).astype('uint16')
                        
                dataList.append([_data,tfSel])
            
            # check all the user selection has been retrieved
            # this is probably quite a weak check, just checking the right no. of
            # images have been loaded
            lenU = np.prod([len(q) for q in userSel])
            lenTFs = sum([np.prod([len(q) for q in d[1]]) for d in dataList])
            # for this check also must add padding tiles for unequal montage sizes
            # since those obv haven't been retrieved
            lenNonFM = [len(q) for i,q in enumerate(userSel) if i not in [1,2]]
            lenNonFM = np.prod(lenNonFM)
            for f in userSel[1]:
                ny = self.FieldNMYs[f]
                nx = self.FieldNMXs[f]
                lenTFs += lenNonFM*((self.NMX-nx)*self.NMY + (self.NMY-ny)*nx)
            if allowMissing and not lenU==lenTFs:
                print(EM.mt6)
            else:
                assert lenU==lenTFs,EM.mt3
                
            if 'aics_RGB' in self.allMeta.keys() and self.allMeta['aics_RGB']:
                userSel[4] = [0,1,2]   
                for d in dataList:
                    d[1][4] = [0,1,2]  
            
            # now fill in fullData with the bits from different files
            fullData = np.zeros(dimsU,dtype='uint16')
            for d in dataList:
                starts = [UQ.index(Q[0]) for Q,UQ in zip(d[1],userSel)]
                starts = tuple([slice(s,s+len(Q)) for s,Q in zip(starts,d[1])])
                fullData[starts] = d[0]
            
            if pTFs==[] and allowMissing:
                pTFs = [BlankTFile(self,None,None,*userSel)]

        tdataD = TData(pTFs,fullData,*userSel,self)
                      
        if isinstance(Segmentation,str):
            Segmentation = [Segmentation]
        if Segmentation:
            for seg in Segmentation:
                tdataD.LoadSegmentationChannel(seg,
                                           Chan='Segmentation_'+seg)
        
        return tdataD

                      
    def parseTFMZC(self,T='all',F='all',M='all',Z='all',C='all'):
        """
        This takes FTMZC in all formates and converts to indices for that 
        Session.
        
        It accepts 'all', int, -1 and list of int. For C it accepts channel 
        names. It accepts C either converted by chanDic or not: it checks for 
        non-converted in self.Chan first, then checks in self.Chan2. Note that 
        asking for converted version might be easier (i.e. you can type DAPI 
        without having to remember what the original channel was called) but 
        if there happens to be another channel that coverts to it (i.e. 
        another 'DAPI') you could end up with the wrong one! For F it accepts 
        FieldID also.
        
        Note this doesn't have the extra feature that TData.parseTFMZC() has 
        that it also searches C for 'Segmentation_'+C because this is never 
        added to Session.Chan only to TData.Chan2.
        """

        if T=='all':
            TT = list(range(self.NT))
        elif T==-1 or T==[-1]:
            TT = [self.NT - 1]        
        elif isinstance(T,list) and all([isinstance(t,int) for t in T]):
            TT = T.copy()     
        elif isinstance(T,(int,np.integer)):
            TT = [int(T)]       
        elif not T:
            TT = []
        else:
            raise Exception('T was not supplied in a supported format!')   
            
        if F=='all':
            FF = list(range(self.NF))
        elif F==-1 or F==[-1]:
            FF = [self.NF - 1]
        elif isinstance(F,(int,np.integer)):
            FF = [int(F)]
        elif isinstance(F,list) and all([isinstance(f,int) for f in F]):
            FF = F.copy()
        elif isinstance(F,str):        
            assert f in self.FieldIDMap,'Couldn\'t find all your requested F in the TData.FieldIDs'
            FF = [self.FieldIDMap.index(F)]
        elif isinstance(F,list) and all([isinstance(f,str) for f in F]):
            check = all([f in self.FieldIDMap for f in F] )
            assert check,'Couldn\'t find all your requested F in the TData.FieldIDs'
            FF = [self.FieldIDMap.index(f) for f in F] 
        elif not F:
            FF = []            
        else:
            raise Exception('F was not supplied in a supported format!')     
            
        if M=='all':
            MM = list(range(self.NM))
        elif M==-1 or M==[-1]:
            MM = [self.NM - 1]           
        elif isinstance(M,list) and all([isinstance(m,int) for m in M]):
            MM = M.copy()    
        elif isinstance(M,(int,np.integer)):
            MM = [int(M)] 
        elif not M:
            MM = []              
        else:
            raise Exception('M was not supplied in a supported format!')        
            
        if Z=='all':
            ZZ = list(range(self.NZ))    
        elif Z==-1 or Z==[-1]:
            ZZ = [self.NZ - 1]        
        elif isinstance(Z,list) and all([isinstance(z,int) for z in Z]):
            ZZ = Z.copy()    
        elif isinstance(Z,(int,np.integer)):
            ZZ = [int(Z)] 
        elif not Z:
            ZZ = []              
        else:
            raise Exception('Z was not supplied in a supported format!')        
            
        if C=='all':
            CC = list(range(self.NC))
        elif C==-1 or C==[-1]:
            CC = [self.NC - 1]        
        elif isinstance(C,list) and all([isinstance(c,int) for c in C]):
            CC = C.copy()     
        elif isinstance(C,(int,np.integer)):
            CC = [int(C)]        
        elif isinstance(C,str):
            if C in self.Chan:
                CC = [self.Chan.index(C)]      
            elif C in self.Chan2:
                CC = [self.Chan2.index(C)]
            else: 
                raise ChannelException('Couldn\'t find all your requested C in the TData')
        elif isinstance(C,list) and all([isinstance(c,str) for c in C]):
            CC = []
            for c in C:
                if c in self.Chan:
                    CC.append(self.Chan.index(c))                 
                elif c in self.Chan2:
                    CC.append(self.Chan2.index(c))
                else:
                    raise ChannelException('couldn\'t find channel '+c+' in Session')
        elif isinstance(C,list) and all([(isinstance(c,int) or isinstance(c,str)) for c in C]):
            CC = []
            for c in C:
                if isinstance(c,int):
                    CC.append(c)
                else:
                    if c in self.Chan:
                        CC.append(self.Chan.index(c))                    
                    elif c in self.Chan2:
                        CC.append(self.Chan2.index(c))
                    else:
                        raise ChannelException('couldn\'t find channel '+c+' in Session')
        elif not C:
            CC = []              
        else:
            raise ChannelException('C was not supplied in a supported format!')        
    
        return (TT,FF,MM,ZZ,CC)

    
    def get_segmentation_xfold(self,
                               Segmentation,
                               assumeConstantDims='auto',
                               FieldIDMapList='auto',
                               SaveXFoldPickle=False,
                               LoadFromPickle=False,):
        """
        see XFold.get_segmentation_xfold()
        """
        return self.ParentXFold.get_segmentation_xfold(Segmentation,
                                        assumeConstantDims=assumeConstantDims,
                                        FieldIDMapList=FieldIDMapList,
                                        SaveXFoldPickle=SaveXFoldPickle,
                                        LoadFromPickle=LoadFromPickle)


    def BuildSummary(self):

        summary = ''
        summary += 'Name of session: ' + self.Name + '\n'
        summary += 'No. of TFiles in session: '+str(len(self.TFilesList))+'\n'
        summary += 'TFiles in session: \n'
        [os.path.split(TF.TPath)[1]+'\n' for TF in self.TFilesList]

        summary += '\nSession channels: '+str(self.Chan)+'\n'
        summary += 'No. of time points: '+str(self.NT)+'\n'
        summary += 'No. of fields: '+str(self.NF)+'\n'
        summary += 'No. of montage tiles: '+str(self.NM)+'\n'
        summary += 'No. of z-slices: '+str(self.NZ)+'\n'
        summary += 'No. of channels: '+str(self.NC)+'\n'
        summary += 'Size in Y: '+str(self.NY)+'\n'
        summary += 'Size in X: '+str(self.NX)+'\n'
        return summary


    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)


    def um_2_pixels(self,length,verbose=True):
        """
        Converts the provided length from um to pixels according to the pixel 
        size (in X) of the Session. 

        Parameters
        ----------
        length : int or float
            The length to be converted. 
        verbose : bool
            Whether to print warning that pixel unit is unknown.
        """
        
        if self.pixUnit=='m':
            length2 = length/(self.pixSizeX*1000000)
        elif self.pixUnit=='micron':
            length2 = length/(self.pixSizeX)
        elif self.pixUnit=='um':
            length2 = length/(self.pixSizeX)   
        elif self.pixUnit is None:
            if verbose:
                print('Warning: pixUnit=None so length not converted')
            length2 = length
        else:
            raise Exception(EM.um1 + str(self.pixUnit))
            
        return length2


class TFile:
    """
    This class represents a tiff file but doesn't contain the image data of
    the file. It holds information about what the file has inside and what
    parts of the Session it relates to.

    Parameters
    ----------
    ParentSession : Session
        The Session which this tiff file belongs to.
    TFileN : int
        Where in the ParentSession's TFileList you will find this TFile.
    TPath : str
        The file's path.
    SeshQ with Q=T,F,M,Z,C : list of int
        Each is a list of indices which give the positions within the
        parent Session where the data in the file comes from (along each axis
        T,F,M,Z and C). I.e. SeshT = [2,3] mean the file contains the 3rd and
        4th timepoint of the Session.
    NT,NF,NM,NZ,NC,NY,NX : int
        The sizes of each dimensions of the image data inside the file.
    Chan : list of str
        The names of the channels in your TFile.
    FieldIDs : list of str
        The IDs of each field in the file.

    Note
    ----
    You generally need to look at all files in a Session to figure out the
    SeshQ. So TFiles are currently only created by Session.makeTFiles() which
    does all that search and hands the SeshQ to the __init__. Perhaps in the
    future it would be useful to be able to hand just the ParentSession and
    TPath and then the __init__ could calculate SeshQs?
    """

    def __init__(
        self,
        ParentSession,
        TFileN,
        TPath,
        SeshT,
        SeshF,
        SeshM,
        SeshZ,
        SeshC,
        NT = None,
        NF = None,
        NM = None,
        NZ = None,
        NC = None,
        NY = None,
        NX = None
    ):

        self.ParentSession = ParentSession
        self.ParentXFold = self.ParentSession.ParentXFold
        self.TFileN = TFileN
        self.TPath = TPath
        self.Chan = [c for i,c in enumerate(ParentSession.Chan) if i in SeshC]

        if not all([NT,NF,NM,NZ,NC,NY,NX]):
            nt,nf,nm,nz,nc,ny,nx = fMeta.file2Dims(self.TPath,checkCorrupt=False)
            self.NT = nt
            self.NF = nf
            self.NM = nm
            self.NZ = nz
            self.NC = nc
            self.NY = ny
            self.NX = nx
            self.Shape = (nt,nf,nm,nz,nc,ny,nx)
        else:
            self.NT = NT
            self.NF = NF
            self.NM = NM
            self.NZ = NZ
            self.NC = NC
            self.NY = NY
            self.NX = NX
            self.Shape = (NT,NF,NM,NZ,NC,NY,NX)

        # lists of indices which locate the TFile data within parent session
        self.SeshT = SeshT
        self.SeshF = SeshF
        self.SeshM = SeshM
        self.SeshZ = SeshZ
        self.SeshC = SeshC

        self.FieldIDs = [self.ParentSession.FieldIDMap[f] for f in SeshF]


    def makeTData(self,T='all',F='all',M='all',Z='all',C='all'):
        """ 
        This method creates a specific TData from the TFile.

        Parameters
        -----------
        Q : list or int or 'all' or str
            For Q = T,F,M,Z,C, these are the indices of the images you want to
            take from the TFile along each axis. Either a single index (int)
            or a list or just 'all' data points. With the channel axis you can
            name the channel you want with a 'str'.
        """
        
        usersel = self.parseTFMZC(T,F,M,Z,C)

        NQ = [self.NT,self.NF,self.NM,self.NZ,self.NC]
        for Q,n in zip(userSel,NQ):
            for q in Q:
                assert q in range(n),EM.mt4s        

        if self.MadeBy=='Leica' or self.MadeBy=='Zeiss':
            img = AICSImage(self.TPath)
            seshF0 = sum([s.NF for s in self.ParentXFold.SessionsList[:self.SessionN]])
            data = np.zeros(dimsU,dtype='uint16')
            for f in userSel[1]:
                scene = seshF0+f
                img.set_scene(scene)
                ar = img.get_image_data("TZCYX",T=userSel[0],Z=userSel[3],C=userSel[4])
                ar = ar[:,np.newaxis,np.newaxis,:,:,:,:]
                data[:,f] = ar.copy()
        # Nikon is one file per session, everything in there
        # haven't done montages yet but have done FTZC
        elif self.MadeBy=='Nikon':        
            img = AICSImage(self.TPath)
            data = np.zeros(dimsU,dtype='uint16')
            for f in userSel[1]:
                img.set_scene(f)
                ar = img.get_image_data("TZCYX",T=userSel[0],Z=userSel[3],C=userSel[4])
                ar = ar[:,np.newaxis,np.newaxis,:,:,:,:]
                data[:,f] = ar.copy()                 
        else:       
            # make an itertools product of the user selected indices
            prod = product(*userSel)
            with tifffile.TiffFile(self.TPath) as tif:
                pageIndices = [genF.unravel(*X,*self.Shape[0:5]) for X in prod]
                data = tif.asarray(key=pageIndices)

        dims = tuple([len(x) for x in userSel]+[self.NY]+[self.NX])
        data.shape = dims

        allSQ = [self.SeshT,self.SeshF,self.SeshM,self.SeshZ,self.SeshC]
        seshQ = [[SQ[q] for q in Q] for Q,SQ in zip(userSel,allSQ)]

        return TData([self],data,*seshQ,self.ParentSession)


    def BuildSummary(self):
        mb = self.ParentSession.MadeBy
        summary = ''
        summary += 'TFile name: '
        summary += genF.stripTags(os.path.split(self.TPath)[1],mb,self.CustomTags)+'\n'
        summary += 'From session: ' + self.ParentSession.Name + '\n'
        summary += 'TFile path: '+ self.TPath+'\n\n'
        summary += 'No. of time points: '+str(self.NT)+'\n'
        summary += 'No. of fields: '+str(self.NF)+'\n'
        summary += 'No. of montage tiles: '+str(self.NM)+'\n'
        summary += 'No. of z-slices: '+str(self.NZ)+'\n'
        summary += 'No. of channels: '+str(self.NC)+'\n'
        summary += 'Size in Y: '+str(self.NY)+'\n'
        summary += 'Size in X: '+str(self.NX)+'\n'
        return summary


    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)


class TData:
    """
    This class holds the actual image data as a numpy.array of dtype uint16.
    So these are the only objects in this package which take a lot of memory.
    The dimension ordering is always:
    (times,fields,montages,zslices,channels,ypixels,xpixels)
    Generally a TData is made by TFile.makeTData() or Session.makeTData().

    Attributes
    ----------
    ParentTFiles : list of SessionStitcher.TFile
        The TFiles used to derive this TData.
    ParentSession : SessionStitcher.Session object
        The Session that the TData is derived from.
    SessionN : int
        The index of the TData's parent session within the parent XFold's
        SessionsList.
    data : numpy.array
        The image data as uint16 numpy array with 7 dimensions.
    SeshQ : list
        For Q=T,F,M,Z and C. These are the indices in the Session that the
        data corresponds to. They are lists containing all the indices.
    FieldIDs : list of str
        The ID for each field in the data.
    Chan : list of str
        The names of channels after having been regularised to standard
        channel names by genF.chanDic(). Note this is different to 
        Session.Chan where Sessions.Chan is not regularised, Sessions.Chan2 is.
    Chan2 : list of str
        The names of channels before having been regularised to standard
        channel names by genF.chanDic(). !! other way round to 
        Session.Chan2... stupid historical reasons !!   
    Times : dict
        For each field and time-point in the data, this provides a time,
        defined as time since XFold.StartTimes[field]. The keys are fields and
        the values are a list of int times in min, one for each time point in
        the TData.
    Aligned : False or str or dict
        False if this tdata hasn't yet been processed with TData.AlignExtract. 
        If it has then this is set to str or dict that you provided the align 
        templates with. Then this is set to false again if you do anything 
        that means the tdata doesn't match this alignment anymore 
        (e.g. rotate/crop/resize...)
    allMeta : dict
        Just a dictionary of repeated metadata, convenient for transferring 
        everything to functions.
    newSeshNQ : int
        A recording of if a function has changed a tdata.data dimension so 
        that the overall saved Session can't possibly be aligned with the 
        original Session because of a change in Shape. E.g. z-projection, 
        montage stitching, new channel (e.g. Segmentation).
    ZStep : float
        The distance between z-slices in um.
    """

    def __init__(
        self,
        ParentTFiles,
        data,
        SeshT,
        SeshF,
        SeshM,
        SeshZ,
        SeshC,
        ParentSession
    ):

        self.ParentTFiles = ParentTFiles
        self.ParentSession = ParentSession
        self.SessionN = self.ParentSession.SessionN
        self.ParentXFold = self.ParentSession.ParentXFold
        self.data = data
        self.Shape = None
        self.pixSizeY = self.ParentSession.pixSizeY
        self.pixSizeX = self.ParentSession.pixSizeX
        self.pixUnit = self.ParentSession.pixUnit

        self.ZStep = self.ParentSession.ZStep       

        self.SeshT = SeshT
        self.SeshF = SeshF
        self.SeshM = SeshM
        self.SeshZ = SeshZ
        self.SeshC = SeshC

        self.SeshQ = [self.SeshT,self.SeshF,self.SeshM,self.SeshZ,self.SeshC]

        # the c!=None here is just important for self.Duplicate()... sometimes 
        # there are None in self.SeshC which would otherwise cause problems
        self.Chan = [self.ParentSession.Chan[c] for c in SeshC if c!=None]
        
        self.Chan2 = self.Chan.copy()
        self.Chan = [genF.chanDic(c) for c in self.Chan]
        self.startChan = tuple(self.Chan)

        self.FieldIDs = [self.ParentSession.FieldIDMap[f] for f in self.SeshF]

        self.Times = self.getTimes()

        self.TemplatePadDic = {} # see alignExtract()
        self.Aligned = False

        self.updateDimensions() # sets NT,NF,NM,NZ,NC,NY,NX,Shape        

        self.allMeta = {
                         'NT':self.NT,
                         'NF':self.NF,
                         'NM':self.NM,
                         'NZ':self.NZ,
                         'NC':self.NC,
                         'NY':self.NY,
                         'NX':self.NX,
                         'SeshT':self.SeshT,
                         'SeshF':self.SeshF,
                         'SeshM':self.SeshM,
                         'SeshZ':self.SeshZ,
                         'SeshC':self.SeshC,
                         'Chan':self.Chan,
                         'FieldIDs':self.FieldIDs,
                         'SessionN':self.SessionN,
                         'Shape':self.Shape,
                         'ZStep':self.ZStep,
                         'Times':self.Times               
                        } 
        
        # not NF or NT yet but should add later perhaps
        # but NF is complicated because of FieldIDs and less normal that if NF=1 then newSeshNF=1
        self.newSeshNT = None
        self.newSeshNF = None
        self.newSeshNM = None
        self.newSeshNZ = None
        self.newSeshNC = None
        self.newSeshNY = None
        self.newSeshNX = None
        self.newSeshNQ = [self.newSeshNT,
                          self.newSeshNF,
                          self.newSeshNM,
                          self.newSeshNZ,
                          self.newSeshNC,
                          self.newSeshNY,
                          self.newSeshNX]        


    def getTimes(self):
        """
        Gets the 'experiment time' of each field and time-point of the tdata.
        I.e. the time from the xfold-defined start times (XFold.StartTimes).

        Returns
        -------
        times : dict
            The keys are fields and the values are a list of int times in min
            from the fields start time, one for each time-point in the TData.
        """
        times = {}
        for f in self.FieldIDs:
            ts = [self.ParentSession.TStep*t for t in self.SeshT]
            ts = [self.ParentSession.StartMom + t for t in ts]
            ts = [t - self.ParentXFold.StartTimes[f] for t in ts]
            times[f] = [int(t.total_seconds()/60) for t in ts]
        return times


    def updateDimensions(self):
        """
        This updates the record of the dimensions of the TData according
        to the shape that numpy finds. This shape is already the one we want
        because makeTData() does it for us.
        """
        dims = self.data.shape
        self.NT = dims[0]
        self.NF = dims[1]
        self.NM = dims[2]
        self.NZ = dims[3]
        self.NC = dims[4]
        self.NY = dims[5]
        self.NX = dims[6]
        self.Shape = dims
        self.allMeta = {
                         'NT':self.NT,
                         'NF':self.NF,
                         'NM':self.NM,
                         'NZ':self.NZ,
                         'NC':self.NC,
                         'NY':self.NY,
                         'NX':self.NX,
                         'SeshT':self.SeshT,
                         'SeshF':self.SeshF, 
                         'SeshM':self.SeshM,
                         'SeshZ':self.SeshZ,
                         'SeshC':self.SeshC,
                         'Chan':self.Chan,
                         'FieldIDs':self.FieldIDs,
                         'SessionN':self.SessionN,
                         'Shape':self.Shape,
                         'ZStep':self.ZStep,
                         'Times':self.Times               
                        }        

    def parseTFMZC(self,T='all',F='all',M='all',Z='all',C='all'):
        """
        This takes FTMZC in all accepted formats (see below) and converts to 
        indices for that tdata.
        
        It accepts 'all', int, -1 and list of int. For C it accepts channel 
        names. It accepts C either converted by chanDic or not: it checks for 
        non-converted in self.Chan2 first, then checks in self.Chan. Note that 
        asking for converted version might be easier (i.e. you can type DAPI 
        without having to remember what the original channel was called) but 
        if there happens to be another channel that coverts to it (i.e. another
        'DAPI') you could end up with the wrong one! For F it accepts FieldID 
        also.

        One more feature is that in Chan2 it also seaches for 
        'Segmentation_'+C because of the special case that 
        Session.makeTData(Segmentation=C) will add 'Segmentation_' to 
        self.Chan2 so that chanDic converts it to 'Segmentation'.
        """
        if T=='all':
            TT = list(range(self.NT))
        elif T==-1 or T==[-1]:
            TT = [self.NT - 1]        
        elif isinstance(T,list) and all([isinstance(t,int) for t in T]):
            TT = T.copy()     
        elif isinstance(T,int):
            TT = [T]   
        elif not T:
            TT = []              
        else:
            raise Exception('T was not supplied in a supported format!')    
            
        if F=='all':
            FF = list(range(self.NF))
        elif F==-1 or F==[-1]:
            FF = [self.NF - 1]
        elif isinstance(F,int):
            FF = [F]
        elif isinstance(F,list) and all([isinstance(f,int) for f in F]):
            FF = F.copy()
        elif isinstance(F,str):        
            assert f in self.FieldIDs,'Couldn\'t find all your requested F in the TData.FieldIDs'
            FF = [self.FieldIDs.index(F)]
        elif isinstance(F,list) and all([isinstance(f,str) for f in F]):
            check = all([f in self.FieldIDs for f in F] )
            assert check,'Couldn\'t find all your requested F in the TData.FieldIDs'
            FF = [self.FieldIDs.index(f) for f in F] 
        elif not F:
            FF = []              
        else:
            raise Exception('F was not supplied in a supported format!')    
            
        if M=='all':
            MM = list(range(self.NM))
        elif M==-1 or M==[-1]:
            MM = [self.NM - 1]           
        elif isinstance(M,list) and all([isinstance(m,int) for m in M]):
            MM = M.copy()    
        elif isinstance(M,int):
            MM = [M]    
        elif not M:
            MM = []              
        else:
            raise Exception('M was not supplied in a supported format!')        
            
        if Z=='all':
            ZZ = list(range(self.NZ))    
        elif Z==-1 or Z==[-1]:
            ZZ = [self.NZ - 1]        
        elif isinstance(Z,list) and all([isinstance(z,int) for z in Z]):
            ZZ = Z.copy()    
        elif isinstance(Z,int):
            ZZ = [Z]  
        elif not Z:
            ZZ = []              
        else:
            raise Exception('Z was not supplied in a supported format!')        
            
        if C=='all':
            CC = list(range(self.NC))
        elif C==-1 or C==[-1]:
            CC = [self.NC - 1]        
        elif isinstance(C,list) and all([isinstance(c,int) for c in C]):
            CC = C.copy()     
        elif isinstance(C,int):
            CC = [C]        
        elif isinstance(C,str):
            if C in self.Chan2:
                CC = [self.Chan2.index(C)]
            elif 'Segmentation_'+C in self.Chan2:
                CC = [self.Chan2.index('Segmentation_'+C)]
            elif C in self.Chan:
                CC = [self.Chan.index(C)]
            else: 
                raise ChannelException('Couldn\'t find all your requested C in the TData')
        elif isinstance(C,list) and all([isinstance(c,str) for c in C]):
            CC = []
            for c in C:
                if c in self.Chan2:
                    CC.append(self.Chan2.index(c))
                elif 'Segmentation_'+c in self.Chan2:
                    CC.append(self.Chan2.index('Segmentation_'+c))
                elif c in self.Chan:
                    CC.append(self.Chan.index(c))
                else:
                    raise ChannelException('couldn\'t find channel '+c+' in Session')  
        elif isinstance(C,list) and all([(isinstance(c,int) or isinstance(c,str)) for c in C]):
            CC = []
            for c in C:
                if isinstance(c,int):
                    CC.append(c)
                else:
                    if c in self.Chan2:
                        CC.append(self.Chan2.index(c))
                    elif 'Segmentation_'+c in self.Chan2:
                        CC.append(self.Chan2.index('Segmentation_'+c))                       
                    elif c in self.Chan:
                        CC.append(self.Chan.index(c))
                    else:
                        raise ChannelException('couldn\'t find channel '+c+' in Session') 
        elif not C:
            CC = []  
        else:
            raise ChannelException('C was not supplied in a supported format!')        
    
        return (TT,FF,MM,ZZ,CC)

    
    def TakeSubStack(self,
                     T='all',
                     F='all',
                     M='all',
                     Z='all',
                     C='all',
                     updateNewSeshNQ=True):
        """
        This method takes subsections of dimensions of the TData.
        The selection is indices wrt the TData, i.e. nothing to do with the 
        session.

        Parameters
        ----------
        Q : anything accepted by self.parseTFMZC()
            The parts of each axis you want to take.
        updateNewSeshNQ : bool
            If you want to later reload the data saved in this routine you will
            need to tell it the overall dimensions the new Session has. So if 
            your use of TakeSubstack() gives the data dimensions that will be 
            used globally in the saved data then you should set this to True!
        """

        userSel = self.parseTFMZC(T,F,M,Z,C)
        userSel = userSel + (range(self.NY),range(self.NX))
        self.data = self.data[np.ix_(*userSel)].copy()

        # update everything
        self.SeshT = [self.SeshT[q] for q in userSel[0]]
        self.SeshF = [self.SeshF[q] for q in userSel[1]]
        self.SeshM = [self.SeshM[q] for q in userSel[2]]
        self.SeshZ = [self.SeshZ[q] for q in userSel[3]]
        self.SeshC = [self.SeshC[q] for q in userSel[4]]
        self.FieldIDs = [self.FieldIDs[q] for q in userSel[1]]
        self.Chan = [self.Chan[c] for c in userSel[4]]
        self.Chan2 = [self.Chan2[c] for c in userSel[4]]
        self.updateDimensions()
        if updateNewSeshNQ:
            if not C=='all':
                self.newSeshNC = self.NC
            if not Z=='all':
                self.newSeshNZ = self.NZ
            if not M=='all':
                self.newSeshNM = self.NM


    def AddArray(self,array,Chan_name='channel',axis=4):
        """
        Adds the array to the TData.data. Just adds to channel dimension so 
        far.

        Parameters
        ----------
        array : array_like
            The array that you are adding to the TData.
        Chan_name : str
            The name of the channel, must be in genF.chanDic.
        """
        assert axis==4,'Only implemented for channel addition so far.'
        self.data = np.concatenate((self.data,array),axis=4)
        self.SeshC.append(None)
        self.Chan2.append(Chan_name)
        self.Chan.append(genF.chanDic(Chan_name))
        self.updateDimensions()  
        self.newSeshNC = self.NC

    
    def MatchChannels(self,endChans,verbose=False):
        """
        Matches tdata channels (as well as ordering) to the user provided
        tuple of channels. Will add blanks images for channels that are in
        endChans but not in the data.

        Parameter
        ---------
        endChans : tuple or None or 'Auto'
            Output data will have these channels. If None or 'Auto' then it
            will get set of all channels that appear in all sessions and order
            according to XFold.chanOrder.
        """
        if np.prod(np.array(self.Shape))==0:
            return

        if endChans == None or endChans == 'Auto':
            pxfold = self.ParentXFold
            allSesh = [c for s in pxfold.SessionsList for c in s.Chan]
            endChans = set(allSesh)
            endChans = [genF.chanDic(c) for c in endChans]
            # sort them according to class variable chanOrder
            endChans = sorted([[XFold.chanOrder[c],c] for c in endChans])
            endChans = tuple([j for i,j in endChans])

        if verbose:
            print(endChans)

        # first add a blank channel at the end of the channels
        padTuple = ((0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0))
        self.data = np.pad(self.data,padTuple)
        # make foundChan list: position of endChan in self.chan
        foundChan = []
        for endChan in endChans:
            if endChan in self.Chan:
                foundChan.append(self.Chan.index(endChan))
            else:
                foundChan.append(-1)

        # reorder tifdata channels according to foundChannels
        self.data = self.data[:,:,:,:,foundChan,:,:].copy()

        self.Chan = list(endChans)
        self.updateDimensions()
        self.SeshC = [None if c==-1 else c for c in foundChan]

        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')


    def DownSize(self,downsize=None,verbose=False):
        """
        Reduces downsizes the image data in x and y.

        Parameters
        ----------
        downsize : int or list
            If int then downsize x and y by this factor.
            If list [y,x] then downsize by different factors y and x.
        """
        if downsize==1 or downsize==[1,1]:
            downsize = None
        if np.prod(np.array(self.Shape))==0 or not downsize:
            return

        if isinstance(downsize,int):
            downsize = [downsize,downsize]

        self.data = downscale_local_mean(self.data,(1,1,1,1,1,*downsize))
        self.data = self.data.astype('uint16')
        
        if self.pixSizeY:
            self.pixSizeY = self.pixSizeY/downsize[0]
        if self.pixSizeX:
            self.pixSizeX = self.pixSizeX/downsize[1]

        self.updateDimensions()            
        
        self.Aligned = False

        if verbose:
            print('Downsized by: ',str(downsize))

        print('Warning: we havent updated self.newNQ in this function yet so saving and reloading might have troubles.')


    def EmptyQ(self):
        """Return True for all 0 value pixels or no pixels, False otherwise."""
        if np.prod(np.array(self.Shape))==0:
            return True
        else:
            return not self.data.any()


    def DeleteEmptyT(self,verbose=False,hideWarning=False,
                     file='.sessionstitcher'):
        """
        Delete any time points that don't contain data.

        Notes
        -----
        Best not to use on TDatas with > 1 fields, see the warning printed in
        code for explanation.
        """

        if np.prod(np.array(self.Shape))==0:
            return

        if self.NF>1 and not hideWarning:
            print(EM.de1)

        # find which time points have anything other than all zeros:
        nonEmpT = [self.data[t].any() for t in range(self.NT)]

        # save info to meta file
        if self.NF==1 and file:
            file = os.path.join(self.ParentXFold.XPathP,file)
            with open(file,'a') as theFile:
                for t in range(self.NT):
                    s = self.SessionN
                    x = [s,self.SeshT[t],self.SeshF[0]]
                    if nonEmpT[t] and x not in self.ParentXFold.nonBlankTPs:
                        line = '\nnonBlankTP ['+str(s)
                        line += ','+str(self.SeshT[t])+','+str(self.SeshF[0])+']'
                        theFile.write(line)
                    elif not nonEmpT[t] and x not in self.ParentXFold.blankTPs:
                        line = '\nblankTP ['+str(s)
                        line += ','+str(self.SeshT[t])+','+str(self.SeshF[0])+']'
                        theFile.write(line)

        # keep only nonEmpty
        self.data = self.data[nonEmpT].copy()

        self.updateDimensions()
        self.SeshT = [x for x,q in zip(self.SeshT,nonEmpT) if q]

        if verbose:
            print('deleted empty time points')
            
        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')




    def Homogenise(self,HFiles={},chan='All',verbose=False):
        """
        This corrects for inhomogenous field of view sensitivities by
        subtracting a provided 'dark field' image and dividing by a provided
        'flat field' image.

        Parameters
        -----------
        HFiles : str or dict
            The locations of the images used for homogenisation. See
            XFold.buildHomogFilts() for format. If empty list then it will
            look to XFold.HomogFiltDic.
        chan : 'All' or list of ints or str
            If 'All' then all channels will be homogenised. If list then the
            ints or str provide which channels to homogenise.

        Notes
        -----
        It will fail if the filters are not the same size as the TData so you
        have to downsize your filters if you have downsized your TData etc.
        """

        if np.prod(np.array(self.Shape))==0:
            return

        assert HFiles or self.ParentXFold.HomogFiltDic,EM.hf1

        if not HFiles:
            HFilts = self.ParentXFold.HomogFiltDic
        else:
            HFilts = {}
            if isinstance(HFiles,str):
                if not os.path.split(HFiles)[0]:
                    HFiles = os.path.join(self.ParentXFold.XPathP,HFiles)
                fps = os.listdir(HFiles)
                HF = ['_FF.tif','_DF.tif']
                hNames = [c+f for c in XFold.chanOrder.keys() for f in HF]
                for h in hNames:
                    if h in fps:
                        if h[:-7] not in HFilts.keys():
                            HFilts[h[:-7]] = {}
                        im = io.imread(os.path.join(HFiles,h)).astype('float32')
                        HFilts[h[:-7]][h[-6:-4]] = im
            elif isinstance(HFiles,dict):
                for k,v in HFiles.items():
                    if k not in HFilts.keys():
                        HFilts[k] = {}
                    for k2,v2 in v.items():
                        if isinstance(v2,str):
                            im = io.imread(v2).astype('float32')
                            HFilts[k][k2] = im
                        elif isinstance(v2,np.ndarray):
                            HFilts[k][k2] = v2
                        else:
                            raise Exception('HFiles format not correct.')
            else:
                raise Exception('HFiles must be a str or dict')

        if isinstance(chan,str) and chan!='All':
            chan = [chan]
        if chan=='All':
            chan = [i for i in range(len(self.Chan))]
        elif isinstance(chan,list):
            if all([isinstance(c,str) for c in chan]):
                chan = [self.Chan.index(c) for c in chan]
            elif not all([isinstance(c,int) for c in chan]):
                raise Exception('chan not in recognisable format.')
        else:
            raise Exception('chan not in recognisable format.')


        for c in chan:
            C = self.Chan[c]
            if C not in HFilts.keys():
                raise Exception(f'No filter found for {C} channel.')
            elif 'FF' not in  HFilts[C].keys():
                raise Exception(f'No FF filter found for {C} channel.')
            elif HFilts[C]['FF'].shape!=(self.NY,self.NX):
                raise Exception('Filter doesn\'t match the data shape.')
            elif 'DF' in HFilts[C].keys():
                if HFilts[C]['DF'].shape!=(self.NY,self.NX):
                    raise Exception('Filter doesn\'t match the data shape.')

        # loop over image in self.data
        dims = [self.NT,self.NF,self.NM,self.NZ,self.NC]
        ranges = map(range,dims)
        for t,f,m,z,c in product(*ranges):
            if c not in chan: # skip if this isn't a chosen channel
                continue
            C = self.Chan[c]
            # create a float32 of just that one image for division:
            _data = self.data[t,f,m,z,c].astype('float32')
            if 'DF' in HFilts[C].keys():
                _data = _data - HFilts[C]['DF']
                _data[_data<0] = 0 # otherwise negative gives overflow problem!
                _data = _data/HFilts[C]['FF']
            else:
                _data = _data/HFilts[C]['FF']
            # add to count if >=1 pix has become bigger than UINT16MAX
            self.ParentXFold.HomogOverflow[0] += 1
            if _data.max() > self.ParentXFold.UINT16MAX:
                self.ParentXFold.HomogOverflow[1] += 1
            # convert back to uint16 to put in self.data
            self.data[t,f,m,z,c] = _data.astype('uint16')
            del _data

        if verbose:
            print('homogenised')

            
    
    def BaSiCHomogenise(self):
        
        """
        This corrects for inhomogenous field of view using BaSiCPy.

        Note how it calculates the filters using all images of each channel 
        - i.e. currently there isn't much control and you might want to add 
        control if it is slow for big tdatas.
        """
        
        for c in range(self.NC):
            dd = self.NT*self.NF*self.NM*self.NZ
            _data = self.data[:,:,:,:,c].reshape((dd,self.NY,self.NX))
            basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)
            basic.fit(_data)
            self.data[:,:,:,:,c] = basic.transform(self.data[:,:,:,:,c])
        
        return
    
    



    def zProject(self,
                 meth='maxProject',
                 downscale=None,
                 slices=1,
                 fur=False,
                 chan=None,
                 proj_best=True,
                 sliceBySlice=False,
                 meth_fun=signalF,
                 *args,
                 verbose=False):
        """
        This does z-projection of the data.

        Parameters
        ----------
        meth : {'maxProject', 'avProject','minProject',
                'signalDetect'}
            basic ones: maxProject,avProject,minProject
            Or it can do a fancy one that finds the layer with most features.
        downscale : int
            Just used if meth=signalDetect - how much to downscale the images
            by when calculating which slices have signal... svaes lots of time.
        slices : int
            How many slices to return in the signalDetect method.
        fur : bool
            Whether to return the slice furthest from the signal (e.g. for
            pulling out a background measure).
        chan : str or int
            For signalProject this calculates the slices based on just the
            channel you give here. You can give as the channel name or the
            index in stack. If None then all are calculated separately, but 
            remember not to put False because it will interpret this as the 
            integer 0 because python is broken like that.
        proj_best : bool
            In signal detect version, whether to do a final average projection 
            once you have done your signal projection.
        sliceBySlice : bool
            If True then the same z-slice is chosen across all x-y. I.e. we just 
            pick the slice with most signal rather than the filter method that 
            picks the best z for each pixel based on its surroundings. Only 
            relevant for signalProj.
        meth_fun : python function
            Function to pass to measure 'signal' (would be better called 
            sharpness) in image if using signalDetect method.
        *args : variable
            Arguments to pass to meth_fun

        Example
        ------
        tdata.zProject('signalDetect',1,1,False,None,True,True,
        ms.zProj.tenengrad,feature_size_pixels,signal_to_noise_factor)

        Here we import the tenengrad function to use for signal detection. 
        This is a standard function for measuring sharpness and requires the 
        feature_size_pixels to define the size of area over which gradients 
        are calculated - choose something that captures the size of feature 
        that is sharp in the good focus. We have added a feature that excludes 
        regions that are 'signal_to_noise_factor' times higher pixel intensity 
        than the image median intensity - this stops very bright rubbish spots 
        ruining the focal plane detection.
        
        """

        if np.prod(np.array(self.Shape))==0 or self.NZ==1:
            return
        dims = [self.NT,self.NF,self.NM,1,self.NC,self.NY,self.NX]
        if slices!=1:
            dims[3] = slices+1
        dims = tuple(dims)
        _data = np.zeros(dims,dtype='uint16')

        ranges = map(range,[self.NT,self.NF,self.NM,self.NC])
        ranges2 = map(range,[self.NT,self.NF,self.NM])
        # method1: maximum projection
        if meth == 'maxProject':
            for t,f,m,c in product(*ranges):
                _data[t,f,m,0,c] = maxProj(self.data[t,f,m,:,c])
        # method2: the signal detection one I made:
        elif meth == 'signalDetect':
            assert isinstance(downscale,int),EM.zp1
            p = self.pixSizeX
            if isinstance(chan,int) or isinstance(chan,str):
                if isinstance(chan,str):
                    chan = self.Chan.index(chan)
                for t,f,m in product(*ranges2):
                    stack2 = findSliceSelection(self.data[t,f,m,:,chan],
                                                p,
                                                downscale,
                                                fur,
                                                sliceBySlice,
                                                meth_fun,
                                                *args)
                    for c in range(self.NC):
                        _data[t,f,m,:,c] = takeSlicesSelection(stack2,
                                                               self.data[t,f,m,:,c],
                                                               slices,
                                                               proj=proj_best,
                                                               sliceBySlice=sliceBySlice)
            else:
                for t,f,m,c in product(*ranges):
                    _data[t,f,m,:,c] = signalProj(self.data[t,f,m,:,c],
                                                  p,
                                                  downscale,
                                                  slices,
                                                  proj_best,
                                                  fur,
                                                  sliceBySlice,
                                                  meth_fun,
                                                  *args)
        elif meth == 'avProject':
            for t,f,m,c in product(*ranges):
                _data[t,f,m,0,c] = avProj(self.data[t,f,m,:,c])
        elif meth == 'minProject':
            for t,f,m,c in product(*ranges):
                _data[t,f,m,0,c] = minProj(self.data[t,f,m,:,c])
        else:
            raise Exception('z-project method not recognised')

        self.data = _data.copy()
        del _data
        self.SeshZ = [None]*slices
        self.SeshQ[3] = [None]*slices
        self.updateDimensions()                     
        self.newSeshNZ = slices

        if verbose:
            print('z-projected')


    def qProject(self,axis):
        """ The idea of this is to do projection just like zProject() but over 
        any axis. Making a simple version for now...

        Parameters
        ----------
        axis : int
            The index of the axis you want to project according to list 
            [T,F,M,Z,C,Y,X].
        """
        self.data = np.max(self.data,axis=axis,keepdims=True)
        self.updateDimensions()
        if axis<5:
            self.SeshQ[axis] = [None]
            attributes = ['SeshT','SeshF','SeshM','SeshZ','SeshC']
            setattr(self,attributes[axis],[None])            
        self.newSeshNQ[axis] = 1
        attributes2 = ['newSeshNT',
                       'newSeshNF',
                       'newSeshNM',
                       'newSeshNZ',
                       'newSeshNC',
                       'newSeshNY',
                       'newSeshNX']
        setattr(self,attributes2[axis],1)
    

    def LocalIntensity(self,
                       downscale=False,
                       radius=10,
                       mode='step',
                       retFloat=False):
        """
        This replaces each pixel by the average value of the surrounding
        region. It removes (and later re-adds) padding so that that doesn't
        affect the image edges.

        Parameters
        ----------
        downscale : int
            Factor to downscale everything first to speed up. (It will be
            resized to original size).
        radius : int
            The radius of the structure element in um.
        retFloat : bool
            Whether to return a as float32 instead of normal uint16.
        """
        if np.prod(np.array(self.Shape))==0:
            return

        if downscale:
            radius = np.ceil(radius/(self.pixSizeY*downscale))
        else:
            radius = np.ceil(radius/(self.pixSizeY))
        selem = disk(radius)

        if retFloat:
            self.data = self.data.astype('float32')

        for f in range(self.NF):
            p = self.TemplatePadDic[self.FieldIDs[f]]
            _,(pt,b,l,r) = genF.removePads(self.data[0,f,0,0,0],p,retPad=True)

            dims = (self.NT,self.NM,self.NZ,self.NC)
            ranges = map(range,dims)
            for t,m,z,c in product(*ranges):
                data = self.data[t,f,m,z,c,pt:self.NY-b,l:self.NX-r]
                if downscale:
                    data = downscale_local_mean(data,(downscale,downscale))

                if mode=='step':
                    data = generic_filter(data,
                                          lambda x:np.mean(x),
                                          footprint=selem,
                                          mode='reflect')
                elif mode=='gaussian':
                    data = gaussian_filter(data,radius)
                else:
                    raise Exception('Unrecognised mode')

                if downscale:
                    resize(data,(self.NY-pt-b,self.NX-l-r))
                self.data[t,f,m,z,c] = np.pad(data,((pt,b),(l,r)))


    def Rotate(self,degree):
        """
        This rotates all the images in a TData. It doesn't change the size of
        the image to account for the rotation. Uses fast open cv.

        Parameters
        -----------
        degree : float
            The degrees rotation clockwise that you want to apply.
        """
        sh = self.Shape
        sh1 = (self.NX,self.NY)
        image_center = (self.NX/2,self.NY/2)
        M = cv.getRotationMatrix2D(image_center,degree,1)
        lenDat = self.NT*self.NF*self.NM*self.NZ*self.NC
        sh2 = (self.NY,self.NX,lenDat)
        self.data = np.moveaxis(np.moveaxis(self.data,6,0),6,0).reshape(sh2)

        _data = []
        for i in range(lenDat//512):
            _data.append(cv.warpAffine(self.data[:,:,i*512:(i+1)*512],M,sh1))
        _data.append(cv.warpAffine(self.data[:,:,(lenDat//512)*512:],M,sh1))
        self.data = np.concatenate(_data,axis=2)
        self.data = np.moveaxis(np.moveaxis(self.data,0,-1),0,-1).reshape(sh)
        
        self.Aligned = False


    def Reflect(self,reflectX=False,reflectY=False):
        if reflectX:
            self.data = np.flip(self.data,axis=6)
        if reflectY:
            self.data = np.flip(self.data,axis=5)
            
        self.Aligned = False


    def StitchIt(self,
                 method='noAlign',
                 chans=False,
                 returnCens=False,
                 cens=None,
                 verbose=False,
                 flipYQ=False,
                 printWarnings=True):
        """
        Stitches together grid-like montage tiles in axis i=2 of the TData.

        Parameters
        ----------
        method : str
            Put 'noAlign' to use guess from metadata instead of
            cross-correlation.
        chans : list
            The channels names of the channels you want it to use in the
            cross-correlation. If False then it searches for all fluorescent
            channels.
        returnCens : bool
            Whether to return the centres that you found so they can be used
            on another equivalent TData.
        cens : list of numpy.ndarray
            centres found previously to use instead of doing the search.
        flipYQ : bool
            Opera requires tiles to be flipped vertically for stitching

        Notes
        -----
        The parameters you give find here are in the form:
        [threshold,ampPower,maxShiftFraction,boxsize,sdt]

        threshold - it searches for signal and does 'auto' aligning if
                    not enough. This threshold defines 'enough signal'
        ampPower - the image is raised to this power at somepoint to
                    amplify signal could increase for increase sensitivity?
        maxShiftFraction - maximum detected that is applied, as a fraction of
                            image size b/c if a big shift is detected it is
                            probably wrong if detected shift is bigger it does
                            'auto' aligning
        boxsize - size of the box (in pixels) used in measuring
                    signal with sobel
        sdt - standard dev of gaussian blur applied during
                signal measuring
        minSize - for images smaller than this it won't do cross-correlation
                    alignment because the small size risks stupid results.

        Since we are finding the best alignments for each tile, we may end up
        with different sized images, both in the TData and across the whole
        XFold. To avoid this the stitch functions does padding and/or cropping
        to the final image so that you can define beforehand what size output
        you want. To give the same size across a whole XFold, this looks to
        the parent XFold to define final size.
        """

        if self.NM==1 or np.prod(np.array(self.Shape))==0:
            return

        # find if TData has been downsized to help define final size
        NY = self.ParentSession.NY
        NX = self.ParentSession.NX
        downX = round(NX/self.NX)
        downY = round(NY/self.NY)

        # decide what the output size will be
        # get all the values we need from all sessions in the XFold
        sessionsList = self.ParentXFold.SessionsList
        allOverlaps = [s.MOlap for s in sessionsList]
        allNY = [s.NMY for s in sessionsList]
        allNX = [s.NMX for s in sessionsList]
        allSizeY = [s.NY/downY for s in sessionsList]
        allSizeX = [s.NX/downX for s in sessionsList]
        # zip all these together
        allZipY = zip(allOverlaps,allNY,allSizeY)
        allZipX = zip(allOverlaps,allNX,allSizeX)

        # the size in x or y, call it q, would be given by:
        # (no. of tiles in Q)*(no. of pixels in Q) - overlapping part
        # with overlapping part given by: (N tiles - 1)*(N pixels)*overlap
        # we divide the overlap by 2 to give a number that is an overestimate
        # i.e. we hope we will always have some black padding at the edges:
        ySizeOut = [NMY*NY-(NMY-1)*NY*OL/2 for OL,NMY,NY in allZipY]
        xSizeOut = [NMX*NX-(NMX-1)*NX*OL/2 for OL,NMX,NX in allZipX]

        # we expect these to all be the same
        if len(set(ySizeOut)) != 1 or len(set(xSizeOut)) != 1:
            self.ParentXFold.Warnings.append(EM.si1)
        ySizeOut = int(max(ySizeOut))
        xSizeOut = int(max(xSizeOut))
        shapeOut = (ySizeOut,xSizeOut)

        # parameters to send to findCentres() (see __doc__)
        threshold = 0.04
        ampPower = 3
        maxShiftFraction = self.ParentSession.MOlap/2
        boxsize = 70
        sdt = 4
        minSize = 150
        cnts = self.ParentXFold.StitchCounts
        pars = [threshold,ampPower,maxShiftFraction,boxsize,sdt,minSize,cnts]

        # print a warning if we are skipping alignment due to small tiles
        if self.NY<minSize or self.NX<minSize:
            self.ParentXFold.Warnings.append(EM.si2 % minSize)

        xMon = self.ParentSession.NMX
        yMon = self.ParentSession.NMY
        olap = self.ParentSession.MOlap

        # reorder M-axis if it's not the normal LRUP ordering
        # I guess you don't worry about tdata.SeshM because NM=1 after all this
        if self.ParentSession.MontageOrder == 'UDRL':
            reshape = genF.UDRL2LRUD(list(range(self.NM)),yMon,xMon)
            self.data = self.data[:,:,reshape]
        elif self.ParentSession.MontageOrder == 'TilePos':
            reshape = genF.sort_grid_points(self.ParentSession.TilePosX,self.ParentSession.TilePosY)
            self.data = self.data[:,:,reshape]
        if flipYQ:
            self.data = self.data[:,:,:,:,:,-1::-1,:].copy()

        # only want to align using these channels
        # i.e. no BF because cross-correlation doesn't work well
        if chans:
            fluoChans = chans
        else:
            fluoChans = ['YFP','CFP','RFP','GFP','FR','DAPI']
        # remove possible blank channels added by matchChannels
        fluoChans = [c for c in fluoChans if c in self.startChan]
        aCh = []
        for c in fluoChans:
            if c in self.Chan:
                aCh.append(self.Chan.index(c))
        if aCh==[]:
            self.ParentXFold.Warnings.append(EM.si3)
            method += 'noAlign'
            aCh.append(0)

        if verbose:
            print('aCh: ',aCh)
        
        # initiate arrays to send to findCentres
        # sigIms is zproj of 1 time, 1 field and only chans of aCh
        # also centreLists storage array: one list (len=NM) for each t,f-point
        sigDims = (self.NM,len(aCh),self.NY,self.NX)
        sigIms = np.zeros(sigDims,dtype='uint16')
        cenListsTF = np.zeros((self.NT,self.NF,self.NM,2),dtype='uint16')
        
        # build new sigIm for each t,f-point, then find aligments
        if isinstance(cens,np.ndarray):
            cenListsTF = cens.copy()
        else:
            ranges = map(range,[self.NT,self.NF])
            for t,f in product(*ranges):
                for c in range(len(aCh)):
                    sigIms[:,c] = self.data[t,f,:,:,aCh[c]].copy().max(axis=1)
                cenListsTF[t,f] = findCentres(sigIms,xMon,olap,method,pars)
            del sigIms

        # initiate data for storage of final assemblies
        dims = (self.NT,self.NF,1,self.NZ,self.NC,ySizeOut,xSizeOut)
        _data = np.zeros(dims,dtype='uint16')

        # now assemble montages
        ranges = map(range,[self.NT,self.NF,self.NZ,self.NC])
        for t,f,z,c in product(*ranges):
            _data2 = self.data[t,f,:,z,c].copy()
            _data[t,f,0,z,c] = noMergeStitch2(_data2,cenListsTF[t,f],
                                              shapeOut,xMon)
            del _data2

        self.data = _data.copy()
        del _data
        self.updateDimensions()
        self.SeshM = [None]
        if verbose:
            print('stitched')
        if returnCens:
            return cenListsTF
        self.newSeshNM = 1
        self.newSeshNY = self.NY
        self.newSeshNX = self.NX

    

    def StitchIt2(self,chans=False,returnAll=False,cens=None,verbose=False,flipYQ=False,printReport=True):
        """
        Stitches together grid-like montage tiles in axis i=2 of the TData.

        Parameters
        ----------
        chans : list
            The channels names of the channels you want it to use in the
            cross-correlation. If False then it tries all.
        returnAll : bool
            Whether to return the centres that you found so they can be used
            on another equivalent TData. Also returns the shifts, in format 
            [x_back_shift_S,y_shift_S,y_back_shift_T,x_shift_T]. Values are 
            np.nan where not appropriate. _S/T means from side/top alignment. 
            back_shift is in direction of overlap, is how much is shifted 
            back from putting edges next to each other. _shift is other 
            direction, measured as shift after aligning image centres.
        cens : list of numpy.ndarray
            centres found previously to use instead of doing the search.
        flipYQ : bool
            E.g. opera requires tiles to be flipped vertically for stitching.
        printReport : bool
            Whether to print some info about the stitching results.

        Notes
        -----
        This function always does maximum z-projection before sending tiles 
        to be aligned.

        Since we are finding the best alignments for each tile, we may end up
        with different sized images, both in the TData and across the whole
        XFold. To avoid this the stitch functions does padding and/or cropping
        to the final image so that you can define beforehand what size output
        you want. To give the same size across a whole XFold, this looks to
        the parent XFold to define final size.
        """

        if self.NM==1 or np.prod(np.array(self.Shape))==0:
            return

        # find if TData has been downsized to help define final size
        actualNX = self.NX
        actualNY = self.NY
        NY = self.ParentSession.NY
        NX = self.ParentSession.NX
        downX = round(NX/self.NX)
        downY = round(NY/self.NY)

        # decide what the output size will be
        # get all the values we need from all sessions in the XFold
        sessionsList = self.ParentXFold.SessionsList
        allOverlaps = [s.MOlap for s in sessionsList]
        allNY = [s.NMY for s in sessionsList]
        allNX = [s.NMX for s in sessionsList]
        allSizeY = [s.NY/downY for s in sessionsList]
        allSizeX = [s.NX/downX for s in sessionsList]
        # zip all these together
        allZipY = zip(allOverlaps,allNY,allSizeY)
        allZipX = zip(allOverlaps,allNX,allSizeX)

        # the size in x or y, call it q, would be given by:
        # (no. of tiles in Q)*(no. of pixels in Q) - overlapping part
        # with overlapping part given by: (N tiles - 1)*(N pixels)*overlap
        # we divide the overlap by 2 to give a number that is an overestimate
        # i.e. we hope we will always have some black padding at the edges:
        ySizeOut = [NMY*NY-(NMY-1)*NY*OL/2 for OL,NMY,NY in allZipY]
        xSizeOut = [NMX*NX-(NMX-1)*NX*OL/2 for OL,NMX,NX in allZipX]

        # we expect these to all be the same
        if len(set(ySizeOut)) != 1 or len(set(xSizeOut)) != 1:
            self.ParentXFold.Warnings.append(EM.si1)
        ySizeOut = int(max(ySizeOut))
        xSizeOut = int(max(xSizeOut))
        shapeOut = (ySizeOut,xSizeOut)

        xMon = self.ParentSession.NMX
        yMon = self.ParentSession.NMY
        olap = self.ParentSession.MOlap

        # reorder M-axis if it's not the normal LRUP ordering
        # I guess you don't worry about tdata.SeshM because NM=1 after all this
        if self.ParentSession.MontageOrder == 'UDRL':
            reshape = genF.UDRL2LRUD(list(range(self.NM)),yMon,xMon)
            self.data = self.data[:,:,reshape]
        elif self.ParentSession.MontageOrder == 'TilePos':
            reshape = genF.sort_grid_points(self.ParentSession.TilePosX,self.ParentSession.TilePosY)
            self.data = self.data[:,:,reshape]
        if flipYQ:
            self.data = self.data[:,:,:,:,:,-1::-1,:].copy()

        # only want to align using these channels
        # i.e. no BF because cross-correlation doesn't work well
        if chans:
            fluoChans = chans
        else:
            fluoChans = self.Chan
        # remove possible blank channels added by matchChannels
        fluoChans = [c for c in fluoChans if c in self.startChan]
        aCh = []
        for c in fluoChans:
            if c in self.Chan:
                aCh.append(self.Chan.index(c))
        if aCh==[]:
            self.ParentXFold.Warnings.append(EM.si3)
            aCh.append(0)

        if verbose:
            print('aCh: ',aCh)
        
        # initiate arrays to send to findCentres
        # sigIms is zproj of 1 time, 1 field and only chans of aCh
        # also centreLists storage array: one list (len=NM) for each t,f-point
        sigDims = (self.NM,len(aCh),self.NY,self.NX)
        sigIms = np.zeros(sigDims,dtype='uint16')
        cenListsTF = np.zeros((self.NT,self.NF,self.NM,2),dtype='uint16')
        all_shifts = []
        all_ccss = []
        
        # build new sigIm for each t,f-point, then find aligments
        if isinstance(cens,np.ndarray):
            cenListsTF = cens.copy()
        else:
            ranges = map(range,[self.NT,self.NF])
            for t,f in product(*ranges):
                for c in range(len(aCh)):
                    sigIms[:,c] = self.data[t,f,:,:,aCh[c]].copy().max(axis=1)
                cenListsTF[t,f],ccss,shifts = findCentres2(sigIms,xMon,olap,returnAll=True)
                all_shifts.append(shifts)
                all_ccss.append(ccss)
            del sigIms

        if printReport:
            print('*** report from StitchIt2() *** \n')
            bad_shift_count = -1 # -1 b/c the first non-aligned tiles is included in all_shifts
            total_alignments = -1
            blank_shift = [np.nan,np.nan,np.nan,np.nan]
            for shifts in all_shifts:
                for shift in shifts:
                    total_alignments += 1
                    if shift==blank_shift:
                        bad_shift_count += 1
            print(bad_shift_count,' out of ',total_alignments,' had bad ccs so default used.')
            all_shifts_a = np.array(all_shifts)
            x_back_shift_mean = np.nanmean(all_shifts_a[:,:,0])
            y_shift_mean = np.nanmean(all_shifts_a[:,:,1])
            y_back_shift_mean = np.nanmean(all_shifts_a[:,:,2])
            x_shift_mean = np.nanmean(all_shifts_a[:,:,3])
            print('the mean horizontal overlap shift in pixels was ',x_back_shift_mean)
            print('the mean horizontal side shift in pixels was ',y_shift_mean)
            print('the mean vertical overlap shift in pixels was ',y_back_shift_mean)
            print('the mean vertical side shift in pixels was ',x_shift_mean)     
            print('you metadata overlap was ',olap*100,'%')
            print('mean horizontal overlap shift as a percentage was ',100*(x_back_shift_mean/actualNX))
            print('mean vertical overlap shift as a percentage was ',100*(y_back_shift_mean/actualNY))
            print('remember, side shifts should be near zero since they can be +ve and -ve')
            print('you should reset your overlap parameter if it is far from the average one found because it is used when a default is needed')
        
        # initiate data for storage of final assemblies
        dims = (self.NT,self.NF,1,self.NZ,self.NC,ySizeOut,xSizeOut)
        _data = np.zeros(dims,dtype='uint16')

        # now assemble montages
        ranges = map(range,[self.NT,self.NF,self.NZ,self.NC])
        for t,f,z,c in product(*ranges):
            _data2 = self.data[t,f,:,z,c].copy()
            _data[t,f,0,z,c] = noMergeStitch2(_data2,cenListsTF[t,f],
                                              shapeOut,xMon)
            del _data2

        self.data = _data.copy()
        del _data

        self.SeshM = [None]
        self.updateDimensions()
        
        if verbose:
            print('stitched')
        if returnAll:
            return cenListsTF,all_ccss,all_shifts

        self.newSeshNM = 1

    
        
    def SimpleStitch(self,extra_percent=False):
        """
        This is a simpler version of StitchIt. It doesn't do alignments, it 
        just uses centres stored in the Session metadata (in: s.TilePosX,Y). 
        Also doesn't attempt to make an output image the size of the biggest 
        one in the XFold. In fact this means it can do one thing better: if 
        your tdata is just a subsection of the total tiling, it will assemble 
        the tiles in a minimally sized final image.
        """
        imSizeX = self.NX
        imSizeY = self.NY
        
        posX = [self.ParentSession.TilePosX[i] for i in self.SeshM]
        posY = [self.ParentSession.TilePosY[i] for i in self.SeshM]
        
        posXpix = [i/self.ParentSession.pixSizeX for i in posX]
        posYpix = [i/self.ParentSession.pixSizeY for i in posY]

        if isinstance(extra_percent,float):
            extra_pix = extra_percent*self.NX
            all_tile_x = sorted(set(posXpix))
            all_tile_y = sorted(set(posYpix))
            posXpix = [p + (extra_pix*all_tile_x.index(p)) for p in posXpix]
            posYpix = [p + (extra_pix*all_tile_y.index(p)) for p in posYpix]
            
            
        
        minXpix,maxXpix = [min(posXpix),max(posXpix)]
        minYpix,maxYpix = [min(posYpix),max(posYpix)]
        
        posXpixShift = [round(i - minXpix + imSizeX/2) for i in posXpix]
        posYpixShift = [round(i - minYpix + imSizeY/2) for i in posYpix]
        
        out_nx = math.ceil(maxXpix - minXpix + (imSizeX))
        out_ny = math.ceil(maxYpix - minYpix + (imSizeY))

        out_array = np.zeros((self.NT,self.NF,1,self.NZ,self.NC,out_ny,out_nx),dtype='uint16')
        
        dims = (self.NT,self.NF,self.NM,self.NZ,self.NC)
        ranges = map(range,dims)
        for t,f,m,z,c in product(*ranges):
            y0 = round(posYpixShift[m]-(imSizeY/2))
            y1 = round(posYpixShift[m]+(imSizeY/2))
            x0 = round(posXpixShift[m]-(imSizeX/2))
            x1 = round(posXpixShift[m]+(imSizeX/2))
            out_array[t,f,0,z,c,y0:y1,x0:x1] = self.data[t,f,m,z,c,-1::-1,:]
        
        self.data = out_array.copy()
        self.updateDimensions()
        self.SeshM = [None]  
        self.allMeta['SeshM'] = [None]
        self.newSeshNM = 1
        
        
    
    def LabelTime(self,
                  roundM=30,
                  verbose=False,
                  style='hh:mm',
                  label_size_divisor=None):
        """
        Adds time labels to the data in new channel.

        Parameter
        ----------

        roundM : int
            The minute interval that it will round to.
        style : str {'hh:mm','mm:ss'}
            Whether to print as hh:mm or mm:ss.
        label_size_divisor : int
            The height of the text will be roughly the height of the image 
            divided by this.
        """

        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.Shape))==0:
            return

        pSesh = self.ParentSession
        pXFold = self.ParentXFold

        assert pXFold.StartTimes,EM.la1

        # you need a new blank channel for the label to go in:
        padDims = ((0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0))
        self.data = np.pad(self.data,padDims)
        # update attributes
        self.Chan.append('Label')
        self.NC += 1
        self.SeshC.append(None)

        # find time since field's experiment started for all t and f of
        # data. f nested within t to make a list of lists of timedeltas
        tStrings = []
        for i,t in enumerate(self.SeshT):
            tStrings.append([])
            for iF in range(self.NF):

                # find the moment when frame was taken:
                frameTakenMom = pSesh.StartMom + pSesh.TStep*t
                # get the start time from StartTimes
                fieldStartMom = pXFold.StartTimes[self.FieldIDs[iF]]

                # this is the time we want to print on the image:
                tSinceFdStart = frameTakenMom - fieldStartMom
                tSinceFdStart = moment2Str(tSinceFdStart,roundM,style=style)
                tStrings[i].append(tSinceFdStart)

        # add the time string to the label channel:
        dims = [self.NT,self.NF,self.NM,self.NZ]
        ranges = map(range,dims)
        for t,f,m,z in product(*ranges):
            addTimeLabel(self.data[t,f,m,z,-1],tStrings[t][f],label_size_divisor)

        self.updateDimensions()

        self.newSeshNC = self.NC
        if verbose:
            print('labeled')     


    def ScaleBar(self,length='auto',thickness='auto',position='bottom_right'):
        """
        This draws a scale bar on your image. It is added as a channel called 
        'Label_ScaleBar_length'.

        Parameters
        ------------
        length : int or float or 'auto'
            The length in um of your scalebar. 'auto' will give a rounded length 
            close to 1/5th of the width of the image and you'll have to look 
            at the channel name to find out how long it is.
        thickness : int or float or 'auto'
            The thickness in um of your scalebar. 'auto' will give a thickness 
            of 1/8th of the length.
        position : (int,int) or {'top-left','top-right','bottom-left','bottom-right'}
            (int,int) is interpreted as the pixel indices of the top left 
            corner of the scale bar. Or the strings options give a rough 
            position.
        """
    

    def DrawSquare(self,thickness,width=None,pos=None,boundingBox=None):
        """
        Adds time labels to the data in new channel.

        Parameters
        ----------
        thickness : int
            The thickness of the lines in pixels.
        boundingBox : [int,int,int,int]
            The bounding box corner pixel indices (numpy format) as y0,x0,y1,x1.
        width : int
            The width of the square in um.
        pos : [y,x] int
            The position of the centre of the square in pixels from top left (i.e. numpy indices).            
        """

        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.Shape))==0:
            return

        pSesh = self.ParentSession
        pXFold = self.ParentXFold

        # you need a new blank channel for the label to go in:
        padDims = ((0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0))
        self.data = np.pad(self.data,padDims)
        # update attributes
        self.Chan.append('Label')
        self.NC += 1
        self.SeshC.append(None)
        if width:
            width = self.um_2_pixels(width)    
            assert pos,'if you provide width yiou have to provide pos'
    
            halfWidth = int(width/2)
            y0,x0,y1,x1 = (pos[0]-halfWidth,pos[1]-halfWidth,pos[0]+halfWidth,pos[1]+halfWidth)
        elif boundingBox:
            y0,x0,y1,x1 = boundingBox
        
        if y0<0:
            y0 = 0
        if x0<0:
            x0 = 0
        if y1+thickness>self.data.shape[-2] - 1:
            y1 = self.data.shape[-2] - thickness
        if x1+thickness>self.data.shape[-1] - 1:
            x0 = self.data.shape[-1] - thickness

        # find the maximum value of the datatype
        # for integer type
        if np.issubdtype(self.data.dtype, np.integer):
            max_val = np.iinfo(self.data.dtype).max
        # for float type
        elif np.issubdtype(self.data.dtype, np.floating):
            max_val = np.finfo(self.data.dtype).max
            
        # add the time string to the label channel:
        dims = [self.NT,self.NF,self.NM,self.NZ]
        ranges = map(range,dims)
        for t,f,m,z in product(*ranges):
            self.data[t,f,m,z,-1,y0:y0+thickness,x0:x1] = max_val
            self.data[t,f,m,z,-1,y0:y1,x0:x0+thickness] = max_val
            self.data[t,f,m,z,-1,y1:y1+thickness,x0:x1+thickness] = max_val
            self.data[t,f,m,z,-1,y0:y1,x1:x1+thickness] = max_val

        self.updateDimensions()

        self.newSeshNC = self.NC

    
    def AlignExtract(self,templateDic,deltaAng=0.25,clip=False,maxAng=15,
                     manualScale=False,storedAlignments=False,verbose=False):
        """
        This does cross-correlation template matching to extract the regions
        of your data corresponding to templates that you give it. The
        templates should be rectanglular maximum projected BF images of the
        original data, i.e. it will pull the max-projected BF image from the
        TData to match against. The template may be rotated with respect to
        the original data.

        Parameters
        ----------
        templateDic : str or dict
            If string then it must be a path to a folder containing the
            template images saved as tifs. String with only one level of path
            is interpreted as a directory within the parent of XPath. The
            images must be in subdirectories for each field with name
            'Exp'+fieldID. If dict then it must be a dictionary with
            {fieldID : path to tif}.
        deltaAng : float
            The size of the steps in angle that it tests, in degrees.
        clip : bool or int
            Whather to clip the template intensity values to avoid problems
            with bright spots. E.g. clip=90 will clip values above 90% of the
            max.
        maxAng : float
            The maximum rotation that it will apply in the search, in degrees.
            It will apply this angle in +ve and -ve directions.
        manualScale : int or list of ints
            If TData has been downscaled you must provide this to downscale
            the template before cross-correlation. If int the it will
            downscale by (int,int).
        storedAlignments : bool or str
            Whether to use use alignments stored in filepath storedAlignments
            instead of doing the whole calculation. If not False then it will
            also save alignments it does calculate to this file.

        Notes
        -----
        For best results you must of course have processed the templates
        exactly as you have processed the TData.

        If the template size varies between fields fshapemxthen a global (i.e. the
        same for entire XFold) output size will be determined and all
        extract will be padded to this size. This avoids problems of jagged
        arrays for TDatas containing multiple fields.
        """
        assert isinstance(clip,bool) or isinstance(clip,int),EM.ae0
        
        templateDic_Old = copy.copy(templateDic)

        pXFold = self.ParentXFold
        if storedAlignments:
            sAP = storedAlignments
            if os.path.split(sAP)[0]=='':
                sAP = os.path.join(pXFold.XPathP,sAP)
        if manualScale:
            if isinstance(manualScale,int):
                manualScale = tuple([manualScale,manualScale])
            if isinstance(manualScale,list):
                manualScale = tuple(manualScale)

        if np.prod(np.array(self.Shape))==0:
            return
        assert self.NM==1,EM.ae6

        # package these to not bloat code
        ps = (deltaAng,maxAng)

        # load storedAlignments and build alignDic
        if storedAlignments:
            self.ParentXFold.buildStoredAlignments(storedAlignments)
            alignDic = self.ParentXFold.AlignDic[storedAlignments]

        # make templateDic
        if isinstance(templateDic,str):
            templatesName = templateDic
            self.ParentXFold.buildTemplateDic(templateDic)
            templateDic = self.ParentXFold.TemplateDic[templateDic].copy()
        else:
            templatesName = 'templates'

        for FID in self.FieldIDs:
            assert FID in templateDic.keys(),EM.ae3

        # first find the max y and max x size of all templates
        # and downscale template if needed
        if manualScale:
            for k,tem in templateDic.items():
                templateDic[k] = downscale_local_mean(tem,manualScale)
        pXFold.buildExtractedSizeDic(templatesName,templateDic)
        maxYSize,maxXSize = pXFold.ExtractedSizeDic[templatesName]
        shapeMx = (maxYSize,maxXSize)

        dims = (self.NT,self.NF,1,self.NZ,self.NC,maxYSize,maxXSize)
        _data = np.zeros(dims,dtype='uint16')

        # cycle over field and then times because each field has a
        # different template to be loaded and each time needs aligning
        for i,f in enumerate(self.SeshF):
            template = templateDic[self.FieldIDs[i]]
            shape = template.shape
            assert shape[0]<self.data.shape[5],EM.ae8
            assert shape[1]<self.data.shape[6],EM.ae8

            if 'BF' in self.Chan:
                BFIndex = self.Chan.index('BF')
            else:
                BFIndex = None

            for t in range(self.NT):
                code = 'S'+str(self.SessionN)
                code += 'T'+str(self.SeshT[t])+'F'+str(f)
                if storedAlignments and code in alignDic.keys():
                    qs = alignDic[code]
                    if verbose:
                        print('found stored alignment: ',code)
                else:
                    # make max projection of chan BF to do the search in
                    assert isinstance(BFIndex,int), EM.ae4
                    BFZProj = self.data[t,i,0,:,BFIndex].copy().max(axis=0)
                    BFZProj = BFZProj.astype('float32')
                    if clip:
                        if isinstance(clip,int):
                            maxP = clip
                        else:
                            maxP = 90
                        maxV = np.percentile(np.ravel(BFZProj),maxP)
                        BFZProj[BFZProj>maxV] = maxV
                        template[template>maxV] = maxV
                    ang,shift = findRegion(BFZProj,template,*ps)
                    qs = (ang,shift)
                    if storedAlignments:
                        line = '\n'+code+' : '+str(ang)+' '+str(shift[0])
                        line += ' '+str(shift[1])
                        with open(sAP,'a') as file:
                            file.write(line)
                _data[t,i] = extractRegion(self.data[t,i],*qs,*shape,*shapeMx)

            # record in the TData that extraction from this template was padded
            pad = [a-b for a,b in zip(shapeMx,shape)]
            self.TemplatePadDic[self.FieldIDs[i]] = pad

        self.data = _data.copy()
        del _data
        self.updateDimensions()
        self.Aligned = templateDic_Old
        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')



    def stabiliseVideo_BF(self,N_sec=4):
        """
        Parameters
        ----------
        N_sec : int
            How many sections to divide the data into (along each axis). 
            Since we use cv.matchTemplate, you need sections to be small to 
            search whole image, i.e. to check for big shifts. So keep the 
            large if there are big shifts or make small for extra speed when 
            shifts are all small.
        """

        assert 'BF' in self.Chan, 'no channel called BF found'
        BFIndex = self.Chan.index('BF')
        
        assert self.NF==1,'only works for 1 field at a time so far'
        assert self.NM==1,'only works with stitched montages so far'
        assert self.NZ==1,'only works for 1 Z-slice so far, z-project your tdata first'
      
        y_shifts_T = [0]
        x_shifts_T = [0]
        
        minmaxs = genF.find_grid_indices(self.NY,self.NX,N=N_sec)
        
        most_recent_good_t = 0
        
        for t in range(1,self.NT):
        
            im1 = self.data[most_recent_good_t,0,0,0,BFIndex].copy()
            im1 = genF.clip_and_rescale_image(im1,1,99)
            im2 = self.data[t,0,0,0,BFIndex].copy()
            im2 = genF.clip_and_rescale_image(im2,1,99)
            
            im_segs_y_shifts = []
            im_segs_x_shifts = []
            ccms = []
            max_values = []
            for i in range(N_sec*N_sec):
                ((y_min,y_max),(x_min,x_max)) = minmaxs[i]
                im2_si = im2[y_min:y_max,x_min:x_max].copy()
            
                ccm = cv.matchTemplate(im1.astype('float32'),
                                       im2_si.astype('float32'),
                                       eval('cv.TM_CCOEFF_NORMED'))
                ccms.append(ccm)
                ccm_b = filters.gaussian(ccm,sigma=3,preserve_range=True)
                    
                max_y_ind,max_x_ind = np.unravel_index(np.argmax(ccm_b),ccm_b.shape)
                
                im_segs_y_shift = max_y_ind - y_min
                im_segs_x_shift = max_x_ind - x_min

                ccm_b_thr = ccm_b>(0.7*np.max(ccm_b))
                if len(np.unique(measure.label(ccm_b_thr)))==2:      
                    im_segs_y_shifts.append(im_segs_y_shift)
                    im_segs_x_shifts.append(im_segs_x_shift)
                    max_values.append(np.max(ccm_b))
        
            if im_segs_y_shifts==[]:
                y_shifts_T.append(0)
                x_shifts_T.append(0)        
            elif len(im_segs_y_shifts)==1:
                most_recent_good_t = t 
                y_shifts_T.append(im_segs_y_shifts[0])
                x_shifts_T.append(im_segs_x_shifts[0])                
            else:
                # we try to find a cluster of close shifts
                most_recent_good_t = t 
            
                coordinates = np.array([(y,x) for y,x in zip(im_segs_y_shifts,
                                                            im_segs_x_shifts)])
                db = DBSCAN(eps=10, min_samples=2).fit(coordinates)
                
                labels = db.labels_
                if all([l==-1 for l in labels]): 
                    # if we can't group them we just take highest one
                    max_value_ind = np.argmax(max_values)
                    y_shifts_T.append(im_segs_y_shifts[max_value_ind])
                    x_shifts_T.append(im_segs_x_shifts[max_value_ind])                       
                else:
                    labels_mode = genF.get_modes(labels,exclude=-1)
                    
                    if len(labels_mode)>1:
                        max_value_ind = np.argmax(max_values)
                        y_shifts_T.append(im_segs_y_shifts[max_value_ind])
                        x_shifts_T.append(im_segs_x_shifts[max_value_ind]) 
                    else:
                        s1 = coordinates[labels == labels_mode[0]]
                        y_shifts_T.append(int(np.mean(s1[:,0])))
                        x_shifts_T.append(int(np.mean(s1[:,1])))
        
        y_shifts_T_cum = []
        x_shifts_T_cum = []
        for i in range(len(y_shifts_T)):
            y_shifts_T_cum.append(sum(y_shifts_T[:i+1]))
            x_shifts_T_cum.append(sum(x_shifts_T[:i+1]))
        
        y_shifts_T = y_shifts_T_cum
        x_shifts_T = x_shifts_T_cum
        
        min_y_shift = np.min(y_shifts_T)
        max_y_shift = np.max(y_shifts_T)
        min_x_shift = np.min(x_shifts_T)
        max_x_shift = np.max(x_shifts_T)
        
        out_im = np.zeros((self.NT,1,1,1,self.NC,
                           self.NY+abs(min_y_shift)+max_y_shift,
                           self.NX+abs(min_x_shift)+max_x_shift))

        y0 = abs(min_y_shift)
        y1 = abs(min_y_shift)+self.NY
        x0 = abs(min_x_shift)
        x1 = abs(min_x_shift)+self.NX
        out_im[0,0,0,0,:,y0:y1,x0:x1] = self.data[0,0,0,0,:].copy()
        
        for t in range(1,self.NT):
            y00 = abs(min_y_shift)+y_shifts_T[t]
            y11 = abs(min_y_shift)+y_shifts_T[t]+self.NY
            x00 = abs(min_x_shift)+x_shifts_T[t]
            x11 = abs(min_x_shift)+x_shifts_T[t]+self.NX
            out_im[t,0,0,0,:, y00:y11, x00:x11] = self.data[t,0,0,0,:].copy()

        self.data = out_im.astype('uint16').copy()
        del out_im
        print('Warning: we havent updated self.newNQ in this function yet so saving and reloading might have troubles.')


    
    def CorrectMisalignedChannel(self,badChan,goodChan,maxShift):
        """
        This is another align and extract function. Made for example where 
        microscope was misaligned so one channel had to be shifted slightly. 
        The badChan is the misaligned one. goodChan is the channel among any 
        of others that will overlap badChan best using a cross-correlation. 
        The resulting data will have had 2*maxShift pixels removed from x and 
        y dimentions.

        Parameters
        ----------
        badChan : int or str
            The channel than needs to be realigned. You can provide as channel 
            index or name.
        """
        badChan = self.chan_index(badChan)
        goodChan = self.chan_index(goodChan)        
            
        ranges = [range(self.NT),range(self.NF),range(self.NM),range(self.NZ)]
        
        _data = np.zeros((self.NT,self.NF,self.NM,self.NZ,
                            self.NC,
                            self.NY-2*maxShift,
                            self.NX-2*maxShift))
        
        for t,f,m,z in product(*ranges):
            
            im1 = self.data[t,f,m,z,badChan].copy().astype('uint8')
            im2 = self.data[t,f,m,z,goodChan,maxShift:-maxShift,
            maxShift:-maxShift].copy().astype('uint8')
            match = cv.matchTemplate(im1,im2,cv.TM_CCOEFF_NORMED)
            minV,maxV,minI,maxI = cv.minMaxLoc(match)
            yShift,xShift = maxShift-np.array(maxI)
        
            _data[t,f,m,z] = self.data[t,f,m,z,:,maxShift:-maxShift,
                                                    maxShift:-maxShift].copy()
            if yShift==maxShift and xShift==maxShift:
                _data[t,f,m,z,badChan] = self.data[t,f,m,z,
                                        badChan,
                                        maxShift+yShift:,
                                        maxShift+xShift:].copy()
            elif yShift==maxShift:
                _data[t,f,m,z,badChan] = self.data[t,f,m,z,
                                        badChan,
                                        maxShift+yShift:,
                                        maxShift+xShift:-maxShift+xShift].copy() 
            elif xShift==maxShift:
                _data[t,f,m,z,badChan] = self.data[t,f,m,z,
                                        badChan,
                                        maxShift+yShift:-maxShift+yShift,
                                        maxShift+xShift:].copy()
            else:
                _data[t,f,m,z,badChan] = self.data[t,f,m,z,
                                        badChan,
                                        maxShift+yShift:-maxShift+yShift,
                                        maxShift+xShift:-maxShift+xShift].copy()

        self.data = _data.copy()
        del _data
        self.updateDimensions()    
        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')
            
                
    
    def ManualAlignExtract(self,al,dep,endLen=None,dil=[0,0],
                           storedAlignments=False):
        """
        This extracts a rectangular region from the TData according to two
        points (which mark the top edge) and a provided region depth. The
        rectangle does not have to be aligned with the image axes.

        Parameters
        ----------

        al : [[int_y0,int_x0],[int_yf,int_xf]]
            The points must mark the top two corners of the rectangle that
            you want to extract.
        dep : int
            You are only marking the top line of the rectangle that you want
            to extract so this gives the length of the region perpendicular
            to the line.
        endLen : int (optional)
            The line can be shortened to this many pixels in length. This is
            used to ensure all images are the same size when extracting from
            many related images.
        dil : [y_int,x_int]
            The number of pixels you want to dilate the extraction by. I.e. to
            extract more that the rectangle you have marked. Gives number in
            y-direction and x, respectively. This is additional to endLen.

        Notes
        ------
        Remember you must have marked the points on a tif which has been
        processed exactly as you are processing during the extraction.

        Also, unlike most methods in this package, this method only works on
        TDatas with one field, so processing must be done in the script
        outside our package.
        """
        y0,x0,yf,xf = [x for p in al for x in p]
        if not endLen:
            endLen = math.sqrt((xf-x0)**2 + (yf-f0)**2)
        ang = math.atan((yf-y0)/(xf-x0))
        ang = (ang/math.pi)*180

        TL = rotCoord_NoCrop(al[0],self.NY,self.NX,ang)
        NIms = self.NT*self.NF*self.NM*self.NZ*self.NC
        newD = (NIms,self.NY,self.NX)
        self.data = rotate_image(np.moveaxis(self.data.reshape(newD),0,-1),ang)
        h,w = self.data.shape[0:2]
        self.data = np.moveaxis(self.data,-1,0).reshape(self.Shape[0:5]+(h,w))
        xslice = slice(TL[1]-dil[1],TL[1]+endLen+dil[1],1)
        yslice = slice(TL[0]-dil[0],TL[0]+dep+dil[0],1)
        self.data = self.data[:,:,:,:,:,yslice,xslice]

        self.updateDimensions()
        print('Warning: we havent updated self.newNQ in this function yet so saving and reloading might have troubles.')


    def SaveData(self,
                 outDir,
                 overwrite=False,
                 verbose=False,
                 newSessionData=False,
                 customTag='',
                 newSeshNF=None,
                 compress=False):
        """
        This takes the image data from your TData and saves it such that
        image j will be able to open it and have all the channel and
        hyperstack info it needs.

        It always separates fields into different files and folders, with the
        folders named ExpFID where FID is the field ID.

         !!! IMPORTANT NOTE: only T and C (and effectively F by parent 
         directory) tags are added to the saved files so far. If we ever want 
         to save other Q in separate files and be able to reload them then 
         we would need those tags added. !!!        

        Parameters
        ----------
        outDir : str
            Where to put the data you save. A path with just one level is 
            interpreted as the folder name within the XPath parent directory.
            You can give it a dictionary of outDirectories which gives
            the path for each field.
        newSessionData : dict
            If not provided then the original session data is just copied 
            directly to the saved file. This makes a problem though if you 
            want to make a new XFold from the processed data in order to do 
            further analysis... because e.g. it will think there are multiple 
            Z-slices whereas you might have z-projected everything. So here 
            you can provide this information in format {'parameter':newValue} 
            and it will be added to the sesh_meta_dict.
        customTag : str
            A string that is added to the filename just before the other tags.
        newSeshNF : None or int
            Use if you want your saved data to be reopenable and you are not 
            saving all Fields from your original Session. I.e. without this 
            multisesh will assume the saved Session has the same number of 
            Fields as the original Session... but often you do not save all 
            Fields, so NF should be put here.
        compress : bool
            Whether to save a compressed image file using np.savez_compressed. 
            Only use for boolean masks.        

        Notes
        -----
        It strips old tags from the file name and adds new tags to
        all names. We have our own tag system, _s000N_t000M for the session
        the data comes from and the time point relative to that session.
        """
        
        if np.prod(np.array(self.Shape))==0:
            return

        if isinstance(outDir,str):
            if os.path.split(outDir)[0]:
                parnt = os.path.split(outDir)[0]
                analPath = os.path.join(parnt,os.path.split(outDir)[1])
            else:
                analPath = os.path.join(self.ParentXFold.XPathP,outDir)
        else:
            raise Exception('You must provide an outDir.')

        if not os.path.exists(analPath):
            os.makedirs(analPath)

        for i,f in enumerate(self.SeshF):

            fieldDir = defs.FieldDir+self.FieldIDs[i]
            outDirPath = os.path.join(analPath,fieldDir)

            if not os.path.exists(outDirPath):
                os.mkdir(outDirPath)

            sessionN = self.SessionN
            sessionTag = '_s' + str(sessionN).zfill(4)
            timeTag = '_t' + str(self.SeshT[0]).zfill(4)
            chanTag = ''
            # only add channel tag if your tdata is not all of the session chan
            if not all([c in self.Chan for c in self.ParentSession.Chan2]):
                chanTag = '_C'
                for c in self.Chan:
                    chanTag += '_Ch' + c
                    
            if self.SeshM[0]:
                mTag = '_m' + str(self.SeshM[0]).zfill(4)
            else:
                mTag = '_m0000'
                
            tags = customTag + sessionTag + timeTag + chanTag +mTag
            if compress:
                outName = self.ParentSession.StrippedFileName + tags + '.npz'
            else:
                outName = self.ParentSession.StrippedFileName + tags + '.tif'
            outPath = os.path.join(outDirPath,outName)

            sesh_meta_dict = self.ParentSession.allMeta.copy()
            if newSeshNF:
                sesh_meta_dict.update({'NFOriginal':sesh_meta_dict['NF']})
                sesh_meta_dict['NF'] = newSeshNF  
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[1] = newSeshNF
                sesh_meta_dict['Shape'] = tuple(new_shape)  
            if self.newSeshNM:
                sesh_meta_dict.update({'NMOriginal':sesh_meta_dict['NM']})
                sesh_meta_dict['NM'] = self.newSeshNM   
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[2] = self.newSeshNM
                sesh_meta_dict['Shape'] = tuple(new_shape)                
            if self.newSeshNZ:
                sesh_meta_dict.update({'NZOriginal':sesh_meta_dict['NZ']})
                sesh_meta_dict['NZ'] = self.newSeshNZ   
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[3] = self.newSeshNZ
                sesh_meta_dict['Shape'] = tuple(new_shape)
            if self.newSeshNC:
                sesh_meta_dict.update({'NCOriginal':sesh_meta_dict['NC']})
                sesh_meta_dict['NC'] = self.newSeshNC  
                sesh_meta_dict.update({'ChanOriginal':sesh_meta_dict['Chan']})
                sesh_meta_dict['Chan'] = self.Chan
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[4] = self.newSeshNC
                sesh_meta_dict['Shape'] = tuple(new_shape)
            if self.newSeshNY:
                sesh_meta_dict.update({'NYOriginal':sesh_meta_dict['NY']})
                sesh_meta_dict['NY'] = self.newSeshNY   
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[5] = self.newSeshNY
                sesh_meta_dict['Shape'] = tuple(new_shape)  
            if self.newSeshNX:
                sesh_meta_dict.update({'NXOriginal':sesh_meta_dict['NX']})
                sesh_meta_dict['NX'] = self.newSeshNX
                if not 'ShapeOriginal' in sesh_meta_dict.keys():
                    sesh_meta_dict.update({'ShapeOriginal':sesh_meta_dict['Shape']})
                new_shape = list(sesh_meta_dict['Shape'])
                new_shape[6] = self.newSeshNX
                sesh_meta_dict['Shape'] = tuple(new_shape)                      
            if newSessionData:
                for k,v in newSessionData.items():
                    if not k+'Original' in sesh_meta_dict.keys():
                        sesh_meta_dict.update({k+'Original':sesh_meta_dict[k]})
                    sesh_meta_dict[k] = v
            
            sesh_meta_dict['FieldIDMap'] = self.ParentSession.FieldIDMap
            sesh_meta_dict['LRUPOrdering'] = self.ParentSession.LRUPOrdering
            sesh_meta_dict['StartTimes'] = self.ParentXFold.StartTimes
        
            seshQ = [self.SeshT,[f],self.SeshM,self.SeshZ,self.SeshC]
            ss = genF.saveTiffForIJ(
                               outPath,
                               self.data[:,[i]],
                               self.Chan,
                               seshQ,
                               overwrite=overwrite,
                               zSize=self.ZStep,
                               pixSizeX=self.pixSizeX,
                               pixSizeY=self.pixSizeY,
                               tstep=self.ParentSession.TStep.seconds,
                               sesh_meta_dict=sesh_meta_dict,
                               tdata_meta_dict=self.allMeta,
                               compress=compress)

            self.ParentXFold.SavedFilePaths.append(outPath)

        return 


    def SwapXYZ(self,axisA,axisB,rescaleToAspectRatio1=False):
        """ 
        This swaps the XYZ axes around.

        Parameters
        ---------
        axisA/B : int
            The index of the axes you want to swap.
        rescaleToAspectRatio1 : bool
            If True then it stretches (or compresses) self.data so that the 
            pixel sizes are equal in the new YX dimensions.
        """
        # don't allow swapping of anything except XYZ for now:
        permittedAxes = [3,5,6]
        if axisA not in permittedAxes or axisB not in permittedAxes:
            errMess = 'You can only use SwapXYZ() to swap axes'\
                        '{permittedAxes}.'
            raise Exception(errMess.format(permittedAxes=permittedAxes))

        # do the swap
        self.data = np.swapaxes(self.data,axisA,axisB)

        if axisA == 3 or axisB ==3:
            self.SeshZ = [None for z in range(self.NZ)]
        self.updateDimensions()            
        
        self.Aligned = False
        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')


    def SwapZY(self,rescaleToAspectRatio1=False,initialDownsample=False):
        """ 
        Simpler version of SwapXYZ.

        Parameters
        ---------
        rescaleToAspectRatio1 : bool
            If True then it stretches (or compresses) self.data so that the 
            pixel sizes are equal in the new YX dimensions. Note that this can 
            multiply the size of your data by a lot if the original x and z 
            reolutions were very different. Also, the rescale function is very 
            slow. It prints a warning about this...see initialDownsample.
        initialDownsample : False or int
            This makes shrinks the data along the x-axis by the factor 
            provided. Do this to reduce the increase in data size created by 
            rescaleToAspectRatio1... in fact it doesn't do the shrinking if no 
            rescale.
        """
        
        # do the swap
        self.data = np.swapaxes(self.data,3,5)

        _ny = self.NY
        _nz = self.NZ
        self.NY = _nz
        self.NZ = _ny
        self.SeshZ = [None for z in range(self.NZ)]

        _zstep = self.ZStep
        _pixSizeY = self.pixSizeY
        self.ZStep = _pixSizeY
        self.pixSizeY = _zstep
        
        # ZStep is always in um so may need to convert if XY in m
        if self.pixUnit=='m':
            self.ZStep = self.ZStep*1000000
            self.pixSizeY = self.pixSizeY/1000000

        if rescaleToAspectRatio1:
            if initialDownsample:
                downsam = (1,1,1,1,1,1,initialDownsample)
                self.data = downscale_local_mean(self.data,downsam).astype('uint16')
                self.pixSizeX = self.pixSizeX*initialDownsample
            scaleby = self.pixSizeY/self.pixSizeX
            if not initialDownsample:
                print(EM.sw1,scaleby,EM.sw2)
            # we do order=0 rather than 1 because this function is so slow...
            self.data = rescale(self.data,(1,1,1,1,1,scaleby,1),order=0)
            self.pixSizeY = self.pixSizeX
        
        self.updateDimensions()   
        self.newSeshNZ = self.NZ
        self.newSeshNY = self.NY
        self.newSeshNX = self.NX           
        
        self.Aligned = False


    def TakeXYSection(self,Yi,Yf,Xi,Xf):
        """
        This takes a basic rectangular section in XY of your data.
        Give the values you would use for a numpy array.
        """

        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.Shape))==0:
            return

        a = abs(Xi)>self.NX
        b = abs(Xf)>self.NX
        c = abs(Yi)>self.NY
        d = abs(Yf)>self.NY
        if any([a,b,c,d]):
            raise Exception(EM.xy1)
        self.data = self.data[:,:,:,:,:,Yi:Yf,Xi:Xf].copy()
        self.updateDimensions()
        self.Aligned = False
        self.newSeshNY = self.NY
        self.newSeshNX = self.NX        

    
    def PadXY(self,Yi=0,Yf=0,Xi=0,Xf=0):
        """
        Pads the Y and X dimensions with zeros
        """
        self.data = np.pad(self.data,((0,0),(0,0),(0,0),(0,0),(0,0),
                                      (Yi,Yf),(Xi,Xf)))
        self.updateDimensions()
        self.Aligned = False
        self.newSeshNY = self.NY
        self.newSeshNX = self.NX  

    
    def Clip(self,maxV):
        """
        very simple method to just clip pixel intensities according to max.
        Needed because ExtractProfileRectangle needs the mask outline you draw
        to be the maximum pixel value in the image and that can be annoying
        for images with overexposed regions or random high pixels.
        """
        self.data[self.data>maxV] = maxV


    def AsDType(self,dtype,low_percentile,high_percentile):
        """
        This changes the dtype of the data and concurrently rescales the 
        histogram so you get nicely viewable images at the end. Mainly used 
        just before saving to perhaps make a less heavy image that can be 
        directly opened in FIJI.

        Parameters
        ----------
        dtype : str
            Should be a recognised numpy dtype.
        low/high_percentile : int [0,100]
            The lower and upper percentile that you will clip to when 
            rescaling.

        Note
        ----
        It treats channels individually but everything else it caluclates in 
        one go, i.e. with just one lower and upper pixel value.
        """

        new_data = np.zeros(self.Shape,dtype=dtype)
        
        for c in range(self.NC):
        # Calculate the percentile values
            low_value = np.percentile(self.data[:,:,:,:,c], low_percentile)
            high_value = np.percentile(self.data[:,:,:,:,c], high_percentile)
            
            # Clip the image
            clipped_image = np.clip(self.data[:,:,:,:,c].copy(), low_value, high_value).astype('float32')
            
            # Rescale the image
            old_range = high_value - low_value  # Old range of pixel values
            new_range = np.iinfo(dtype).max - np.iinfo(dtype).min  # New range of pixel values
            
            # Rescale the pixel values to the new range
            rescaled_image = (((clipped_image - low_value) * new_range) / old_range) + np.iinfo(dtype).min
            
            # Convert the image to the desired data type
            rescaled_image = rescaled_image.astype(dtype)

            new_data[:,:,:,:,c] = rescaled_image.copy()
            del rescaled_image

        self.data = new_data.copy()
               

        
    def ExtractRings(self,
                    outDir=False,
                    masks=None,
                    NSegs=10,
                    ringWidth=5,
                    overwrite=False,
                    returnData=False,
                    normIm=False
                   ):
        """
        This saves csvs containing averaged pixel values taken from the rings 
        made by dilating/eroding the mask. I.e. if you want pixel intensites 
        but saving a csv of all pixels is too large then do averaging with 
        this.

        Parameters
        ----------
        outDir : str
            Name of directory to save csvs too. It is in the parent directory
            of XPath. This directory will be created if needed. If False then 
            it won't save anything.
        masks : str or None or array-like
            This is an image which defines a region of arbitrary shape to
            grow/shrink to define the rings that pixels are extracted from.
            This image should have the usable region outlined with the highest
            pixel intensity of the image.
        NSegs : int
            The number of segments to return along the x-axis. All pixel
            values are averaged within each segment. The X position in pixels
            of the middle of each segment will be given in the CSV row heading
            (measured relative to the window).
        ringWidth : int
            The size of the dilation/erosion and therefore width of the ring 
            you make each time (and average over), in pixels.
        overwrite : bool
            Whether to allow csvs to be overwritten.
        returnData : bool
            If True then it will return the extracted values. See Returns below.
        normIm : bool or array-like
            If not False then an image to divide your image through by before 
            extracting the data.

        Returns
        -------
        If returnData=False then nothing returned but csvs are saved. The csvs
        have NSegs+1 columns and no. of rows depends on the image and mask. 
        One separate csv for each T,F,M,Z,C.

        If returnData=True then it returns : (outList,ypos,xpos).
        outList : list of numpy arrays
            Shape is (NT,NM,NZ,NC,ny,nx) where ny,nx are the number of
            measures taken along y and x. Each element corresponds to a field
            in self.
        ypos,xpos : lists of lists.
            Each inner list corresponds to a field in self. Then these are the
            positions of the measures in dataList within the gradient window.
        """

        # don't want to include label channel in future loops
        CC = [i for i,c in enumerate(self.Chan) if c!='Label']


        # this collects out data if you are returning it.
        # one element for each field since they may have differnt dimensions
        if returnData:
            outList = []
            xpos = []
            ypos = []
            
        if not masks in self.ParentXFold.RingDic.keys():
            self.ParentXFold.RingDic[masks] = {}
        RingDic = self.ParentXFold.RingDic[masks]

        for i,FID in enumerate(self.FieldIDs):

            if isinstance(NSegs,dict):
                NSegsF = NSegs[FID]
            else:
                NSegsF = NSegs
            # will need to know padding applied to self.data wrt to templates
            if FID in self.TemplatePadDic.keys():
                pd = self.TemplatePadDic[FID]
            else:
                pd = [0,0]
            # put pad in format for easy removal of pads from image
            pyf = self.NY-math.ceil(pd[0]/2)
            pxf = self.NX-math.ceil(pd[1]/2)
            yi,yf,xi,xf = [pd[0]//2,pyf,pd[1]//2,pxf]

            # similarly import the mask
            if masks is None:
                mask = np.ones((self.NY,self.NX)).astype('int')
            elif isinstance(masks,str):
                self.ParentXFold.buildMasks(masks)
                mask = self.ParentXFold.MaskDic[masks][FID]
            elif isinstance(masks,np.ndarray):
                mask = masks
            else:
                raise Exception('masks provided in unknown format')
            
            # now load or make the ring masks 
            k1 = (FID,ringWidth,NSegsF)
            if k1 in RingDic.keys():
                rings = [r[0] for r in RingDic[k1]]
                rlab = np.array(['R distance']+[r[1] for r in RingDic[k1]])
                ringSegs = [r[3] for r in RingDic[k1]]
                alab = np.array(RingDic[k1][0][2])
                dr = ringWidth 
                NR = len(rings)
                da = 2*np.pi/NSegsF # i.e. angle of segment
            else: 
                # starting from mask, dilate to make all bigger rings
                rings = []
                rlab = []
                d1 = disk(ringWidth)
                mask1 = mask.copy()
                dilMask = dilation(mask, selem=d1)
                ring = np.logical_xor(dilMask,mask1)
                rings.append(ring)
                rlab.append(ringWidth)
                while not np.all(dilMask):
                    mask1 = dilMask.copy()
                    dilMask = dilation(mask1, selem=d1)                   
                    ring = np.logical_xor(dilMask,mask1)
                    rings.append(ring)
                    rlab.append(ringWidth+rlab[-1])
                 
                # starting from drawn outline, erode to make all smaller rings
                rings2 = []
                rlab2 = []
                mask1 = mask.copy()
                erMask = erosion(mask1, selem=d1)
                ring = np.logical_xor(mask1,erMask)
                rings2.append(ring)
                rlab2.append(-ringWidth)
                while np.any(erMask):
                    mask1 = erMask.copy()
                    erMask = erosion(mask1, selem=d1)
                    ring = np.logical_xor(mask1,erMask)
                    rings2.append(ring)
                    rlab2.append(-ringWidth+rlab2[-1])
                
                # combine ring lists
                rings2.reverse()
                rlab2.reverse()
                rings = rings2 + rings
                rlab = rlab2 + rlab
    
                # setup sizes of sections
                dr = ringWidth 
                NR = len(rings) # i.e. number of rings
                da = 2*np.pi/NSegsF # i.e. angle of segment            
                
                # make labels for csvs
                rlab = ['r='+str(r)+'pixels' for r in rlab]
                rlab = np.array(['R distance']+rlab)            
                alab = np.linspace(-np.pi + da/2,np.pi - da/2,NSegsF)
                alab = [str(round(a,2)) for a in alab]
                alab = np.array(['ang='+a+' radians' for a in alab])
                
                # find centre of mask
                label_img = measure.label(rings[0])
                regions = measure.regionprops(label_img)
                ceny,cenx = regions[0].centroid    
                
                ringSegs = []
                for r,ring in enumerate(rings):
                    # setup do divide rings into segments by angle
                    y,x = ring.nonzero()
                    angles = np.arctan2(y-ceny,x-cenx)
                    minAngs = np.linspace(-np.pi,np.pi - (2*np.pi/NSegsF),NSegsF)
                    maxAngs = np.linspace(-np.pi + (2*np.pi/NSegsF),np.pi,NSegsF)
                    
                    ringSegs2 = []
                    for a,[minA,maxA] in enumerate(zip(minAngs,maxAngs)):
                        # make mask of just segment
                        ysel = y[np.logical_and(angles>minA,angles<=maxA)]
                        xsel = x[np.logical_and(angles>minA,angles<=maxA)]
                        ringSeg = np.zeros_like(ring)
                        ringSeg[ysel,xsel] = True
                        ringSegs2.append(ringSeg)
                    ringSegs.append(ringSegs2)
                # save all this to the parent XFold
                RingDic[k1] = list(zip(rings,rlab[1:],[alab]*NR,ringSegs))
                
            if returnData:
                outData = np.zeros((self.NT,self.NM,self.NZ,len(CC),NR,NSegsF))

            # process separately for T,M,Z,chan
            ranges = [range(self.NT),range(self.NM),range(self.NZ),CC]
            for t,m,z,c in product(*ranges):
                # remove padding and take one TFMZC
                data = self.data[t,i,m,z,c,yi:yf,xi:xf].copy()
                if isinstance(normIm,np.ndarray):
                    normIm2 = normIm[yi:yf,xi:xf].copy()
                    data = data.astype('float32')/normIm2
                # initiate np array and steps for collecting data
                _csv = np.zeros((NR,NSegsF))
                for r,[ring,rSs] in enumerate(zip(rings,ringSegs)):
                    for a,rS in enumerate(rSs):
                        if data[rS].size==0:
                            _csv[r,a] = 'NaN' 
                        else:
                            _csv[r,a] = np.mean(data[rS])
                    
                # add ang-slice and r-distance headings
                _csv = np.vstack((alab,_csv))
                _csv = np.hstack((rlab.reshape((NR+1,1)),_csv))
                if returnData:
                    outData[t,m,z,c] = _csv[1:,1:]
                
                # save csv
                if outDir:
                    outName ='T'+str(self.Times[FID][t])+'min_M'+str(m)+'_Z'
                    outName += str(z)+'_C'+self.Chan[c]+'.csv'
                    outF = defs.FieldDir + FID
                    outPath1 = os.path.join(self.ParentXFold.XPathP,outDir,outF)
                    if not os.path.exists(outPath1):
                        os.makedirs(outPath1)
                    outPath = os.path.join(outPath1,outName)
                    if not overwrite:
                        assert not os.path.exists(outPath),EM.ae5
                    with open(outPath,'w',newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(_csv)

                if returnData:
                    outList.append(outData)
                    ap = [da*a+da//2 for a in range(NSegsF)]
                    apos.append(ap)
                    rp = [dr*r+dr//2 for r in range(NR)]
                    rpos.append(rp)

        if returnData:
            return (outList,rpos,apos)
        

    def TakeWindow(self,windows,dilate=None):
        """
        Crops a TData that has been aligned to a template with alignExtract(),
        to leave just the window.

        windows : str
            A path to a folder containing the window files saved as .txt.
            String with only one level of path is interpreted as a directory
            within the parent of XPath. The files must be in subdirectories
            for each field with name 'Exp'+fieldID. It doesn't reload the
            windows if they are already loaded into ParenXFold.WindowDic.
        dilate : int
            Number of pixels to expand the window size by at each edge.

        Notes
        -----
        Only works for TDatas with one Field otherwise you would have jagged
        arrays!
        """
        assert self.NF==1,'Çan only TakeWindow when TData.NF==1.'
        if np.prod(np.array(self.Shape))==0:
            return

        self.ParentXFold.buildWindows(windows)

        FID = self.FieldIDs[0]
        win = self.ParentXFold.WindowDic[windows][FID]
        if FID in self.TemplatePadDic.keys():
            pd = self.TemplatePadDic[FID]
        else:
            pd = [0,0]
        win = [[a+b//2 for a,b in zip(v,pd)] for v in win]
        yi,yf,xi,xf = [win[0][0],win[2][0],win[0][1],win[2][1]]
        if dilate:
            yi -= dilate
            yf += dilate
            xi -= dilate
            xf += dilate
        assert yi>=0 and xi>=0,'dilation took window outside of image limits.'
        assert yf<=self.NY,'dilation took window outside of image limits.'
        assert xf<=self.NX,'dilation took window outside of image limits.'
        self.data = self.data[:,:,:,:,:,yi:yf,xi:xf].copy()
        self.updateDimensions()
        self.Aligned = False
        print('Warning: we havent updated self.newNQ in this function yet so saving and reloading might have troubles.')

    
    def Segment(self,
                C,
                mask=False,
                returnSigNoise=False,
                oldOutput=True,
                maskOutput=False,
                method=False,
                erodeMask=False,
                blur_sig=False,
                addSeg2TData=False,
                returnSegmentations=True,
                closing=None,
                removeSmall=None,
                clearBorder=False):
        """
        Returns a mask of the segmented image using cv.threshold().
        
        Parameters
        ----------
        C : int or str
            The channel you will segment, either as its index in tdata.Chan 
            or the name of the channel.
        mask : str or False or array-like
            This is an image which defines a region of arbitrary shape to
            decide which pixels to include in the calculation of threshold. It 
            can be input as a string to indicate a template-matched mask (see 
            parameter in TData.ExtractProfileRectangle).      
        returnSigNoise : bool
            Whether to return the eroded and dilated-inverted masks for 
            definite signal and definite noise.
        oldOutput : bool
            Whether to save masks in the old output format. Or new way like 
            self.Cellpose.
        maskOutput : bool
            Whether to use the original mask to restrict the area of returned 
            masks. I.e. the mask is intially used to restrict area taken into 
            account for the threshold calculation, whereas this controls 
            whether regions outside that mask appear at all in the returned 
            masks.
        method : bool or cv parameter.
            Default is to use otsu. Put cv.THRESH_TRIANGLE to try that method.
        erodeMask : bool or ints
            If int then it will erode the first mask to stop side edge pixels 
            being taken into account in the threshold calculation.
        blur_sig : int or False
            Sometimes useful to blur the image a bit before segmentation 
            because cell pose does too detailed segmentation for high 
            resolution images. 
        addSeg2TData : bool
            If True then the calculated masks will be added as a new channel 
            to the TData.            
        returnSegmentations : bool
            Whether to return the calculated segmentation masks.   
        closing : None or int
            The rectangular kernel size of the morphological opening if you 
            chose to do it.
        removeSmall : None or int
            Remove objects with area smaller than this (in pixels).
        clearBorder : bool
            Whether to clear the segmentation mask border.
            
        Returns
        -------
        If oldOutput==True:
        segs : list of numpy array-like or list of lists of array-like
            The segmented images, one for each field. If return sigNoise then 
            each element is the list [seg,sig,noise].
            
        Notes
        -----
        Currently only works for tdata with NQ=1 for Q=T,M,Z.
        """

        _,_,_,_,C = self.parseTFMZC(C=C)

        if isinstance(mask,str):
            self.ParentXFold.buildMasks(mask)            
            
        segs = []
        for i,FID in enumerate(self.FieldIDs):
            
            # take the required bit of the image
            seg = self.data[0,i,0,0,C].copy()
            if isinstance(blur_sig,int):
                seg = filters.gaussian(seg,
                                         sigma=blur_sig,
                                         preserve_range=True).astype('uint16')            
            seg = tu.makeuint8(seg,1,0)            
            
            # build the mask
            if not mask:
                mask = np.ones(seg.shape)  
                mask3 = mask.copy()
            elif isinstance(mask,str):
                # need to know padding applied to self.data wrt to templates
                if FID in self.TemplatePadDic.keys():
                    pd = self.TemplatePadDic[FID]
                else:
                    pd = [0,0]                
                mask = self.ParentXFold.MaskDic[mask][FID]
                pad1 = [[p//2,math.ceil(p/2)] for p in pd]
                mask = np.pad(mask,pad1).astype('int16')     
                mask3 = mask.copy()
                if erodeMask:
                    kernel2 = np.ones((erodeMask,erodeMask),np.uint8)
                    mask3 = cv.erode(mask3,kernel2)
            
            mask = mask.copy().astype('bool')
            mask3 = mask3.copy().astype('bool')
            
            # threshold the chosen channel, only in mask region
            if not method:
                thr_param = cv.THRESH_BINARY+cv.THRESH_OTSU
            else:
                thr_param = cv.THRESH_BINARY+method
            ret,_ = cv.threshold(seg[mask3],0,255,thr_param)
            seg[seg<ret] = 0
            seg[seg>=ret] = 1
            
            if returnSigNoise:
                # nuclei and noise masks
                kernel = np.ones((3,3),np.uint8)
                sig = cv.erode(seg,kernel)
                noise = 1-cv.dilate(seg,kernel)
                if maskOutput:
                    mask2 = cv.erode(mask.astype('uint16'),kernel)
                    seg = np.logical_and(seg,mask2)
                    sig = np.logical_and(sig,mask2)
                    noise = np.logical_and(noise,mask2)  
                seg = [seg,sig,noise]
            elif maskOutput:
                seg = np.logical_and(seg,mask.astype('uint16'))
            if closing:
                kern = cv.getStructuringElement(cv.MORPH_ELLIPSE,(closing,closing))
                seg = cv.morphologyEx(seg, cv.MORPH_CLOSE, kern)
            if removeSmall:
                seg = remove_small_objects(seg.astype('bool'), min_size=removeSmall, connectivity=1)
            if clearBorder:
                seg = clear_border(seg)                
            
            segs.append(seg)

        if not oldOutput:
            segsArray = np.zeros((self.NT,self.NF,self.NM,self.NZ,self.NY,self.NX),dtype='uint16')
            for i,seg in enumerate(segs):
                segsArray[:,i,:,:] = seg
            
        if addSeg2TData:
            assert not oldOutput,'set oldOutput==False for addSeg2TData'
            self.AddArray(segsArray[:,:,:,:,np.newaxis,:,:],'Segmentation')     

        if returnSegmentations:
            return segs
   
        
    def Cellpose(self,
                 diameter,
                 seg_chan,
                 nuc_chan=None,
                 model_type='nuclei',
                 flow_threshold=0.4,
                 cellprob_threshold=0,
                 normalise='auto',
                 blur_sig=None,
                 clear_borderQ=False,
                 remove_small=None,
                 addSeg2TData=True,
                 saveSegmentationsAndDelete=None,                 
                 printWarnings=True,
                 verbose=False,
                 compress=False):
        """
        This uses cellpose to segment the channel you specify. It can save the 
        segmentation masks as a new channel in self, or return them as an 
        array. So far we always segment all TFMZ in self so that there are no 
        problems adding it as a new channel but we could change this...

        Could also save files etc but for now the user can just do 
        tdata.SaveData() or use XFold.Cellpose() which does file saving.

        Parameters
        ----------
        diameter : float or array of floats
            The diameter of the thing being segmented in um. If array then the 
            shape must match the shape of the array minus the last 2 'XY' 
            dimensions.
        seg_chan : int or str
            The channel index or name you want to segment.
        nuc_chan : int or str
            If you are doing a cytoplasm segmentation you can also 
            provide a nucleus channel to help. Leave this as None if you don't 
            have one.
        model_type : 'cyto' or 'nuclei' or 'cyto3'
            ...
        flow_threshold : float
            A parameter in the segmentation algorithm. Increase to get more 
            masks.
        cellprob_threshold : float
            A parameter in the segmentation algorithm. Decrease to get more 
            masks. It seems like this should be varied in the range -6 to 6.
        normalise : 'auto' or bool or float or int
            I noticed a problem with cellpose normalisation (bug raised on 
            github 4/11/24). They do normalisation to 0=1st percentile and 
            1=99th percentile but don't do clipping. This results in bad 
            segmentation in images with few nuclei which is solved by adding 
            clipping. So here the default ('auto') is to do our own 
            normalisation which is the same but with clipping so image range 
            is 0 to 1. There is also 'otsu' which finds what fraction X of 
            the image is in the otsu threshold segmentation and uses the 
            percentiles X/2 and 100 - X/2 instead of 1st and 99th. Or put True 
            or False to use cellpose normalisation or no normalisation. Or put 
            an int or float to control what percentile is used in 'auto'.
        blur_sig : int or False
            Sometimes useful to blur the image a bit before segmentation 
            because cellpose does too detailed segmentation for high 
            resolution images.
        clear_borderQ : bool
            Whether to remover the objects in the output labelled which 
            are connected to the border.
        remove_small : None or int or 'auto'
            If int then remove mask objects that have an area smaller than 
            this in um^2. If 'auto' then it removes objects with an effective 
            diameter more than 4 times smaller than the diameter you have 
            provided to cellpose.            
        addSeg2TData : bool
            If True then the calculated masks will be added as a new channel 
            to the TData.
        saveSegmentationsAndDelete : None or str
            Whether to save the segmentation masks. Str will be the name of 
            the folder they are saved into. This deletes everything except for 
            the segmentation just before saving because SaveData doesn't yet 
            have one channel only.            
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.
        verbose : bool
            Whether to print progress
        compress : bool
            Whether the saved files should be compressed or not.

        Returns
        --------
        masks : numpy array
            Has shape of data except for the channels axis removed.
        """

        from cellpose import models

        diameter = self.um_2_pixels(diameter,printWarnings)

        if remove_small=='auto':
            remove_small = (diameter/4)**2
        elif isinstance(remove_small,int):
            remove_small = self.um_2_pixels(
                            self.um_2_pixels(
                                remove_small,printWarnings),
                                            printWarnings)

        _,_,_,_,seg_chan = self.parseTFMZC(C=seg_chan)
        seg_chan = seg_chan[0]
                     
        if not nuc_chan is None:
            _,_,_,_,nuc_chan = self.parseTFMZC(C=nuc_chan)
            nuc_chan = nuc_chan[0]

        masks = np.zeros((self.NT,self.NF,self.NM,self.NZ,
                          self.NY,self.NX),dtype='uint16')
        
        model = models.Cellpose(gpu=False, model_type=model_type)

        dims = [self.NT,self.NF,self.NM,self.NZ]
        ranges = map(range,dims)
        for t,f,m,z in product(*ranges):
            if verbose:
                print('t: ',t,'f: ',f,'m: ',m,'z: ',z)
            if isinstance(diameter,np.ndarray):
                d2 = diameter[t,f,m,z]
            else:
                d2 = diameter
            if isinstance(nuc_chan,int):
                ch1 = [seg_chan,nuc_chan,nuc_chan]
                _data = self.data[t,f,m,z,ch1].copy()
                channels1 = [1,2]
            else:
                _data = self.data[t,f,m,z,seg_chan].copy().reshape((self.NY,
                                                                    self.NX))
                channels1 = [0,0]
            
            if normalise is True:
                norm1 = True
            elif normalise is False:
                norm1 = False
            elif normalise=='auto':
                norm1 = False
                _data = genF.normalise_image(_data,1)
            elif isinstance(normalise,float) or isinstance(normalise,int):
                norm1 = False
                _data = genF.normalise_image(_data,normalise)
            elif normalise=='otsu':
                norm1 = False
                _data = genF.normalise_image(_data,method='otsu')
                
            if isinstance(blur_sig,int):
                _data = filters.gaussian(_data,
                                         sigma=blur_sig,
                                         preserve_range=True)   
            
            masks[t,f,m,z],_,_,_ = model.eval(_data,
                                              diameter=d2,
                                              channels=channels1,
                                              flow_threshold=flow_threshold,
                                              cellprob_threshold=cellprob_threshold,
                                              normalize=norm1)

            del _data
            if clear_borderQ:
                masks[t,f,m,z] = clear_border(masks[t,f,m,z])
            if remove_small:
                masks[t,f,m,z] = remove_small_objects(masks[t,f,m,z],remove_small)
                
        if addSeg2TData or saveSegmentationsAndDelete:
            chan2_name = 'Segmentation_'
            chan2_name += self.Chan2[seg_chan] + '_'
            chan2_name += model_type + '_'
            if isinstance(diameter,float) or isinstance(diameter,int):
                chan2_name += 'D' + str(diameter).replace('.','p')
            self.AddArray(masks[:,:,:,:,np.newaxis,:,:],chan2_name)

        if saveSegmentationsAndDelete:
            self.TakeSubStack(C='Segmentation',updateNewSeshNQ=True)
            self.SaveData(saveSegmentationsAndDelete,compress=compress)            

    
    def YOLOSAM(self,
                 diameter,
                 nuc_chan,
                 yolo_model=None,
                 sam_predictor=None,
                 DEVICE=None,
                 clear_borderQ=False,
                 addSeg2TData=True,
                 printWarnings=True):
        """
        This uses a combination of YOLO and SAM to segment the channel you 
        specify. It can save the segmentation masks as a new channel in self, 
        or return them as an array. So far we always segment all TFMZ in self 
        so that there are no problems adding it as a new channel but we could 
        change this...

        Could also save files etc but for now the user can just do 
        tdata.SaveData() or use XFold.Cellpose() which does file saving.

        Parameters
        ----------
        diameter : float or array of floats
            The diameter of the thing being segmented in um. If array then the 
            shape must match the shape of the array minus the last 2 'XY' 
            dimensions.
        nuc_channel : int
            If you are doing a cytoplasm segmentation you can also 
            provide a nucleus channel to help. Leave this as None if you don't 
            have one.
        clear_borderQ : bool
            Whether to remover the objects in the output labelled which 
            are connected to the border.
        addSeg2TData : bool
            If True then the calculated masks will be added as a new channel 
            to the TData.
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.
            
        Returns
        --------
        masks : numpy array
            If returnSegmentations then: Has shape of data except for the 
            channels axis removed.
        """

             
        import torch

        if not DEVICE:
            DEVICE = 'cuda:3' if torch.cuda.is_available else 'cpu'
            
        if not yolo_model:
            from ultralytics import YOLO
            YOLO_CKPT = r'/weka/kgao/projects/jump-nuclei-detection/ckpts/best_model.pt'
            yolo_model = YOLO(YOLO_CKPT).to(DEVICE)
            
        if not sam_predictor:
            from segment_anything import SamPredictor,sam_model_registry
            SAM_CKPT = r'/weka/kgao/public-repos/segment-anything/sam-ckpts/sam_vit_b_01ec64.pth'
            SAM_MODEL_TYPE = 'vit_b'
            sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT).to(device=DEVICE) 
            sam_predictor = SamPredictor(sam_model)                     

        CLIP = [1, 99]    
        # Parameters
        AUGMENT = False
        CONF = 0.05
        IOU = 0.3
        MAX_DET = 3000
        
        yolo_params = {
            'clip': CLIP,
            'augment': AUGMENT,
            'conf': CONF,
            'iou': IOU,
            'max_det': MAX_DET,
        }                     
        
        diameter = self.um_2_pixels(diameter)

        masks = np.zeros((self.NT,self.NF,self.NM,self.NZ,self.NY,self.NX),dtype='uint16')
        
        MIN_OVERLAP = int(2.5*diameter)
        TILE_SHAPE = 1024
        
        dims = [self.NT,self.NF,self.NM,self.NZ]
        ranges = map(range,dims)
                     
        for t,f,m,z in product(*ranges):
            if isinstance(diameter,np.ndarray):
                d2 = diameter[t,f,m,z]
            else:
                d2 = diameter
                
            
            img = self.data[t,f,m,z,nuc_chan].copy()
            patches,N_tiles,olap,olap_f = genF.patchify2(img, TILE_SHAPE, MIN_OVERLAP)
            patches = np.reshape(patches,(patches.shape[0]*patches.shape[1],TILE_SHAPE,TILE_SHAPE))

            bboxes = list()
            for patch in patches:
                bbox = genF.nucl_detect_yolo(patch, yolo_model, **yolo_params)
                if isinstance(bbox,pd.DataFrame):
                    shp = patch.shape
                    bbox = bbox.to_numpy()
                    bbox[:, 0] = np.clip(np.rint(bbox[:, 0] * shp[1]), 0, shp[1])
                    bbox[:, 2] = np.clip(np.rint(bbox[:, 2] * shp[1]), 0, shp[1])
                    bbox[:, 1] = np.clip(np.rint(bbox[:, 1] * shp[0]), 0, shp[0])
                    bbox[:, 3] = np.clip(np.rint(bbox[:, 3] * shp[0]), 0, shp[0])
                    bbox = torch.tensor(bbox[:, 0:4].astype(int), device=DEVICE)
                else: 
                    bbox = np.array([])
                bboxes.append(bbox)

            # Perform segmentation
            masksP = np.zeros(patches.shape, dtype=int)
            for i, (patch, bbox) in enumerate(zip(patches, bboxes)):
                if len(bbox)==0:
                    continue
                patch_transformed = cv.normalize(patch, 
                                                 None, 
                                                 alpha=0, 
                                                 beta=255, 
                                                 norm_type=cv.NORM_MINMAX, 
                                                 dtype=cv.CV_8U)
                if len(patch_transformed.shape) != 3:
                    patch_transformed = cv.cvtColor(patch_transformed,
                                                    cv.COLOR_GRAY2RGB)
                    
                mask = genF.nucl_segment_sam(patch_transformed,bbox,sam_predictor)
                
                #mask = np.squeeze(mask) #tw changed this to next line:
                mask = np.reshape(mask,(mask.shape[0]*mask.shape[1],mask.shape[2],mask.shape[3]))
                mask2 = np.zeros([mask.shape[1], mask.shape[2]], dtype=int)
                for val, maski in enumerate(mask):
                    mask2 += maski * (val + 1)
                masksP[i] = mask2.astype(np.int16)
                
            masksP = np.reshape(masksP,(N_tiles,N_tiles,TILE_SHAPE,TILE_SHAPE))
            masks[t,f,m,z] = genF.unpatchify2(masksP,
                                              N_tiles,
                                              olap,
                                              olap_f,
                                              img.shape,
                                              TILE_SHAPE)

            del masksP
            del img
            
            if clear_borderQ:
                masks[t,f,m,z] = clear_border(masks[t,f,m,z])

        # need to get rid of nuclei because of split labels and patch boundaries
        masks = masks > 0       
        
        if addSeg2TData:
            self.AddArray(masks[:,:,:,:,np.newaxis,:,:],'Segmentation')
            


    
    def StructureTensor(self,
                        sig,
                        T='all',
                        F='all',
                        M='all',
                        Z='all',
                        C='all',
                        add2TData=False,
                        printWarnings=True):

        """
        This does an analysis very similar to orientationJ. See that for 
        details.
        
        Parameters
        ----------
        sig : float
            Standard deviation used for the Gaussian kernel applied over the 
            image. It assumes you have provided it in units of um but falls 
            back on units of pixels if your self doesn't have a good 
            self.ParentSession.pixUnit.
        T,F,M,Z,C : {'all',int,list,str}
            The indices of the data points that you want to process.
            'all' for all, int for one index, list for list of indices
            or for C you can request one channel or a list of channels by
            name(s) and for F you can request one field or list of fields by
            its ID(s).    
        add2self : bool
            Whether to add the calculalated arrays as new channels in the 
            self. The new channels will be called 'Orientations_sig',
            'Coherences_sig' and 'Energies_sig' where '_sig' is the sig value 
            used. Note how the calculated orientations array has a range of 
            -pi/2 to +pi/2, it is rescaled to range between 0 and 
            np.iinfo('uint16').max. Parts of self not analysed due to 
            T,F,M,Z,C sub-selection are left as zeros.
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.            
            
        Returns
        -------
        Note that the returned arrays will have the same shape as self.data if 
        you have add2self=True! Otherwise they will have the same dimensions 
        as the T,F,M,Z,C subselection.
        
        ori : ndarray
            The image of orientations.
        coh : ndarray
            The image of coherences.
        ene : ndarray   
            The image of energies.
        """
        
        sig = self.um_2_pixels(sig)      
            
        eps = 1e-20

        T,F,M,Z,C = self.parseTFMZC(T,F,M,Z,C)

        if not add2TData:
            dims = tuple([len(T),len(F),len(M),len(Z),len(C)])
            dimsYX = dims+(self.NY,self.NX)
            
            ori = np.zeros(dimsYX)
            coh = np.zeros(dimsYX)
            ene = np.zeros(dimsYX)  
        else:
            ori = np.zeros(self.Shape)
            coh = np.zeros(self.Shape)
            ene = np.zeros(self.Shape)          

        
        eT = enumerate(T)
        eF = enumerate(F)
        eM = enumerate(M)
        eZ = enumerate(Z)
        eC = enumerate(C)
        for (it,t),(iff,f),(im,m),(iz,z),(ic,c) in product(eT,eF,eM,eZ,eC):
            
            ayy, axy, axx = feature.structure_tensor(
                self.data[t,f,m,z,c].astype(np.float32), 
                sigma=sig, 
                mode="reflect",
                order="rc")
            
            l1, l2 = feature.structure_tensor_eigenvalues((axx, axy, ayy))
            if not add2TData:
                ori[it,iff,im,iz,ic] = np.arctan2(2 * axy, (ayy - axx)) / 2
                coh[it,iff,im,iz,ic] = ((l2 - l1) / (l2 + l1 + eps)) ** 2
                ene[it,iff,im,iz,ic] = np.sqrt(axx + ayy)
                ene[it,iff,im,iz,ic] /= ene[it,iff,im,iz,ic].max()  
            else:
                ori[t,f,m,z,c] = np.arctan2(2 * axy, (ayy - axx)) / 2
                coh[t,f,m,z,c] = ((l2 - l1) / (l2 + l1 + eps)) ** 2
                ene[t,f,m,z,c] = np.sqrt(axx + ayy)
                ene[t,f,m,z,c] /= ene[t,f,m,z,c].max() 

        if add2TData:
            ori2 = ((ori+np.pi/2)*(np.iinfo('uint16').max/np.pi))
            coh2 = coh*np.iinfo('uint16').max
            ene2 = ene*np.iinfo('uint16').max
            self.AddArray(ori2.astype('uint16'),'Orientations_'+str(sig))
            self.AddArray(coh2.astype('uint16'),'Coherences_'+str(sig))
            self.AddArray(ene2.astype('uint16'),'Energies_'+str(sig))
            
        return ori, coh, ene  



    def SwitchAxes(self,ax1,ax2):
        """
        This switches the axes of the image data.

        This was made as a rafistolage for some data that had been saved in 
        imagej with the time axis as the z axis so needed swapping. Note that 
        only the selfs dimensions are updated, so you could run into 
        problems with things disagreeing between the self and the session. 
        It would be complicated to solve this at the session level because 
        e.g. of makeTData.
        """
        assert ax1!=4,'swiching channel axis not supported yet because e.g. self.Chan would also need sorting...'
        assert ax2!=4,'swiching channel axis not supported yet because e.g. self.Chan would also need sorting...'
        self.data = np.swapaxes(self.data, ax1, ax2)
        self.updateDimensions()
        if ax1==0 or ax2==0:
            self.newSeshNT = self.NT
            self.SeshT = [None]*self.NT
        if ax1==2 or ax2==2:
            self.newSeshNM = self.NM
            self.SeshM = [None]*self.NM
        if ax1==3 or ax2==3:
            self.newSeshNZ = self.NZ
            self.SeshZ = [None]*self.NZ
        
        return
        

    def Plot(self,
             channels='all',
             plotSegMask=None,
             plotSegMask2=None,
             region=False,
             plotSize=10,
             colouriseMask=False):
        """
        This plots the data with interactive sliders. So far only plots 4 
        channels.
        
        Parameters
        ----------
        channels : 'all' or int or list of int or str
            Which channels to show. Maximum number is 4. Any Segmentation 
            channels will be removed even if you ask for them. Accepts any of 
            the normal input formats (see parseTFMCZ()).
        plotSegMask,plotSegMask2 : None or bool or str
            If None then it will plot the mask if there is a Segmentation 
            channel in self. If False then it won't. If True then it plots the 
            masks if there is a Segmnetation channel in self and raises an 
            error if it can't find one. If you want to plot masks that are 
            saved in a folder in your XFold's parent folder or stored in the 
            ParentXFold.SegmentationXFolds then this should be the string name 
            that is the key in that dict (which is also the folder name where 
            they are saved). 
        region : False or (yi,yf,xi,xf)
            Give the region by pixel index (numpy format) if you want to 
            plot just a smaller region.
        plotSize : int
            If you set this bigger you will be able to make the plot bigger 
            by changing the browser window size but things will be slower.
        colouriseMask : bool
            If False then the Segmentation channel will be plotted in Yellow. 
            If True then it will assign varried RGB values to different 
            objects in the mask. This is to help show whether masks that are 
            touching have different labels.
            
        Notes
        -----
        remember you will need to have in the cell: %matplotlib widget
        """
        
        all_colours = ['Red', 'Green', 'Blue','Cyan','Magenta','Yellow','Greys',
                       'Hue (HSV)','Saturation (HSV)','Value (HSV)']
        RGB_colours = ['Red', 'Green', 'Blue','Cyan','Magenta','Yellow','Greys']
        HSV_colours = ['Hue (HSV)','Saturation (HSV)','Value (HSV)']
        
        NCh = len([c for c in self.Chan if not 'Segmentation' in c])
        
        _,_,_,_,C = self.parseTFMZC(channels)
        C = [c for c in C if not 'Segmentation' in self.Chan[c]]
        if len(C)>4:
            C = C[:4]
        
        if not region:
            yi,xi = (0,0)
            _,_,_,_,_,yf,xf = self.data.shape
        else:
            yi,yf,xi,xf = region

        if plotSegMask==None:
            if 'Segmentation' in self.Chan:
                plotSegMask = True
        if plotSegMask==True:
            assert 'Segmentation' in self.Chan,EM.pl1
        elif isinstance(plotSegMask,str):
            xfoldS = self.get_segmentation_xfold(plotSegMask)
            sidx = self.ParentSession.SessionN
            tdataS = xfoldS.makeTData(S=sidx,
                                      T=self.SeshT,
                                      F=self.SeshF,
                                      M=self.SeshM,
                                      Z=self.SeshZ)
        if plotSegMask2==None:
            if len([c for c in self.Chan if c=='Segmentation'])==2:
                plotSegMask2 = True
        if plotSegMask2==True:
            assert len([c for c in self.Chan if c=='Segmentation'])==2,EM.pl2
        elif isinstance(plotSegMask2,str):
            xfoldS2 = self.get_segmentation_xfold(plotSegMask2)
            sidx = self.ParentSession.SessionN
            tdataS2 = xfoldS2.makeTData(S=sidx,
                                      T=self.SeshT,
                                      F=self.SeshF,
                                      M=self.SeshM,
                                      Z=self.SeshZ)            

        header1 = widgets.Label(value='Data Selection:',
                                layout=widgets.Layout(justify_content='center', 
                                                                width='75%'))
        Ts = widgets.IntSlider(0,0, self.NT-1,1,description='T')
        Fs = widgets.IntSlider(0,0, self.NF-1,1,description='F')
        Ms = widgets.IntSlider(0,0, self.NM-1,1,description='M')
        Zs = widgets.IntSlider(0,0, self.NZ-1,1,description='Z')
        
        ui16max = np.iinfo('uint16').max

        header2 = widgets.Label(value='Clip range:',
                                layout=widgets.Layout(justify_content='center', 
                                                                width='75%'))
        vClipC1 = widgets.IntRangeSlider(value=[0,ui16max],
                                         min=0,max=ui16max,step=1,disabled=True)                 
        vClipC2 = widgets.IntRangeSlider(value=[0,ui16max],
                                         min=0,max=ui16max,step=1,disabled=True)
        vClipC3 = widgets.IntRangeSlider(value=[0,ui16max],
                                         min=0,max=ui16max,step=1,disabled=True)
        vClipC4 = widgets.IntRangeSlider(value=[0,ui16max],
                                         min=0,max=ui16max,step=1,disabled=True)
        vClipCSeg = widgets.FloatSlider(value=1,
                                        min=0,max=1,step=0.01,disabled=False)
        vClipCSeg2 = widgets.FloatSlider(value=1,
                                        min=0,max=1,step=0.01,disabled=False)                 

        header3 = widgets.Label(value='',
                                layout=widgets.Layout(justify_content='center', 
                                                               width='75%'))     
        checkboxC1 = widgets.Checkbox(value=False,
                                      description='C1 - ',disabled=True,
                                      layout=widgets.Layout(width='max-content'))
        checkboxC2 = widgets.Checkbox(value=False,
                                      description='C2 - ',disabled=True,
                                      layout=widgets.Layout(width='max-content'))
        checkboxC3 = widgets.Checkbox(value=False,
                                      description='C3 - ',disabled=True,
                                      layout=widgets.Layout(width='max-content'))
        checkboxC4 = widgets.Checkbox(value=False,
                                      description='C4 - ',disabled=True,
                                      layout=widgets.Layout(width='max-content'))
        ChSegMc = widgets.Checkbox(value=False,
                                   description='ChSegMc',
                                   layout=widgets.Layout(width='max-content'))
        ChSegMc2 = widgets.Checkbox(value=False,
                                   description='ChSegMc2',
                                   layout=widgets.Layout(width='max-content'))                 

        header4 = widgets.Label(value='Clip limits:',
                                layout=widgets.Layout(justify_content='center', 
                                                                  width='95%'))
        autolimC1 = widgets.BoundedFloatText(value=0,min=0,max=40.0,step=0.02,
                                             layout=widgets.Layout(width='max-content'))
        autolimC2 = widgets.BoundedFloatText(value=0,min=0,max=40.0,step=0.02,
                                             layout=widgets.Layout(width='max-content'))
        autolimC3 = widgets.BoundedFloatText(value=0,min=0,max=40.0,step=0.02,
                                             layout=widgets.Layout(width='max-content'))
        autolimC4 = widgets.BoundedFloatText(value=0,min=0,max=40.0,step=0.02,
                                             layout=widgets.Layout(width='max-content'))
        
        axesOn = widgets.Checkbox(value=True,
                                  description='axes',
                                  layout=widgets.Layout(width='max-content',
                                                        margin='0 -15px 0 0'))
        titleOn = widgets.Checkbox(value=True,
                                  description='title',
                                  layout=widgets.Layout(width='max-content',
                                                        margin='0 0 0 -20px'))      

        header5 = widgets.Label(value='Select LUT:',
                                layout=widgets.Layout(justify_content='center', 
                                                                width='75%'))
        col_dropC1 = widgets.Dropdown(options=all_colours,
                                      layout={'width':'max-content'},
                                      disabled=True)                 
        col_dropC2 = widgets.Dropdown(options=all_colours,
                                      layout={'width':'max-content'},
                                      disabled=True)
        col_dropC3 = widgets.Dropdown(options=all_colours,
                                      layout={'width':'max-content'},
                                      disabled=True)
        col_dropC4 = widgets.Dropdown(options=all_colours,
                                      layout={'width':'max-content'},
                                      disabled=True)
        
        def save_image(b):
            name = 'S'+str(self.ParentSession.SessionN).zfill(3)
            name += '_T'+str(self.SeshT[Ts.value]).zfill(3)
            name += '_F'+str(self.SeshF[Fs.value]).zfill(3)
            name += '_M'+str(self.SeshM[Ms.value]).zfill(3)
            name += '_Z'+str(self.SeshZ[Zs.value]).zfill(3)
            if checkboxC1.value:
                name += '_C1'
            if checkboxC2.value:
                name += '_C2'        
            if checkboxC3.value:
                name += '_C3'        
            if checkboxC4.value:
                name += '_C4'        
            plot_ax123.figure.savefig('./'+name+'.png')
        
        save_button = widgets.Button(description="Save Image")
        save_button.on_click(save_image)
        
        if NCh>3:
            vClipC4.disabled = False
            checkboxC4.disabled = False
            checkboxC4.description='Ch4 - '+self.Chan2[C[3]]
            autolimC4.disabled = False
            col_dropC4.disabled = False
            col_dropC4.value = genF.chan2colour(self.Chan[C[3]])
        if NCh>2:
            vClipC3.disabled = False
            checkboxC3.disabled = False
            checkboxC3.description='Ch3 - '+self.Chan2[C[2]]
            autolimC3.disabled = False
            col_dropC3.disabled = False    
            col_dropC3.value = genF.chan2colour(self.Chan[C[2]])
        if NCh>1:
            vClipC2.disabled = False
            checkboxC2.disabled = False
            checkboxC2.description='Ch2 - '+self.Chan2[C[1]]
            autolimC2.disabled = False
            col_dropC2.disabled = False    
            col_dropC2.value = genF.chan2colour(self.Chan[C[1]])
        if NCh>0:
            vClipC1.disabled = False
            checkboxC1.disabled = False
            checkboxC1.value = True
            checkboxC1.description='Ch1 - '+self.Chan2[C[0]]
            autolimC1.disabled = False
            col_dropC1.disabled = False    
            col_dropC1.value = genF.chan2colour(self.Chan[C[0]])            
            
        if not plotSegMask:
            ChSegMc.disabled = True
            vClipCSeg.disabled = True
        if not plotSegMask2:
            ChSegMc2.disabled = True
            vClipCSeg2.disabled = True 
        
        if self.NT<2:
            Ts.disabled = True
        if self.NF<2:
            Fs.disabled = True  
        if self.NM<2:
            Ms.disabled = True
        if self.NZ<2:
            Zs.disabled = True       
            
        def update(T,F,M,Z,
                   min_maxC1,min_maxC2,min_maxC3,min_maxC4,seg_alpha,seg_alpha2,
                   C1,C2,C3,C4,ChSegM,ChSegM2,autoC1,autoC2,autoC3,autoC4,axes,title,
                  colour1,colour2,colour3,colour4):
        
            global plot_ax123

            cNs = [C1,C2,C3,C4]
            colNs = [colour1,colour2,colour3,colour4]
            active_channels = [clr for cc,clr in zip(cNs,colNs) if cc]
                           
            fig, plot_ax123 = plt.subplots(figsize=(plotSize,plotSize))
            plot_ax123.tick_params(axis='x', labelsize=plotSize)
            plot_ax123.tick_params(axis='y', labelsize=plotSize)
            overlay_image = np.zeros((yf-yi,xf-xi,3),dtype='uint16')
                      
            if autoC1!=0:
                vClipC1.min = np.percentile(self.data[T,F,M,Z,C[0],yi:yf,xi:xf],autoC1)        
                vClipC1.max = np.percentile(self.data[T,F,M,Z,C[0],yi:yf,xi:xf],100-autoC1)
            else:
                vClipC1.min = 0         
                vClipC1.max = ui16max  
            if autoC2!=0:
                vClipC1.min = np.percentile(self.data[T,F,M,Z,C[1],yi:yf,xi:xf],autoC2)        
                vClipC2.max = np.percentile(self.data[T,F,M,Z,C[1],yi:yf,xi:xf],100-autoC2)
            else:
                vClipC2.min = 0        
                vClipC2.max = ui16max  
            if autoC3!=0:     
                vClipC3.min = np.percentile(self.data[T,F,M,Z,C[2],yi:yf,xi:xf],autoC3)        
                vClipC3.max = np.percentile(self.data[T,F,M,Z,C[2],yi:yf,xi:xf],100-autoC3)
            else:
                vClipC3.min = 0        
                vClipC3.max = ui16max  
            if autoC4!=0:
                vClipC4.min = np.percentile(self.data[T,F,M,Z,C[3],yi:yf,xi:xf],autoC4)
                vClipC4.max = np.percentile(self.data[T,F,M,Z,C[3],yi:yf,xi:xf],100-autoC4)
            else:
                vClipC4.min = 0
                vClipC4.max = ui16max          
                
            if C1:  
                im1 = exposure.rescale_intensity(self.data[T,F,M,Z,C[0],yi:yf,xi:xf],
                                                 in_range=(min_maxC1[0],min_maxC1[1]),
                                                 out_range='uint8')
                overlay_image = genF.combine_images(overlay_image,im1,
                                                    colour1,
                                                    active_channels)
                
            if C2:
                im2 = exposure.rescale_intensity(self.data[T,F,M,Z,C[1],yi:yf,xi:xf],
                                                 in_range=(min_maxC2[0],min_maxC2[1]),
                                                 out_range='uint8')
                overlay_image = genF.combine_images(overlay_image,im2,
                                                    colour2,
                                                    active_channels)
                
            if C3:
                im3 = exposure.rescale_intensity(self.data[T,F,M,Z,C[2],yi:yf,xi:xf],
                                                 in_range=(min_maxC3[0],min_maxC3[1]),
                                                 out_range='uint8')
                overlay_image = genF.combine_images(overlay_image,im3,
                                                    colour3,
                                                    active_channels)
                
            if C4:
                im4 = exposure.rescale_intensity(self.data[T,F,M,Z,C[3],yi:yf,xi:xf],
                                                 in_range=(min_maxC4[0],min_maxC4[1]),
                                                 out_range='uint8')
                overlay_image = genF.combine_images(overlay_image,im4,
                                                    colour4,
                                                    active_channels)
                
            if ChSegM:
                if plotSegMask==True:
                    segIm = self.data[T,
                    F,
                    M,
                    Z,
                    self.chan_index('Segmentation'),
                    yi:yf,
                    xi:xf].copy().astype('uint16')
                else:
                    segIm = tdataS.data[T,
                    F,
                    M,
                    Z,
                    0,
                    yi:yf,
                    xi:xf].copy().astype('uint16')
                if colouriseMask:
                    segIm = genF.label2rgb2(segIm)*255
                    segCol = 'RGB'
                else:
                    segIm[segIm!=0] = 255
                    segCol = 'Yellow'
                segIm = (segIm*seg_alpha).astype('uint16')
                overlay_image = genF.combine_images(overlay_image,
                                                    segIm,
                                                    segCol,
                                                    active_channels)
            if ChSegM2:
                if plotSegMask2==True:
                    seg = 'Segmentation'
                    seg_ind = [i for i,c in enumerate(self.Chan) if c==seg][1]
                    segIm2 = self.data[T,
                    F,
                    M,
                    Z,
                    seg_ind,
                    yi:yf,
                    xi:xf].copy().astype('uint16')                    
                else:
                    segIm2 = tdataS2.data[T,
                    F,
                    M,
                    Z,
                    0,
                    yi:yf,
                    xi:xf].copy().astype('uint16')
                if colouriseMask:
                    segIm2 = genF.label2rgb2(segIm2)*255
                    segCol = 'RGB'
                else:
                    segIm2[segIm2!=0] = 255
                    segCol = 'Yellow'
                segIm2 = (segIm2*seg_alpha2).astype('uint16')
                overlay_image = genF.combine_images(overlay_image,
                                                    segIm2,
                                                    segCol,
                                                    active_channels)                
        
            if overlay_image.dtype=='uint16':
                overlay_image = np.clip(overlay_image, 0, 255)
            elif overlay_image.dtype=='float32':
                overlay_image = np.clip(overlay_image, 0, 1)
                overlay_image = hsv_to_rgb(overlay_image)
        
            if not axes:
                plot_ax123.set_axis_off()
            if title:
                title_text = self.ParentSession.Name.split('->')[-1] + '   FieldID: ' + self.FieldIDs[F]
                plot_ax123.set_title(title_text,fontsize=int(plotSize*1.5))
            plot_ax123.imshow(overlay_image)
        
        ax_tit_box = widgets.HBox([axesOn,titleOn])
        
        sliders_box1 = widgets.VBox([header1,Ts,Fs,Ms,Zs,ax_tit_box])
        
        sliders_box2 = widgets.VBox([header2,
                                     vClipC1,
                                     vClipC2,
                                     vClipC3,
                                     vClipC4,
                                     vClipCSeg,
                                     vClipCSeg2])        

        auto_lims_box = widgets.VBox([header4,autolimC1,
                                      autolimC2,
                                      autolimC3,
                                      autolimC4],
                                      layout=widgets.Layout(margin='0 0 0 0'))        
        
        checkboxes_box = widgets.VBox([header3,
                                       checkboxC1,
                                       checkboxC2,
                                       checkboxC3,
                                       checkboxC4,
                                       ChSegMc,
                                       ChSegMc2],
                                      layout=widgets.Layout(margin='0 0 0 0'))
        
        col_select_box = widgets.VBox([header5,
                                       col_dropC1,
                                       col_dropC2,
                                       col_dropC3,
                                       col_dropC4,
                                       save_button],
                                       layout=widgets.Layout(margin='0 0 0 25px'))
        
        widgets_box = widgets.HBox([sliders_box1,
                                    sliders_box2,
                                    auto_lims_box,                                    
                                    checkboxes_box,
                                    col_select_box])
        
        output = widgets.interactive_output(update, 
                                            {'T':Ts,
                                            'F':Fs,
                                            'M':Ms,
                                            'Z':Zs,
                                            'min_maxC1':vClipC1,
                                            'min_maxC2':vClipC2,
                                            'min_maxC3':vClipC3,
                                            'min_maxC4':vClipC4,
                                            'seg_alpha':vClipCSeg,
                                            'seg_alpha2':vClipCSeg2,                                             
                                            'C1':checkboxC1,
                                            'C2':checkboxC2,
                                            'C3':checkboxC3,
                                            'C4':checkboxC4,
                                            'ChSegM':ChSegMc,  
                                            'ChSegM2':ChSegMc2,                                               
                                            'autoC1':autolimC1,
                                            'autoC2':autolimC2,
                                            'autoC3':autolimC3,
                                            'autoC4':autolimC4,
                                            'axes':axesOn,
                                            'title':titleOn,
                                            'colour1':col_dropC1,
                                            'colour2':col_dropC2,
                                            'colour3':col_dropC3,
                                            'colour4':col_dropC4})
        
        display(widgets_box,output)
    
    
    def chan_string_index(self,ch):
        """
        Provide one string and it returns the channel index of this channel in 
        your tdata if it can find it. Error otherwise.
        
        It first checks if there are any 
        channel names that match in the raw channel name, i.e. self.Chan2. 
        Then it will check if the regularised version of your entry matches 
        any of the regularised channel names in self.Chan (regularisation is 
        done with generalFunctions.chanDic).  

        ch : str
            Your string name of channel you're looking for.
        """
        if ch in self.Chan2:
            ch = self.Chan2.index(ch)
        else:
            ch2 = genF.chanDic(ch)
            assert ch2 in self.Chan,'chan_index error: '+ch+'not in self.Chan'
            ch = self.Chan.index(ch2)
        return ch

        
    def chan_index(self,chans):
        """
        Give this a string channel name or list (or even list of list, see 
        parameters) of channels and it returns the index/indices of the 
        channel/s.

        For channels provided as strings, it first checks if there are any 
        channel names that match in the raw channel name, i.e. self.Chan2. 
        Then it will check if the regularised version of your entry matches 
        any of the regularised channel names in self.Chan (regularisation is 
        done with generalFunctions.chanDic).

        Parameters
        ----------
        chans : str or int or list of str or list of lists of str or int
            If string then it returns the index of channel with that name. 
            If list then it does that for a list of names. If list of list 
            then each sublist is all names it searches for that channel 
            and it returns the index of the first one that it finds. Note: 
            behaviour of lists of lists not sorted w.r.t. Chan vs Chan2!
        """ 
        if isinstance(chans,str):
            chans = self.chan_string_index(chans)
        elif isinstance(chans,int):
            pass
        elif isinstance(chans,list):
            intsQ = [isinstance(c,int) for c in chans]
            stringsQ = [isinstance(c,str) for c in chans]
            listsQ = [isinstance(c,list) for c in chans]
            if all(intsQ):
                return chans
            if all(stringsQ):
                chans = [self.chan_string_index(c) for c in chans]
                if len(chans)==1:
                    chans = chans[0]
            elif all(listsQ):
                chans2 = []
                for chs in chans:
                    inChanQ = [c in self.Chan for c in chs]
                    assert any(inChanQ),'chan_index error: none of chan sublist found in self.Chan'
                    for c in chs:
                        assert isinstance(c,str), 'chan_index error: sublist not strings'
                        if c in self.Chan:
                            chans2.append(self.Chan.index(c))
                            continue
                if len(chans2)==1:
                    chans2 = chans2[0]
                chans = chans2            
                        
            else:
                raise Exception('chan_index error: chans format not handled')

        return chans


    def mergeTData(self,tdata2):
        """
        This seems to merge the provided tdata2 into self. Currently they must 
        come from the same Session. Seems like SeshQ of resulting self is well 
        ordered?
        """

        assert self.ParentSession==tdata2.ParentSession,EM.mg1 

        difAx = sum([a!=b for a,b in zip(self.SeshQ,tdata2.SeshQ)])
        assert difAx==1,EM.mg2

        changeQ = [a==b for a,b in zip(self.SeshQ,tdata2.SeshQ)].index(False)

        overlap = any([b in self.SeshQ[changeQ] for b in tdata2.SeshQ[changeQ]])
        assert not overlap,EM.mg3 
        
        sorted_sesh = sorted(self.SeshQ[changeQ] + tdata2.SeshQ[changeQ])
        
        new_shape = list(self.data.shape)
        new_shape[changeQ] = self.data.shape[changeQ]+tdata2.data.shape[changeQ]
        data3 = np.zeros(new_shape,dtype='uint16')
        
        for i,q in enumerate(self.SeshQ[changeQ]):
            ind = sorted_sesh.index(q)
            slices = [slice(None)]*self.data.ndim
            slices2 = [slice(None)]*self.data.ndim
            slices[changeQ] = ind
            slices2[changeQ] = i
            slices = tuple(slices)
            slices2 = tuple(slices2)
            data3[slices] = self.data[slices2]
        
        for i,q in enumerate(tdata2.SeshQ[changeQ]):
            ind = sorted_sesh.index(q)
            slices = [slice(None)]*tdata2.data.ndim
            slices2 = [slice(None)]*tdata2.data.ndim
            slices[changeQ] = ind
            slices2[changeQ] = i
            slices = tuple(slices)
            slices2 = tuple(slices2)
            data3[slices] = tdata2.data[slices2]

        self.data = data3.copy().astype('uint16')

        del data3
        
        if changeQ==0:
            self.SeshT = sorted_sesh
            # getTimes() works from SeshT and FieldIDs so this should work
            self.Times = self.getTimes()
        elif changeQ==1:
            self.SeshF = sorted_sesh
            self.FieldIDs = [self.ParentSession.FieldIDMap[f] for f in self.SeshF]
        elif changeQ==2:
            self.SeshM = sorted_sesh
        elif changeQ==3:
            self.SeshZ = sorted_sesh
        elif changeQ==4:
            self.SeshC = sorted_sesh   
            self.Chan = [self.ParentSession.Chan[c] for c in self.SeshC]
            self.Chan = [genF.chanDic(c) for c in self.Chan]
            self.startChan = tuple(self.Chan)
            
        self.SeshQ[changeQ] = sorted_sesh 

        self.updateDimensions()
        print('Warning: we havent updated self.newSeshNQ in this function yet so saving and reloading might have troubles.')


    def Duplicate(self):
        """
        Returns a copy of the tdata. We can't just use deepcopy because we 
        don't want to create e.g. a new parentSession, we want them to share 
        the same one, as well as xfold. But all attributes of tdata that are 
        not a parentSession or parentXFold are deepcopied. We also can't just 
        initiate a new TData with the tdatas attributes because some 
        attributes are automatically set from the parentSession but the 
        tdatas may have since changed and we want those changed ones.
        """

        tdata2 = TData(self.ParentTFiles,
                       self.data.copy(),
                       self.SeshT.copy(),
                       self.SeshF.copy(),
                       self.SeshM.copy(),
                       self.SeshZ.copy(),
                       self.SeshC.copy(),
                       self.ParentSession)
        
        tdata2.pixSizeY = self.pixSizeY
        tdata2.pixSizeX = self.pixSizeX
        tdata2.pixUnit = self.pixUnit

        tdata2.Chan = self.Chan.copy()
        tdata2.Chan2 = self.Chan2.copy()
        tdata2.startChan = tuple(tdata2.Chan)

        tdata2.Aligned = self.Aligned

        tdata2.newSeshNT = self.newSeshNT
        tdata2.newSeshNF = self.newSeshNF
        tdata2.newSeshNM = self.newSeshNM
        tdata2.newSeshNZ = self.newSeshNZ
        tdata2.newSeshNC = self.newSeshNC
        tdata2.newSeshNX = self.newSeshNX
        tdata2.newSeshNY = self.newSeshNY
        

        return tdata2


    def PIV(self,
            winsize,
            searchsize,
            overlap,
            dt,
            T='all',
            F='all',
            M='all',
            Z='all',
            C='all',
            outputDivArray=True,
            addDiv2TData=False):
        """
        This uses openpiv to calculate PIV velocity and divergence fields on 
        consecutive time points in the tdata.
        
        Parameters
        ----------
        winsize : int
            kkk
        searchsize : int
            kkk
        overlap : int
            jjj
        dt : float
            jjj
        T,F,M,Z,C : int or list of str
            The normal selection parameters.
        outputDivArray : bool
            Whether or not to output the divergence fields as a 7D float 
            array. The final array is the shape of your selection (with one 
            less time point) so may not match you tdata!
        addDiv2TData : bool
            Whether or not to add the divergence fields as new channels in the 
            tdata. The pixel values in the initial divergence field have 
            physical units so converting to uint16 needs to keep track of 
            that. But for now we just multiply by 10000. The negative values 
            are made positive and put in channel 'DivNeg'+ChanName and the 
            positive in 'DivPos'+ChanName. Note that you may not have selected 
            all the data points in the tdata so these bits are left blank in 
            the added channels (as is the first time point where there was 
            nothing to compare with). (Also note there are two new channels 
            per channel that you applied the PIV to).
        """
                
        import openpiv
        from openpiv import tools, pyprocess, validation, filters, scaling
                
        TT,FF,MM,ZZ,CC = self.parseTFMZC(T,F,M,Z,C)

        if outputDivArray:
            outDivDims = (len(TT)-1,len(FF),len(MM),len(ZZ),len(CC),self.NY,self.NX)
            outputDiv = np.zeros(outDivDims,dtype='float64')
        if addDiv2TData:
            addDivDims = (self.NT,self.NF,self.NM,self.NZ,2*len(CC),self.NY,self.NX)
            addDiv = np.zeros(addDivDims,dtype='uint16')            
       
        for it,t in enumerate(TT):
            if t==0:
                continue
            for (iff,f),(im,m),(iz,z),(ic,c) in product(enumerate(FF),enumerate(MM),enumerate(ZZ),enumerate(CC)):
                
                im1 = self.data[t-1,f,m,z,c].copy()
                im2 = self.data[t,f,m,z,c].copy()
                
                u0, v0, sig2noise = openpiv.pyprocess.extended_search_area_piv(
                    im1.astype(np.int32),
                    im2.astype(np.int32),
                    window_size=winsize,
                    overlap=overlap,
                    dt=dt,
                    search_area_size=searchsize,
                    sig2noise_method='peak2peak'
                )        

                x, y = pyprocess.get_coordinates(
                    image_size=im1.shape,
                    search_area_size=searchsize,
                    overlap=overlap,
                )

                invalid_mask = validation.sig2noise_val(u0,v0,
                    sig2noise,
                    threshold = 1.05
                )                

                u2, v2 = filters.replace_outliers(
                    u0, v0,
                    invalid_mask,
                    method='localmean',
                    max_iter=3,
                    kernel_size=3,
                )

                x, y, u3, v3 = scaling.uniform(
                    x, y, u2, v2,
                    scaling_factor = 100,
                )

                x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)         

                ux = gaussian_filter(u3, sigma=1, order=[1, 0])
                vy = gaussian_filter(v3, sigma=1, order=[0, 1])
                divergence = ux + vy   
                
                resized_divergence = zoom(divergence,
                                          (self.NY/ux.shape[0],
                                           self.NX/ux.shape[1]),
                                          order=3)
                
                if outputDivArray:
                    outputDiv[it-1,iff,im,iz,ic] = resized_divergence.copy()

                if addDiv2TData:
                    divNeg = resized_divergence.copy()
                    divNeg[divNeg>0] = 0
                    divNeg = np.round(np.abs(divNeg)*10000).astype('uint16')

                    divPos = resized_divergence.copy()
                    divPos[divPos<0] = 0
                    divPos = np.round(np.abs(divPos)*10000).astype('uint16')

                    addDiv[t,f,m,z,2*ic] = divNeg
                    addDiv[t,f,m,z,(2*ic) + 1] = divPos
        
        if addDiv2TData:
            chan_names = [self.Chan[c] for c in CC]
            chan_names = [['DivNeg'+c,'DivPos'+c]  for c in chan_names]
            chan_names = [cc for c in chan_names for cc in c]
            
            for i,chan_name in enumerate(chan_names):
                self.AddArray(addDiv[:,:,:,:,[i]].copy(),chan_name)

        if outputDivArray:
            return outputDiv


    def RegionProps(self,
                    Segmentation='Segmentation',
                    fun=[],
                    returnDF=True,
                    saveDF=False,
                    verbose=False,
                    tracking=False):
        """
        This takes the Segmentation channel of the TData and labels regions 
        and returns a pandas dataframe of measurements, a row for each region, 
        a la skimage regionprops. The returned dataframe also includes XFold 
        indices T,F,M,Z (i.e. the SeshQ) and XPath, SessionN and Session.Name 
        in the dataframe so the link with original data is maintained. By 
        default it measures all the normal skimage regionprops properties of 
        each region but more are added via func (see parameters below).

        Parameters
        ----------
        Segmentation : str
            This is the ONE channel to use for the first application of 
            regionProps, i.e. nothing to do with fun. Can be provided as Chan2 
            or Chan name and if it isn't found there it will attempt to load 
            as a Segmentation channel (in which case you can provide as 
            absolute or relative path or just folder name if it is located in 
            XPathP.)
            !! Important note !! - this can be a labelled segmentation and it 
            shouldn't mess label ordering up by trying to relabel... but the 
            decision of whether it is labelled or not is quite basic (see 
            genF.isLabelledMaskQ)
            !! Important note if using extra functions !! - The image sent to 
            the function is cropped from the raw data according to the limits 
            of each region in this segmentation. So all regions you want to 
            use in functions must be within this segmentation. E.g. if you are 
            measuring nuclear and cytoplasmic signals, put the cytoplasmic 
            Segmentations here and the nuclear segmentations in the function.
            !! Important note 2 if using extra functions !! - the label from 
            this mask will be passed to the function and it may be assumed 
            (depending on the function) that segmentations passed directly to 
            the function have corresponding labels.
        fun : list [func,[chans],[params]]
            Functions that extract additional measurements from your regions. 
            Each function in this list is applied once to every region 
            found in the original Segmentation, returning a dictionary in 
            the format {"measurement_name":measurement_value,...} - the full 
            list of these dictionaries (one for each region) is then 
            concatenated to the region dataframe making new columns. 
            
            'fun' Parameters
            ----------------
            func (list element 1) : function
                The function itself. 
            chans (i.e. element 2) : list of str
                A list of all channels required by the function. chans can be 
                specified in any of the formats accepted by parseTFMZC() but 
                raw string channel name is recommended since it is explicit. 
                Also, they can be the name you saved a Segmentation under even 
                if this isn't loaded into the tdata - it will be loaded.
            params : list of anything
                Any other parameters that the function requires. 
                
            The function must then be written to accept parameters like this:
                def fun(tdata,tindices,*ims,*params):
                - tdata : is the self TData here
                - tindices : (t,f,m,z,chans,lab) the indices of self from 
                    which we are working. chans is chans above but converted 
                    to integer indices. lab is the label of the current region.
                - *ims : each is a 2D cropped image, one for each of chans
                - *params : params directly from above.
            
            Note that not every function will use all these things but passing 
            everything ensures they can do anything they like.
        returnDF : bool
            Whether to return the DF
        saveDF : bool or str
            If you want to save the dataframe as a csv then this should be the 
            name of the csv. It will be saved in the root of teh XFold.
        verbose : bool
            Whether to print t,f,m,z progress.
        tracking : bool
            See XFold.RegionProps
            
        Returns
        -------
        df : pandas dataframe
            One dataframe is returned where there is a row for every region 
            in the whole tdata. The following are column headings of the 
            dataframe...
            ------------
            XPath : str
                The XPath of the raw data.
            SessionN,T,F,M,Z,C : int
                The index along each axis that this region comes from.
            Session_Name : str
                The name of the parent session.
            FieldID : str
                The FieldID of the field the nucleus comes from.
            centroid-0 : float
                The y-position in pixels of the centre of the object relative 
                to the whole image.
            centroid-1 : float
                The x-position in pixels of the centre of the object relative 
                to the whole image.
        """

        all_props = ('area',
                     'area_bbox',
                     #'area_convex',
                     #'area_filled',
                     'axis_major_length',
                     'axis_minor_length',
                     'bbox',
                     'centroid',
                     'centroid_local',
                     'coords',
                     'eccentricity',
                     #'equivalent_diameter_area',
                     #'euler_number',
                     #'extent',
                     #'feret_diameter_max',
                     'image',
                     'image_convex',
                     'image_filled',
                     #'inertia_tensor',
                     #'inertia_tensor_eigvals',
                     'label',
                     #'moments',
                     #'moments_central',
                     #'moments_hu',
                     #'moments_normalized',
                     'orientation',
                     'perimeter',
                     #'perimeter_crofton',
                     'slice',
                     'solidity')
        
        not_needed = ['coords', # it is the coordinate of each pixel in the mask
                      'image', # this is a boolean array of the mask, cropped to bounding box
                      'image_convex', # the convex version of image
                      'image_filled', #
                      #'slice' # the slice that you would use to take the bounding box
                     ]

        if tracking:
            not_needed += ['area_bbox',
                           #'bbox',
                           #'centroid_local',
                           'eccentricity',]
        
        df = pd.DataFrame()

        combined_chan = []
        for f1,ch1,p1 in fun:
            combined_chan += ch1
        combined_chan += [Segmentation]
        combined_chan = list(set(combined_chan))
        all_seg_chan = []       
        
        for cmbc in combined_chan:
            try:
                _,_,_,_,_ = self.parseTFMZC(C=cmbc)
            except ChannelException:
                all_seg_chan.append(cmbc)
        for seg_chan in all_seg_chan:
            self.LoadSegmentationChannel(seg_chan,'Segmentation_'+seg_chan)

        _,_,_,_,cseg = self.parseTFMZC(C=Segmentation)
            
        dims = [self.NT,self.NF,self.NM,self.NZ]
        ranges = map(range,dims)
        for t,f,m,z in product(*ranges):
            if verbose:
                print('t: ',t,' f: ',f,' m: ',m,' z: ',z)
            
            seg = self.data[t,f,m,z,cseg[0]]

            # we have to be careful here. measure.label() always labels in 
            # order of appearance of regions reading from top-> bottom and 
            # left->right. It even relabels masks that have already been 
            # labelled if they aren't labelled in this order. This is bad 
            # because we often want to pass labelled segmentations here, so 
            # that they correspond with other labelled segmentations, and this 
            # may order according to cells rather than that arbitrary order of 
            # appearance. So don't relabel if it is already labelled!
            if genF.isLabelledMaskQ(seg):
                label_img = seg.copy()
            else:
                label_img = measure.label(seg)
            
            if len(np.unique(label_img))==1:
                continue
            regions = measure.regionprops_table(label_img,properties=all_props)
            dfi = pd.DataFrame(regions)

            for f1,ch1,p1 in fun:
                _,_,_,_,chans = self.parseTFMZC(C=ch1)
                new_rows = []
                for i,row in dfi.iterrows():
                    lab = row['label']
                    ims = ([self.data[t,f,m,z,c,*row['slice']].copy() 
                           for c in chans])
                    new_rows.append(f1(self,(t,f,m,z,chans,lab),*ims,*p1))
                dfi = pd.concat([dfi,pd.DataFrame(new_rows)],axis=1)

            TT = self.SeshT[t]
            FF = self.SeshF[f]
            MM = self.SeshM[m]
            ZZ = self.SeshZ[z]            

            dfi['XPath'] = os.path.abspath(self.ParentSession.ParentXFold.XPath)
            dfi['T'] = TT
            dfi['F'] = FF
            dfi['M'] = MM
            dfi['Z'] = ZZ
            dfi['SessionN'] = self.SessionN
            dfi['Session_Name'] = self.ParentSession.Name        
            dfi['FieldID'] = self.FieldIDs[0]
            dfi['Image ID'] = ('S'+dfi['SessionN'].astype(str)
                                +'T'+dfi['T'].astype(str)
                                +'F'+dfi['F'].astype(str)
                                +'M'+dfi['M'].astype(str)
                                +'Z'+dfi['Z'].astype(str))        
                  
            dfi = dfi.drop(columns=not_needed)
            df = pd.concat([df,dfi])

        # since resolution might change between tdata it's good to save it in 
        # the df, since multiple dfs from different Sessions may be combined. 
        # We use self.um_2_pixels to account for any changes in Unit of 
        # self.pixSizeX
        df['xy resolution (um/pix)'] = 1/self.um_2_pixels(1)

        if tracking and not df.empty:
            TID = 'SN-'+str()
            df['Track ID'] = ('SN-' + df['SessionN'].astype(str) 
                              + '_F'+df['F'].astype(str) 
                              + '_M'+df['M'].astype(str) 
                              + '_Z'+df['Z'].astype(str) 
                              + '_label'+df['label'].astype(str))
            
            df.sort_values(by=['Track ID','T'],inplace=True)
            df['delta y'] = df.groupby('Track ID')['centroid-0'].diff()
            df['delta x'] = df.groupby('Track ID')['centroid-1'].diff()
            df['delta r'] = np.sqrt(df['delta x']**2 + df['delta y']**2)
            df['delta r (um)'] = df['delta r']*df['xy resolution (um/pix)']
            df['delta T'] = df.groupby('Track ID')['T'].diff()
            df['delta T (min)'] = (df.apply(lambda row:row['delta T']
                                *self.ParentXFold.SessionsList[row['SessionN']]
                                .TStep.seconds//60,
                                axis=1))
            df['velocity (um/min)'] = df['delta r (um)']/df['delta T (min)']
            df['centroid-0 Ti'] = (df.groupby('Track ID')['centroid-0']
                                   .transform('first'))
            df['centroid-1 Ti'] = (df.groupby('Track ID')['centroid-1']
                                   .transform('first'))
            df['delta y from yi'] = (df['centroid-0'] - df['centroid-0 Ti'])
            df['delta x from xi'] = (df['centroid-1'] - df['centroid-1 Ti'])
            df['delta r from ri'] = np.sqrt(df['delta y from yi']**2 
                                            + df['delta x from xi']**2)
            df['delta r from ri (um)'] =(df['delta r from ri']
                                         *df['xy resolution (um/pix)'])
            df['track index'] = df.groupby('Track ID').cumcount()        
            
        if not df.empty and not tracking:
            df['minor_major_ratio'] = (df['axis_minor_length']
                                       /df['axis_major_length'])
            df['bbox-width-y'] = df['bbox-2'] - df['bbox-0']
            df['bbox-width-x'] = df['bbox-3'] - df['bbox-1']
            df['roundness'] = (4*math.pi*df['area'])/(df['perimeter'].pow(2))

        if saveDF:
            if not saveDF[-4:]=='.csv':
                saveDF = saveDF+'.csv'
            df.to_csv(os.path.join(self.ParentXFold.XPathP,saveDF))
        if returnDF:
            return df

    
    def Blur(self,sigma,
             method='gaussian_cv',
             T='all',
             F='all',
             M='all',
             Z='all',
             C='all'):
        """
        Blurs the specified parts of the tdata. Leaves non-specified parts 
        unchanged.

        Parameters
        -----------
        sigma : int
            If method=gaussian_cv then this is the STD of the kernel.
        method: {'gaussian_cv'}
            So far only method is this one.
        T,F,M,Z,C : int or list of str
            The normal selection parameters.            
        """
        TT,FF,MM,ZZ,CC = self.parseTFMZC(T,F,M,Z,C)
        for t,f,m,z,c in product(TT,FF,MM,ZZ,CC):
            if method=='gaussian_cv':
                #(0,0) means it calculates kernel size from sigma
                self.data[t,f,m,z,c] = cv.GaussianBlur(self.data[t,f,m,z,c],
                                                       (0,0),
                                                       sigma)
                

    def Threshold(self,
                  threshold=None,
                  blur=False,
                  method='otsu',
                  one_thresh=False,
                  C='all',
                  addSeg2TData=True,
                  returnSegmentations=False,
                  returnThresholds=False):
        """
        This does simple thresholding of all tdata parts in the specified 
        channels. Either does all parts of each channel separately or all 
        together (i.e. one threshold value calculated), see one_thresh. 

        Parameters
        -----------
        threshold : None or int or float or list of int/float
            If None then it calculates threshold, otherwise it uses what is 
            provided here. If a list is provided it takes it as a threshold 
            for each channel in provided C.
        blur : False or int
            If not False then must be an int which is the STD of the gaussian 
            blur applied.
        method: {'otsu'}
            So far only method is this one. 
        one_thresh : bool
            Whether to do all tdata parts of each channel using one globally 
            calculated threshold value (if True) or calculate the threshold 
            separately for each part.
        C : int or list of str
            The channels you segment, all normal selection formats accepted.
        addSeg2TData : bool
            If True then the calculated masks will be added in new channels 
            to the TData, each new channel is called Segmentation_ChanName.            
        returnSegmentations : bool
            Whether to return the calculated segmentation masks.   
        returnThresholds : bool
            Whether to return the calculated thresholds.

        ToDo: All Segmentation stuff (e.g. Plot,RegionProps?) seems to just 
        work for one segmentation called 'Segmentation'.
        """
        _,_,_,_,CC = self.parseTFMZC(C=C)


        if returnSegmentations or addSeg2TData:
            masks = np.zeros((self.NT,self.NF,self.NM,self.NZ,len(C),
                              self.NY,self.NX),
                             dtype='uint16')
        if returnThresholds:
            thresh = np.zeros((self.NT,self.NF,self.NM,self.NZ,len(C)),
                              dtype='uint16')

        if isinstance(threshold,int) or isinstance(threshold,float):
            threshold = [threshold for c in CC]

        for i,c in enumerate(CC):

            if threshold:
                if returnSegmentations or addSeg2TData:
                    if blur:
                        bb = cv.GaussianBlur(self.data[:,:,:,:,c],(0,0),blur)
                        masks[:,:,:,:,i] = (bb > threshold[i]).astype('uint16')
                    else:
                        dd = self.data[:,:,:,:,c]
                        masks[:,:,:,:,i] = (dd > threshold[i]).astype('uint16')
                if returnThresholds:
                    thresh[:,:,:,:,i] = threshold[i]        
            elif one_thresh:
                # we use skimage for global thresh calculation and cv otherwise 
                # because cv is probably faster but seems to have some problems 
                # with calculating one thresh from arbitrary arrays
                if method=='otsu':
                    thr = filters.threshold_otsu(self.data[:,:,:,:,c])
                    if returnSegmentations or addSeg2TData:
                        bb = cv.GaussianBlur(self.data[:,:,:,:,c],(0,0),blur)
                        masks[:,:,:,:,i] = (bb > thr).astype('uint16')
                    if returnThresholds:
                        thresh[:,:,:,:,i] = thr
            else:
                dims = [self.NT,self.NF,self.NM,self.NZ]
                ranges = map(range,dims)                
                for t,f,m,z in product(*ranges):
                    if method=='otsu':
                        thr_param = cv.THRESH_BINARY+cv.THRESH_OTSU
                    if (returnSegmentations or addSeg2TData) and returnThresholds:
                        if blur:
                            bb = cv.GaussianBlur(self.data[t,f,m,z,c],(0,0),blur)
                            thresh[t,f,m,z,i],masks[t,f,m,z,i] = cv.threshold(bb,0,65535,thr_param)
                        else:
                            dd = self.data[t,f,m,z,c]
                            thresh[t,f,m,z,i],masks[t,f,m,z,i] = cv.threshold(dd,0,65535,thr_param)
                    elif (returnSegmentations or addSeg2TData):
                        if blur:
                            bb = cv.GaussianBlur(self.data[t,f,m,z,c],(0,0),blur)
                            _,masks[t,f,m,z,i] = cv.threshold(bb,0,65535,thr_param)         
                        else:
                            dd = self.data[t,f,m,z,c]
                            _,masks[t,f,m,z,i] = cv.threshold(dd,0,65535,thr_param)         
                    elif returnThresholds:
                        if blur:
                            bb = cv.GaussianBlur(self.data[t,f,m,z,c],(0,0),blur)
                            thresh[t,f,m,z,i],_ = cv.threshold(bb,0,65535,thr_param)                        
                        else:
                            dd = self.data[t,f,m,z,c]
                            thresh[t,f,m,z,i],_ = cv.threshold(dd,0,65535,thr_param)                        

            if addSeg2TData:
                self.AddArray(masks[:,:,:,:,[i]],'Segmentation_'+self.Chan2[c])  
                
        if returnSegmentations and returnThresholds:
            return [thresh,masks]
        elif returnSegmentations:
            return masks
        elif returnThresholds:
            return thresh


    def CleanMasks(self,
                   C='Segmentation',
                   areaThreshold=None,
                   minAreaUnit='um^2',
                   circThreshold=None,
                   clearBorder=False,    
                   editMasksInPlace=True,
                   addAsNewChannel=False,
                   saveSegmentationsAndDelete=None,                     
                   printWarnings=True,
                   compress=True):
        """
        This removes border objects, objects smaller than a threshold area and 
        /or objects with a circularity smaller than a threshold. 

        Note: it assumes this is a labelled mask. Should add option to label 
        if needed.

        Parameters
        ----------
        C : {'all',int,list,str}
            The channel of a segmentation that you want to clean up. Accepts 
            in all normal formats (see parseTFMZC).
        areaThreshold : None or int or float
            If you want to remove objects smaller than a threshold then enter 
            it here. It is assumed to be in um^2.
        minAreaUnit : {'pixels','um^2'}
            The unit that minArea is taken to be in.            
        circThreshold : None or int or float
            If you want to remove objects with circularity smaller than a 
            threshold then enter it here.
        clearBorder : bool
            Whether to remove objects touching the image border.
        editMasksInPlace : bool
            If True then it edits the channels C directly.
        addAsNewChannel : bool
            If True then it makes a new mask and adds as a new channel with 
            the same name as before but with 
            '_borderCleared_areaThresholdXX_circThresholdXX' added according 
            to what you chose.
        saveSegmentationsAndDelete : None or str
            Whether to save the cleaned segmentation masks. Str will be the 
            name of the folder they are saved into. This deletes everything 
            except for the segmentation just before saving because SaveData 
            doesn't yet have one channel only option.               
        printWarnings : bool
            Just lets you turn off warnings if they get annoying.    
        compress : bool
            Whether the saved files should be compressed or not.            
        """
        
        _,_,_,_,C = self.parseTFMZC(C=C)

        if areaThreshold:
            if minAreaUnit=='um^2':
                areaThreshold = self.um_2_pixels(areaThreshold)
                areaThreshold = self.um_2_pixels(areaThreshold)

        nt,nf,nm,nz,_,ny,nx = self.Shape

        new_masks = self.data[:,:,:,:,C].copy()
        
        if clearBorder:
            for c in C:
                NQ1 = nt*nf*nm*nz
                cb1 = clear_border(new_masks[:,:,:,:,c].reshape((NQ1,ny,nx)))
                new_masks[:,:,:,:,c] = cb1.reshape((nt,nf,nm,nz,ny,nx))

        ranges = map(range,(nt,nf,nm,nz))
        for t,f,m,z in product(*ranges):
            for c in C:
                if circThreshold or areaThreshold:  
                    props = measure.regionprops(self.data[t,f,m,z,c])

                if circThreshold:
                    for region in props:
                        circ = (region.perimeter**2)/(4*np.pi*region.area)
                        if circ < circThreshold:
                            L1 = region.label
                            new_masks[t,f,m,z,c][new_masks[t,f,m,z,c]==L1] = 0
                        
                if areaThreshold:
                    for region in props:
                        if region.area < areaThreshold:
                            L1 = region.label
                            new_masks[t,f,m,z,c][new_masks[t,f,m,z,c]==L1] = 0

        if addAsNewChannel or saveSegmentationsAndDelete:
            new_names = []
            for c in C:
                new_name = self.Chan2[c]
                if clearBorder:
                    new_name += '_borderCleared'
                if areaThreshold:
                    areaT_str = str(_areaThreshold).replace('.','p')
                    new_name += '_areaThreshold'+areaT_str
                if circThreshold:
                    circT_str = str(_circThreshold).replace('.','p')
                    new_name += '_circThreshold'+circT_str
                new_names.append(new_name)
                self.AddArray(new_masks[:,:,:,:,[c]],new_name)  
        if saveSegmentationsAndDelete:
            self.TakeSubStack(C=new_names,updateNewSeshNQ=True)
            self.SaveData(saveSegmentationsAndDelete,compress=compress)        
        if editMasksInPlace:
            self.data[:,:,:,:,C] = new_masks 

            
    def TrackSegmentations(self,
                           out_name,
                           Segmentation=False,
                           returnMasks=False,
                           saveSegmentationsAndDelete=True,
                           editMasksInPlace=False,
                           addAsNewChannel=False, 
                           returnTracks=False,
                           T='all',
                           F='all',
                           M='all',
                           Z='all',
                           assumeConstantDims='auto',
                           FieldIDMapList='auto',
                           SaveXFoldPickle=False,
                           LoadFromPickle=False,
                           compress=False):
        """
        This does tracking of objects in a Segmentation associated with this 
        TData through the time dimension. 

        If there is a channel with Segmentation in the name then it tracks 
        this. Otherwise provide a name of the Segmentation XFold and it will 
        check if this is already loaded to the ParentXFold and load it if not. 

        It can add the new masks to the TData as a new channel, directly 
        change the masks of the original Segmentation channel, or save them in 
        multisesh format.

        Objects in the original masks that don't appear in the tracks are not 
        put in the new masks.

        It can also return a pandas dataframe of the tracking data.
                
        Parameters
        ----------        
        out_name : str
            The name of the directory where the tracks (h5 files) and tracked 
            masks (if saveMask is True) will be stored. It is assumed to be in 
            the parent directory of the XFold.    
        Segmentation : False or str
            If False then there must be a channel called 'Segmentation' in the 
            TData. Otherwise you give it a string which isthe absolute path to 
            the multisesh saved segmentation dataset corresponding to the 
            ParentXFold. It will check if this XFold is already loaded to 
            self.ParentXFold.SegmentationXFolds and load it with 
            self.get_segmenbtation_xfold() if not. If you give it a string with
            no directory separators the it joins it to XPathP.
        saveSegmentationsAndDelete : bool 
            Whether to save the tracked segmentations. If so they will be 
            saved in the folder specified by out_name. Otherwise it just saves 
            the track files.  
        editMasksInPlace : bool
            If True then it edits the channel that has been tracked directly.
        addAsNewChannel : bool
            If True then it makes a new mask and adds as a new channel with 
            the same name as before but with 
            '_Tracked' added.      
        returnTracks : bool
            Whether to return the tracks data. It is a list of of lists of
            btrack.btypes.Tracklet objects which contain all tracking data... 
            table is shown when executed. I.e. the top level list is one 
            element for each f,m,z combination. And the next level is one 
            track object for each track in that f,m,z time series.
        T,F,M,Z : 'all' or int or list of int (str for F,C too)
            Which of the TData's times, fields, tiles and slices to load. All 
            normal formats accepted (see parseTFMZC()).  
        compress : bool
            Whether the saved files should be compressed or not.
        """
        
        fp = os.path.join(self.ParentXFold.XPathP,out_name)        

        FEATURES = [
            "area", 
            "major_axis_length", 
            "minor_axis_length", 
            "orientation", 
            "solidity"]                             
                  
        TT,FF,MM,ZZ,_ = self.parseTFMZC(T=T,F=F,M=M,Z=Z)  

        if Segmentation and 'Segmentation' in self.Chan:
            print(EM.tr1)
        if Segmentation:
            self.LoadSegmentationChannel(Segmentation,
                                         assumeConstantDims=assumeConstantDims,
                                         FieldIDMapList=FieldIDMapList,
                                         SaveXFoldPickle=SaveXFoldPickle,
                                         LoadFromPickle=LoadFromPickle,)
        else:
            assert 'Segmentation' in self.Chan,EM.ts1

        if returnTracks:
            all_tracks = []

        for f,m,z in product(FF,MM,ZZ):

            masks = self.data[TT,f,m,z,self.chan_index('Segmentation')]

            # create btrack objects (with properties) from the segmentation data
            # (you can also calculate properties, 
            # based on scikit-image regionprops)
            objects = btrack.utils.segmentation_to_objects(masks,
                                                    properties=tuple(FEATURES))

            with btrack.BayesianTracker() as tracker:
            
                # configure the tracker using a config file
                dir_path = os.path.dirname(os.path.realpath(__file__))            
                tracker.configure(os.path.join(dir_path,'btrack_config.json'))
                tracker.max_search_radius = 400
                tracker.tracking_updates = ["MOTION","VISUAL"]
                
                # append the objects to be tracked
                tracker.append(objects)           
                
                # track them (in interactive mode)
                # step size is just how often you print summary stats
                #tracker.track_interactive(step_size=100)
                tracker.track(step_size=100)
                
                # generate hypotheses and run the global optimizer
                tracker.optimise()
                
                # store the data in an HDF5 file
                if not os.path.isdir(fp):
                    os.mkdir(fp)
                fp2 = os.path.join(fp,'tracks_S'
                                       +str(self.ParentSession.SessionN)
                                       +'_F'+str(f)
                                       +'_M'+str(m)
                                       +'_Z'+str(z)+'.h5')
                tracker.export(fp2, obj_type='obj_type_1')
                # get the tracks as a python list
                
                tracks = tracker.tracks
            
            makeMasksQ = any((saveSegmentationsAndDelete,
                             editMasksInPlace,
                             addAsNewChannel))
            if makeMasksQ:
                trk_masks = np.zeros_like(masks)
                for track in tracks:
                    if len(track)==1:
                        continue
                    #print('track')
                    if not str(track['fate'])=='Fates.FALSE_POSITIVE':
                        #print('not false positive')
                        for i in range(len(track)):
                            #print('track_i')
                            if not track['dummy'][i]:
                                #print('not dummy')
                                x = int(track['x'][i])
                                y = int(track['y'][i])
                                t = track['t'][i]
                                old_label = masks[t,y,x]
                                new_lab = track['ID']
                                #print('time ',t,' same ',old_label==new_lab,' old ',old_label,' new ',new_lab)
                                trk_masks[t][masks[t]==old_label] = new_lab
            if saveSegmentationsAndDelete or addAsNewChannel:
                chan2_name = self.Chan2[self.chan_index('Segmentation')] 
                chan2_name += '_Tracked'
                self.AddArray(trk_masks[:,np.newaxis,
                                              np.newaxis,
                                              np.newaxis,
                                              np.newaxis,:,:],chan2_name)   
            if saveSegmentationsAndDelete:
                self.TakeSubStack(C=chan2_name,updateNewSeshNQ=True)
                self.SaveData(out_name,compress=compress)
            if editMasksInPlace:
                self.data[TT,f,m,z,self.chan_index('Segmentation')] = trk_masks
            if returnTracks:
                all_tracks.append(tracks)

        if returnTracks:
            return all_tracks
    
    
    def get_segmentation_xfold(self,
                               Segmentation,
                               assumeConstantDims='auto',
                               FieldIDMapList='auto',
                               SaveXFoldPickle=False,
                               LoadFromPickle=False,):
        """
        see XFold.get_segmentation_xfold()
        """
        return self.ParentXFold.get_segmentation_xfold(Segmentation,
                                        assumeConstantDims=assumeConstantDims,
                                        FieldIDMapList=FieldIDMapList,
                                        SaveXFoldPickle=SaveXFoldPickle,
                                        LoadFromPickle=LoadFromPickle)


    def LoadSegmentationChannel(self,
                                Segmentation,
                                Chan='auto',
                                assumeConstantDims='auto',
                                FieldIDMapList='auto',
                                SaveXFoldPickle=False,
                                LoadFromPickle=False,):
        """
        This loads a segmentation to your TData. It must come from an XFold of 
        Segmentations that corresponds exactly to the Parent XFold of this 
        TData. It will check if the segmentation you specify is already loaded 
        into self.ParentXFold.SegmentationXFolds and load it if not.

        Parameters
        -----------
        Segmentation : str
            Give it a string which is the absolute path to the segmentation 
            dataset  corresponding to self.ParentXFold. It will check if this 
            XFold is  already loaded to self.ParentXFold.SegmentationXFolds 
            and load it with  self.LoadSegmentationXFold() if not. If you give 
            it a  string with no directory separators the it joins it to 
            XPathP.
        Chan : str
            What to name the segmentation channel that you add. If you put 
            'auto' it names it 'Segmentation_' plus what you provided for the 
            parameter Segmentation - that way it is tracked where that channel 
            came from. Remember that any channels containing the string 
            'Segmentation' are are converted in the regularised self.Chan to 
            just 'Segmentation' and being called 'Segmentation' is required 
            for some functions rely on it being called this.
        assumeConstantDims : bool or 'auto'       
        FieldIDMapList : None or str or list of str or list of list of str 
                         or 'get_FieldIDMapList_from_tags'   
        SaveXFoldPickle : bool or str
        LoadFromPickle : bool or str
            All of these get passed to XFold __init__ when the new XFold is 
            made. If you choose 'auto' for assumeConstantDims or 
            FieldIDMapList then it sends the value from self to the new 
            __init__. 
        """ 
        xf = self.get_segmentation_xfold(Segmentation,
                                         assumeConstantDims=assumeConstantDims,
                                         FieldIDMapList=FieldIDMapList,
                                         SaveXFoldPickle=SaveXFoldPickle,
                                         LoadFromPickle=LoadFromPickle)

        iss = self.ParentSession.SessionN

        t = self.SeshT
        f = self.SeshF
        m = self.SeshM
        z = self.SeshZ

        # C='Segmentation' because all multisesh segmentation functions save 
        # the segmentations from TDatas with one channel called 'Segmentation' 
        # (i.e. for both self.Chan and self.Chan2)
        td_seg = xf.makeTData(S=iss,T=t,F=f,M=m,Z=z,C='Segmentation')

        if Chan=='auto':
            Chan = 'Segmentation_'+Segmentation
        
        self.AddArray(td_seg.data,Chan)


    def um_2_pixels(self,length,verbose=True):
        """
        Converts the provided length from um to pixels according to the pixel 
        size (in X) of the TData. Note how it takes this from self rather than 
        self.ParentSession since rescaling might have changed the value in 
        self.

        Parameters
        ----------
        length : int or float
            The length to be converted.

        verbose : bool
            Whether to print warning that pixel unit is unknown.
        """
        
        if self.pixUnit=='m':
            length2 = length/(self.pixSizeX*1000000)
        elif self.pixUnit=='micron':
            length2 = length/(self.pixSizeX)
        elif self.pixUnit=='um':
            length2 = length/(self.pixSizeX)   
        elif self.pixUnit is None:
            if verbose:
                print('Warning: pixUnit=None so length not converted')
            length2 = length
        else:
            raise Exception(EM.um1 + str(self.pixUnit))
        
        return length2

        
class BlankTFile():
    """
    Just a place holder TFile to put in the ParentTFiles list if the TFiles 
    weren't found and you allow that in Session.makeTData using 
    allowMissing.
    """
    def __init__(
        self,
        ParentSession,
        TFileN=None,
        TPath = None,
        SeshT = None,
        SeshF = None,
        SeshM = None,
        SeshZ = None,
        SeshC = None,
        NT = None,
        NF = None,
        NM = None,
        NZ = None,
        NC = None,
        NY = None,
        NX = None
    ):        

        self.ParentSession = ParentSession
        self.TFileN = TFileN
        self.TPath = TPath
        self.Chan = [c for i,c in enumerate(ParentSession.Chan) if i in SeshC]     
        # lists of indices which locate the TFile data within parent session
        self.SeshT = SeshT
        self.SeshF = SeshF
        self.SeshM = SeshM
        self.SeshZ = SeshZ
        self.SeshC = SeshC
        self.FieldIDs = []        
        
        
def AutoLevel(data,minP=2,maxP=98,overWrite=False,returnTDatas=True):
    """
    For each field, this looks at all timepoints and applies autolevelling.
    That is, all pixels below/above the minP/maxP percentile pixel value are
    set to 0/max value (where max is the max value for the data type) and all
    pixel values in between scale linearly.

    You give it a TFile or a TData or a list of either.

    With lists of TFiles or TDatas, it of course autolevels globally across
    the list.

    It can overwrite the input files or just output TDatas (which will match
    the structure of the input).
    """
    # first regularise input into a list of TDatas
    if not isinstance(data,list):
        data = [data]
    if all([isinstance(d,TFile) for d in data]):
        data = [tf.makeTData() for tf in data]
    elif all([isinstance(d,TData) for d in data]):
        pass # because this is wat we want
    else:
        # anything else is an error
        raise Exception(EM.al1)

    # now do the levelling
    for td in data:
        for f in range(td.NF):
            for c in range(td.NC):
                minV = np.percentile(np.ravel(td.data[:,f,:,:,c]),minP)
                maxV = np.percentile(np.ravel(td.data[:,f,:,:,c]),maxP)
                td.data[:,f,:,:,c][td.data[:,f,:,:,c]<minV] = minV
                td.data[:,f,:,:,c][td.data[:,f,:,:,c]>maxV] = maxV
                td.data[:,f,:,:,c] = td.data[:,f,:,:,c] - minV
                td.data[:,f,:,:,c] = td.data[:,f,:,:,c]*XFold.UINT16MAX/maxV
                td.data = td.data.astype('uint16')

    if overWrite:
        pass

    if returnTDatas:
        return outTDatas


def PlotDataFrame(df,col_names,seg_col_name=False,measure_col_names=False):
    """
    Give this a dataframe containing paths to images and measurements from the 
    images and it will load the images as you cycle through and display the 
    measurements that you select. If your dataframe has column headings 
    T,F,M,Z,C then it gives you controls to select these and only selects rows 
    which correspond to this selection.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe must contain columns corresponding to col_names, seg_col_name 
        and measure_col_names. Currently it must also have columns called 'T',
        'F','M' and 'Z' though maybe we should make that work optionally. 
    col_names : list of str
        A list of the dataframe column names which contain paths to the images 
        you want to load. The list may contain up to 4 elements, one for each 
        channel.
    seg_col_name : False or str
        The column name for the segmentation channel if it exists.
    measure_col_names : False or list of str
        The names of columns for which you want to display the result.
    """
    width = 500
    height = 500
    font_scale = 0.6
    font = cv.FONT_HERSHEY_SIMPLEX
    text = 'no row in dataframe with these parameters' 
    font_thickness = 1
    blank = np.zeros((width, height), dtype=np.uint8)
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    blank = cv.putText(blank, 
                       text, 
                       (x, y), 
                       font, 
                       font_scale, 
                       (255, 255, 255), 
                       font_thickness, 
                       cv.LINE_AA)
    
    assert len(col_names)<5,'maximum 4 channels in col_names'
    
    Svals = list(np.unique(df['SessionN']))
    Tvals = list(np.unique(df['T']))
    Fvals = list(np.unique(df['F']))
    Mvals = list(np.unique(df['M']))
    Zvals = list(np.unique(df['Z']))

    S0 = df.iloc[0]['SessionN']
    T0 = df.iloc[0]['T']
    F0 = df.iloc[0]['F']
    M0 = df.iloc[0]['M']
    Z0 = df.iloc[0]['Z']
    
    dfb = df[(df['SessionN']==S0)&(df['T']==T0)&(df['F']==F0)&(df['M']==M0)&(df['Z']==Z0)]
    
    ims = [tifffile.imread(dfb.iloc[0][pn]) for pn in col_names]
    if seg_col_name:
        assert isinstance(seg_col_name,str),'seg_col_name must be a string'
        segIm = tifffile.imread(dfb.iloc[0][seg_col_name])

    Ss = widgets.SelectionSlider(value=S0,description='SessionN',options=Svals)
    Ts = widgets.SelectionSlider(value=T0,description='T',options=Tvals)
    Fs = widgets.SelectionSlider(value=F0,description='F',options=Fvals)
    Ms = widgets.SelectionSlider(value=M0,description='M',options=Mvals)
    Zs = widgets.SelectionSlider(value=Z0,description='Z',options=Zvals)
    
    vminC1s = widgets.IntSlider(0,0,255,10,description='vminC1')
    vmaxC1s = widgets.IntSlider(255,0,255,10,description='vmaxC1')
    vminC2s = widgets.IntSlider(0,0,255,10,description='vminC2')
    vmaxC2s = widgets.IntSlider(255,0,255,10,description='vmaxC2')
    vminC3s = widgets.IntSlider(0,0,255,10,description='vminC3')
    vmaxC3s = widgets.IntSlider(255,0,255,10,description='vmaxC3')
    vminC4s = widgets.IntSlider(0,0,255,10,description='vminC4')
    vmaxC4s = widgets.IntSlider(255,0,255,10,description='vmaxC4')
    
    Ch1c = widgets.Checkbox(value=True,description='Ch1')
    Ch2c = widgets.Checkbox(value=False,description='Ch2')
    Ch3c = widgets.Checkbox(value=False,description='Ch3')
    Ch4c = widgets.Checkbox(value=False,description='Ch4')
    ChSegMc = widgets.Checkbox(value=False,description='ChSegMc')

    auto = widgets.Checkbox(value=False,description='autoscale')
    
    if len(Svals)<2:
        Ss.disabled = True
    if len(Tvals)<2:
        Ts.disabled = True
    if len(Fvals)<2:
        Fs.disabled = True
    if len(Mvals)<2:
        Ms.disabled = True
    if len(Zvals)<2:
        Zs.disabled = True
    
    if len(col_names)<4:
        Ch4c.disabled = True
        vminC4s.disabled = True
        vmaxC4s.disabled = True
    if len(col_names)<3:
        Ch3c.disabled = True
        vminC3s.disabled = True
        vmaxC3s.disabled = True   
    if len(col_names)<2:
        Ch2c.disabled = True
        vminC2s.disabled = True
        vmaxC2s.disabled = True    
    if not seg_col_name:
        ChSegMc.disabled = True
        
    #rows = widgets.IntSlider(value=0,min=0,max=len(df1b)-1,step=1,description='row')
    rows_wij = widgets.BoundedIntText(value=0,
                                      min=0,
                                      max=len(dfb)-1,
                                      step=1,
                                      description='row')
    
    sliders_box1 = widgets.VBox([Ss,Ts,Fs,Ms,Zs])
    sliders_box2 = widgets.VBox([vminC1s,
                                 vmaxC1s,
                                 vminC2s,
                                 vmaxC2s,
                                 vminC3s,
                                 vmaxC3s,
                                 vminC4s,
                                 vmaxC4s])   
    
    checkboxes_box = widgets.VBox([Ch1c,Ch2c,Ch3c,Ch4c,ChSegMc,auto])
    
    widgets_box = widgets.HBox([sliders_box1,
                                sliders_box2,
                                checkboxes_box,
                                rows_wij])

    def update(S,T,F,M,Z,
               vminC1,vmaxC1,
               vminC2,vmaxC2,
               vminC3,vmaxC3,
               vminC4,vmaxC4,           
               row,
               Ch1,Ch2,Ch3,Ch4,ChSegM,autoscale):
        
        dfb = df[(df['SessionN']==S)&(df['T']==T)&(df['F']==F)&(df['M']==M)&(df['Z']==Z)]
        if len(dfb)==0:
            ims = [blank]*len(col_names)
            rows_wij.min = 0
            rows_wij.max = 0
        else:
            ims = [tifffile.imread(dfb.iloc[row][pn]) for pn in col_names]
            if seg_col_name:
                segIm = tifffile.imread(dfb.iloc[row][seg_col_name])
            rows_wij.max = len(dfb)-1
    
        overlay_image = np.zeros((ims[0].shape[0],
                                  ims[0].shape[1],
                                  3),dtype='uint16')
        print('ims[0].shape: ',ims[0].shape)
        if Ch1:
            if not autoscale:
                vminC1b=(vminC1/255)*np.iinfo(ims[0].dtype).max
                vmaxC1b=(vmaxC1/255)*np.iinfo(ims[0].dtype).max
            else:
                vminC1b=(vminC1/255)*np.percentile(ims[0],99.5)
                vmaxC1b=(vmaxC1/255)*np.percentile(ims[0],99.5)          
            image1 = exposure.rescale_intensity(ims[0],
                                                in_range=(vminC1b,vmaxC1b),
                                                out_range=(0,255))
            print('image1.shape: ',image1.shape)
            print('overlay_image.shape: ',overlay_image.shape)
            overlay_image[:,:,0] = image1
        if Ch2:
            if not autoscale:
                vminC2b=(vminC2/255)*np.iinfo(ims[1].dtype).max
                vmaxC2b=(vmaxC2/255)*np.iinfo(ims[1].dtype).max
            else:
                vminC2b=(vminC2/255)*np.percentile(ims[1],99.5)
                vmaxC2b=(vmaxC2/255)*np.percentile(ims[1],99.5)                
            image2 = exposure.rescale_intensity(ims[1],
                                                in_range=(vminC2b,vmaxC2b),
                                                out_range=(0,255))
            overlay_image[:,:,1] = image2
        if Ch3:
            if not autoscale:
                vminC3b=(vminC3/255)*np.iinfo(ims[2].dtype).max
                vmaxC3b=(vmaxC3/255)*np.iinfo(ims[2].dtype).max
            else:
                vminC3b=(vminC3/255)*np.percentile(ims[2],99.5)
                vmaxC3b=(vmaxC3/255)*np.percentile(ims[2],99.5)  
            image3 = exposure.rescale_intensity(ims[2],
                                                in_range=(vminC3b,vmaxC3b),
                                                out_range=(0,255))
            overlay_image[:,:,2] = image3
        if Ch4:
            if not autoscale:
                vminC4b=(vminC4/255)*np.iinfo(ims[3].dtype).max
                vmaxC4b=(vmaxC4/255)*np.iinfo(ims[3].dtype).max
            else:
                vminC4b=(vminC4/255)*np.percentile(ims[3],99.5)
                vmaxC4b=(vmaxC4/255)*np.percentile(ims[3],99.5) 
                
            image4 = exposure.rescale_intensity(ims[3],
                                                in_range=(vminC4b,vmaxC4b),
                                                out_range=(0,255))
            image4 = np.repeat(image4[:,:,np.newaxis],
                               3,
                               axis=2).astype('uint16')
            overlay_image += image4
        if ChSegM:
            segIm[segIm!=0] = 255
            segIm = np.repeat(segIm[:,:,np.newaxis], 3, axis=2)
            segIm[:, :, 2] = 0 # make it yellow
            segIm = segIm.astype('uint16')
            overlay_image += segIm 
    
        overlay_image[overlay_image>255] = 255 
    
        if measure_col_names:
            for m in measure_col_names:
                print(m+': '+str(dfb.iloc[row][m]))
                   
        fig,ax = plt.subplots(figsize=(10,10))
        ax.imshow(overlay_image)

    display(widgets_box)
    display(widgets.interactive_output(update, 
                                        {'S':Ss,
                                         'T':Ts,
                                         'F':Fs,
                                         'M':Ms,
                                         'Z':Zs,
                                         'vminC1':vminC1s,'vmaxC1':vmaxC1s,
                                         'vminC2':vminC2s,'vmaxC2':vmaxC2s,
                                         'vminC3':vminC3s,'vmaxC3':vmaxC3s,
                                         'vminC4':vminC4s,'vmaxC4':vmaxC4s,            
                                         'row':rows_wij,
                                         'Ch1':Ch1c,
                                         'Ch2':Ch2c,
                                         'Ch3':Ch3c,
                                         'Ch4':Ch4c,
                                        'ChSegM':ChSegMc,
                                        'autoscale':auto}))