import re
import os
import math
import numpy as np
import pandas as pd
import tifffile
from datetime import datetime
from datetime import timedelta
from collections import Counter
import json
from skimage.measure import regionprops,label
from skimage.morphology import dilation,erosion,disk
from skimage import filters,exposure,transform
from scipy.ndimage import binary_fill_holes
from scipy import stats

import cv2 as cv

# this makes the import work when this file is being imported as part 
# of a package or by itself
if __package__ == '':
    from exceptions import UnknownTifMeta
    import findMeta as fMeta
    import errorMessages as EM
    import definitions as defs
else:
    from .exceptions import UnknownTifMeta
    from . import findMeta as fMeta
    from . import errorMessages as EM
    from . import definitions as defs


def stripTags(filename,madeBy='Andor',customTags=None):
    """ 
    This removes from a filename the file extension and the tags (which 
    are software specific) which are added when a session is divided up into 
    different files.
    
    madeBy dependent behaviour
    -------------------------
    Andor : removes tags like _qnnnn from the end. Removes any number from the 
            end but not ever from the middle of the filename.
    MicroManager : removes .ome from end, then _MMStack_n, then Pos_nnn_nnn. 
            Haven't tested this on many variants.
    Opera : Done.
    customTags : dict
        These are extra user-supplied tags to remove that aren't automatic from
        the software that made the images. All values in dict should be 
        compiled regexs and are removed before anything else.
    """ 
    
    if customTags:
        assert madeBy=='aicsimageio_compatible',EM.md4
    
    # first get rid of extension
    #filename = filename.split('.')[0] # this breaks if there are other . in filename
    filename = filename[:filename.rfind('.')]
    
    if customTags:
        for k,v in customTags.items():
            filename = v.sub('',filename)
            
    if madeBy=='Andor':
        tagRegex = re.compile(r'(_(t|f|m|z|w|s)\d{3,9})+$')
        return tagRegex.sub('',filename)
        
    elif madeBy=='multisesh' or madeBy=='multisesh_compressed':
        # note how this strips the time and chan tags added by tdata.saveTData 
        # but not the sessions one... because stripTags is usually used to 
        # group sessions
        timeRegex = re.compile(r'_t\d{4}')
        filename = timeRegex.sub('',filename)
        chanRegex = re.compile(r'_C(_Ch[^\W_]+){1,10}')
        filename = chanRegex.sub('',filename)
        monRegex = re.compile(r'_m\d{4}')
        filename = monRegex.sub('',filename)        
        # for multisesh the parent directory must also be stripped for session file gathering!
        if os.path.split(filename)[0]: 
            path1,name1 = os.path.split(filename)
            path2,expDir = os.path.split(path1)
            filename = os.path.join(path2,name1)            
        return filename
        
    elif madeBy=='MicroManager':
        return filename.split('_MMStack_')[0]
        # old way:
        #regPos = re.compile(r'(-?Pos_\d\d\d_\d\d\d$)')
        #regOME = re.compile(r'(.ome$)')
        #regF = re.compile(r'(_MMStack_\d-?$)')
        #return regF.sub('',regPos.sub('',regOME.sub('',filename)))
        
    elif madeBy=='Incucyte':
        tagRegex = re.compile(r"_[A-Z]\d+_\d+_\d{4}y\d{2}m\d{2}d_\d{2}h\d{2}m$")
        tagRegex2 = re.compile(r"_[A-Z]\d+_\d+_\d{2}d\d{2}h\d{2}m$")
        if re.search(tagRegex,filename):
            return tagRegex.sub('',filename)
        elif re.search(tagRegex2,filename):
            return tagRegex2.sub('',filename)
        else:
            raise Exception('Unknown Incucyte filename format in stripTags!')
            
    elif madeBy=='Opera':
        # this will probably need updating for multi-z,t images
        tagRegex = re.compile(r'r\d+c\d+f\d+(p\d+)?-ch\d+')
        return tagRegex.sub('',filename)
        
    # for ones which are aicsimageio compatible but nothing else, there 
    # are no tags
    elif madeBy=='aicsimageio_compatible':
        return filename
    elif madeBy=='Leica':
        return filename
    elif madeBy=='Zeiss': 
        # dont know if there are tags in czi, havent seen any yet
        return filename
    elif madeBy=='Nikon': 
        # dont think there are tags in Nikon
        return filename        
    else:
        raise Exception(f'Unknown software {madeBy}')
                      

def regexFromfilePath(reg,fPath,findAll=False,g=1,chars=10000,isFloat=False):
    """ this return regexes from fPath
        if the thing found is a digit char it always converts it to an int
        you can access other groups with g but default is g=1
        only loads no. 'chars' of characters from file in case too big
        for findAll = False you can use a compiled regex or a raw string
    """
    # loads data from fPath
    with open(fPath, 'rt') as theFile:  
        fileData = theFile.read(chars)            
    if findAll:
        N = re.findall(reg,fileData)
    else:
        if type(reg)==str:
            reg = re.compile(reg)
        N = reg.search(fileData)
        if N and type(N)!=list:
            N = N.group(g)
            if N.isdigit():
                N = int(N)
            if isFloat:
                N = float(N)
    return N

    
def chanDic(channel,guessMissing=True):
    """ 
    This converts the channel's actual protocol 
    name to it's 'general name'.
    """
    
    channelDic = {'BF':'BF',
            'Brightfield':'BF',                  
            'DIC':'BF',
            'GFP':'GFP',
            'EGFP':'GFP',                  
            'YFP':'YFP',
            'RFP':'RFP',
            'CFP':'CFP',
            'DAPI':'DAPI',
            'FR':'FR',
            'Unknown_Gray':'Unknown_Gray',
            'Unknown_Green':'Unknown_Green',
            'Unknown_Blue':'Unknown_Blue',
            'Unknown_Red':'Unknown_Red',
            'T PMT-T1':'BF',                  
            'Channel:0:0':'DAPI', 
            'Channel:0:1':'GFP',
            'Channel:0:2':'RFP', 
            'Channel:0:3':'FR',
            'Channel:0:4':'BF',
            'trans':'BF',                  
            'Tom_BF':'BF',
            'Tom_YFP':'YFP',
            'Tom_GFP':'GFP',                  
            'Tom_CFP':'CFP',
            'Tom_RFP':'RFP',
            'Tom_FR':'FR',
            'Far_Red':'FR',
            'FarRed':'FR',
            'Far Red':'FR',
            'Cy5':'FR',
            'Alexa 647':'FR',
            'Tom_DAPI':'DAPI',
            'RFP_Wide':'RFP',
            'rhodamine':'RFP',
            'Label':'Label',
            'Segmentation':'Segmentation',
            'Blue':'DAPI',
            'Cyan':'Cyan',
            'Green':'GFP',
            'Yellow':'FR',
            'White':'BF',
            'Red':'RFP',
            'Magenta':'Purple',                  
            'Purple':'Purple',
            'EGFP-T2':'GFP',
            'DAPI-T3':'DAPI',
            'DAPI - extended':'DAPI',                  
            'AF647-T3':'FR',  
            'Alexa 488':'GFP', 
            'Alexa 546':'RFP',                   
            'Alexa 568':'RFP', 
            'Alexa Fluor 488':'GFP', 
            'Alexa Fluor 555':'RFP',    
            'Alexa Fluor 568':'RFP',         
            'Alexa Fluor 633':'FR',                  
            'Alexa Fluor 647':'FR', 
            'CellMask Green':'GFP',
            '353':'DAPI',                  
            '405':'DAPI',
            '488':'GFP',
            '555':'RFP',
            '561':'RFP',
            '642':'FR',
            '653':'FR',
            '647':'FR',
            '920':'FR',
            'mCherry':'RFP',
            'DivNeg':'DAPI',
            'DivPos':'RFP',
            'Orientations':'Orientations',
            'Coherences':'Coherences',
            'Energies':'Energies',
            None:None
                 }
    
    if 'DivNeg' in channel:
        channel = 'DivNeg'
    if 'DivPos' in channel:
        channel = 'DivPos'   
    if 'Segmentation' in channel:
        channel = 'Segmentation'  
    if 'Orientations' in channel:
        channel = 'Orientations'
    if 'Coherences' in channel:
        channel = 'Coherences'
    if 'Energies' in channel:
        channel = 'Energies'        
    
    if guessMissing:
        if not channel in channelDic.keys():
            print('Warning: unknown channel, add '+channel+' to generalFuctions.chanDic')
            channel = 'GFP'
        return channelDic[channel]
    else:
        assert channel in channelDic.keys(),EM.ch1 + 'Channel: '+channel
        return channelDic[channel]


def chan2colour(chan):
    """
    This converts from a regularised channel name to a standard colour string.
    """
    colourDict = {
        'BF':'Greys',
        'DAPI':'Blue',
        'GFP':'Green',
        'RFP':'Red',
        'FR':'Yellow',
        'YFP':'Yellow',        
        'Purple':'Magenta',
        'Cyan':'Cyan',
        'Orientations':'Hue (HSV)',
        'Coherences':'Saturation (HSV)',
        'Energies':'Value (HSV)'
    }
    return colourDict[chan]


def LUTDic(channel):
    """
    This takes a channel's 'general name' (see channelDic) and assigns 
    an LUT mix rule (see LUTMixer()).
    You might want to change this around or add new channels etc.
    """
    
    theLUTDic = {'BF':[True,True,True],
                 'DAPI':[False,False,True],
                 'GFP':[False,True,False],
                 'YFP':[False,True,False],
                 'CFP':[False,False,True],
                 'RFP':[True,False,False],
                 'FR':[True,True,False], # = yellow!
                 'Purple':[True,False,True],
                 'Label':[True,True,True],
                 'Segmentation':[True,False,True], # = purple because not used often!                 
                 'Cyan':[False,True,True],
                 'Unknown_Grey':[True,True,True],
                 'Unknown_Green':[False,True,False],
                 'Unknown_Blue':[False,False,True],
                 'Unknown_Red':[True,False,False],
                 'Chan_unknown':[True,True,True]
                }
    
    assert channel in theLUTDic.keys(),EM.ch2
    
    return theLUTDic[channel]



def LUTMixer(mixVector):
    """
    This makes an LUT in image j format from a boolean vector.
    I.e. you give it a vector like [True, False, True] to say which 
    RGB channels to put in the LUT.
    """
    val_range = np.arange(256, dtype=np.uint8)
    LUT = np.zeros((3, 256), dtype=np.uint8)
    LUT[mixVector] = val_range
    return LUT


def LUTInterpreter(LUT):
    """
    You give a list of LUTs and it return the channel names.
    i.e. you can put imagej_metadata LUTs in and it tells you what colour 
    it is so that you can rebuild the LUTs when you resave for imagej. 
    Works alongside LUTDic.
    """

    assert LUT.shape[1]==3 and LUT.shape[2]==256,EM.LT1
    
    boolDic = {[True,True,True]:'Unknown_Grey',
               [False,False,True]:'Unknown_Green',
               [False,False,True]:'Unknown_Blue',
               [True,False,False]:'Unknown_Red',
              }
    
    boolVec = [[L[0].sum()==0,L[1].sum()==0,L[2].sum()==0] for L in LUT]
    boolVec = [[not BV[0],not BV[1],not BV[2]] for BV in boolVec]
    return [boolDic[bv] for bv in boolVec]


def LUT2ColourDict(lut):
    theLUTDic = {(True,True,True):'White',
                 (False,False,True):'Blue',
                 (False,True,False):'Green',
                 (True,False,False):'Red',
                 (True,True,False):'Yellow',
                 (False,True,True):'Cyan',
                 (True,False,True):'Purple'
                }
    
    assert lut in theLUTDic.keys(),EM.ch2
    
    return theLUTDic[lut]   


def getProcessedDataDs(xPath,xSig):
    """
    This function looks in the parent directory of the path given for any
    directories containing the signature xSig in their name.
    It returns a set of paths to those directories.
    """
    
    # get parent directory path
    parPath = os.path.split(xPath)[0]
    # get the names of all objects in that directory
    listDir = os.listdir(parPath)
    # filter to get just the directories
    allDirs = [d for d in listDir if os.path.isdir(os.path.join(parPath,d))]
    # filter for directories must contain the signature xSig
    allDirs = [os.path.join(parPath,d) for d in allDirs if xSig in d]
    
    return set(allDirs)



def listStr2List(listString,convertNumeric=True):
    """
    This converts a string of a python list to a list.
    Only currently works for elements that are ints or 'None'
    """
    reg = r'\[(.*)\]'
    list1 = re.search(reg,listString).group(1)
    list1 = list1.split(',')
    
    list2 = []
    for l in list1:
        if l=='None':
            list2.append(None)
        elif l.replace(' ','').isdecimal() and convertNumeric:
            list2.append(int(l.replace(' ','')))
        elif l.replace('.','',1).isdecimal() and convertNumeric:
            list2.append(float(l))
        else:
            list2.append(l.replace('\'','').replace(' ',''))
    
    return list2



def maskFromOutlinePath(outLinePath):
    """ 
    This takes the path of an image outline you have drawn and returns
    a binary mask with all values within the outline set to 
    1 and 0 elsewhere.
    The only requirement is that your outline has the pixel value that 
    is the highest in the image.
    """
    # import image
    with tifffile.TiffFile(outLinePath) as tif:
        outLine = tif.asarray()
    # normalise it
    outLine = outLine/np.max(outLine)
    # set non-maximum pixels to zero so we have a binary image
    outLine[outLine!=1.0] = 0
    # find connected components
    labels = label(outLine)
    # find the connected component with the most pixels
    # we assume this is your outline
    biggestComponent = np.bincount(labels.flatten())[1:].argmax()+1
    # set everything to zero except your outline
    labels[labels != biggestComponent] = 0
    labels[labels == biggestComponent] = 1
    # fill in the outline
    mask = binary_fill_holes(labels)
    # return your mask
    return mask



def shapeFromFluoviewMeta(meta):
    """ 
    This gets the 7D dimensions from fluoview metadata. 
    In some fluoview versions it doesn't include the dimension 
    name if the dimension size is 1 so we have to add it.
    """
    dims = meta['Dimensions']
    dimsDic = {l[0]:l[1] for l in dims}
    shapeKeys = ['Time','XY','Montage','Z','Wavelength','y','x']
    dims = []
    for k in shapeKeys:
        if k in dimsDic.keys():
            dims.append(dimsDic[k])
        else:
            dims.append(1)
    return dims



def tif2Dims(tif):
    """ This takes a tifffile.TiffFile object and returns dimensions 
        of the associated tifffile in the 7D format that we use. Currently
        works for fluoview and files we use in image j but we want to do 
        it for as many file types as possible.
        
        It returns an error if the data is not in one of these forms unless 
        there is only one image in which case it knows T,F,M,Z,C dims are 1. 
        This is useful because we use images of one frame to do things like 
        draw the masks but when you save them image j will erase all metadata.
    """
    fluo = 'fluoview_metadata'
    d = dir(tif)
    # for fluoview files:
    if fluo in d and tif.fluoview_metadata != None:
        meta = tif.fluoview_metadata
        dims = shapeFromFluoviewMeta(meta)                    
    # for image j ready files that we saved:
    elif ('imagej_metadata' in d and 
            tif.imagej_metadata != None and 
            'tw_nt' in tif.imagej_metadata.keys()):
        meta = tif.imagej_metadata
        baseString = 'tw_n'
        dimStrings = ['t','f','m','z','c','y','x']
        dims = [meta[baseString+L] for L in dimStrings]
    elif len(tif.asarray().shape)==2:
        dims = [1 for i in range(5)] + list(tif.asarray().shape)
    else:
        raise UnknownTifMeta()
    return dims



def meta2StartMom(meta):
    """ 
    This takes a session's metadata file and returns a datetime object 
    of the moment when the file was started.
    """
    # the format of the datatime string we give it
    TX = '%d/%m/%Y %H:%M:%S'
    startTimeReg = re.compile(r'Time=(\d\d:\d\d:\d\d)\n\[Created End\]')
    # start date regex:
    startDateReg = re.compile(r'\[Created\]\nDate=(\d\d/\d\d/\d\d\d\d)')
    # delay reg, i.e. time between session starting and imaging starting
    delayReg = re.compile(r'Delay - (\d+) (\w+)')
    # take start moment from the vth metadata
    startT = re.search(startTimeReg,meta).group(1)
    startDate = re.search(startDateReg,meta).group(1)
    startMom = startDate + ' ' + startT
    startMom = datetime.strptime(startMom,TX)
    
    # add the delay time if necessary
    if re.search(startTimeReg,meta):
        delayT = int(re.search(delayReg,meta).group(1))
        if re.search(delayReg,meta).group(2)=='min':
            delayT = timedelta(minutes=delayT)
            startMom += delayT
        elif re.search(delayReg,meta).group(2)=='hr':
            delayT = timedelta(hours=delayT)
            startMom += delayT
        else:
            raise Exception(EM.sm1)
             
    return startMom



def meta2TStep(meta):
    """ 
    This takes a session's metadata file and returns a timedelta object
    of the time between time points.
    """
    # time interval group(1), units group(2)
    DTReg = re.compile(r'Repeat T - \d+ times? \((\d+) (\w+)\)')
    
    # find the time between time-points of this TData (from its 
    # parent session metadata)
    seshTStep = int(re.search(DTReg,meta).group(1))
    
    if re.search(DTReg,meta).group(2) == 'hr':
        seshTStep = timedelta(hours=seshTStep)
    elif re.search(DTReg,meta).group(2) == 'min':
        seshTStep = timedelta(minutes=seshTStep)
    elif re.search(DTReg,meta).group(2) == 'sec':
        seshTStep = timedelta(seconds=seshTStep)
    else:
        raise Exception(EM.sm2)

    return seshTStep



def onlyKeepChanges(theList):
    """ 
    This makes a list from theList in which only elements which are 
    different from the previous are kept.
    """
    
    if len(theList)==0:
        return theList
    
    newList = []
    newList.append(theList[0])
    
    for l in theList[1:]:
        if l != newList[-1]:
            newList.append(l)
    
    return newList


def savedByXFoldQ(filepath):
    """ 
    Returns True if the files was saved by this package, False otherwise
    Tests this by looking at metadata.
    """
    with tifffile.TiffFile(filepath) as tif:
        d = dir(tif)
        if ('imagej_metadata' in d and 
            tif.imagej_metadata != None and 
            'tw_nt' in tif.imagej_metadata.keys()):
            return True
        else:
            return False
        
        
def saveTiffForIJ(
    outPath,
    data,
    chan=False,
    seshDs=False,
    autoscale=True,
    minP=2,
    maxP=98,
    overwrite=False,
    zSize=1,
    pixSizeX=1,
    pixSizeY=1,
    tstep=1,
    sesh_meta_dict=False,
    tdata_meta_dict=False,
    compress=False):
    """
    Uses tifffile package to save 7D data in way that is good for imagej.
    Also saves metadata inside the tif that xfold will recognise.
    
    Parameters
    ----------
    outPath : str
        File path to save to.
    data : numpy.array
        Image array. Must have 7 dimensions as in standard TData structure.
    chan : list of str
        The channel names. Used to set LUT. If not provide it will try to set 
        default channels but currently this only works if the number of 
        channels can be read from provided seshDs or the data shape. It only 
        knows how to read from the data shape if the shape has 2,3 or 7 axes.
    seshDs : list of lists
        Used to save as metadata, should be: [seshT,seshF,seshM,seshZ,seshC]. 
        Also used to try and reshape data to 7D if it is not 7D and set 
        default channels if they're not provided. If seshDs not provided then 
        it can continue only if the data shape has 2,3 or 7 axes.
    autoscale : bool
        Whether to autoscale or not. This doesn't change the data, it just 
        sets the metadata for imagej.
    minP : int [0-100]
        The percentile to set the min pixel value to during autoscaling.
    maxP : int [0-100]
        The percentile to set the max pixel value to during autoscaling.
    overwrite : bool
        Whether to allow overwriting of data with the same filename.
    sesh_meta_dict : dict
        If you are saving a tdata you can put the tdata.ParentSession.allMeta 
        dictionary here and it converts it to a string and saves in 'metadata'.
    compress : bool
        Whether to save a compressed image file using np.savez_compressed. Only 
        use for boolean masks.
        
    Notes
    -----
    It will try to reshape the data if it isn't 7D - it will work if 
    images are properly ordered and you give the correct seshDs. It will also 
    work if the data is 3D - it will assume the first axis is channels.
    """
    assert not (os.path.exists(outPath) and not overwrite),EM.sv1

    dims = data.shape
    dims = (dims[0]*dims[1]*dims[2],dims[3],dims[4],dims[5],dims[6])
    assert len(dims)<8 and len(dims)>1,'data must be maximum 7D and minimum 2D'        
        
    if compress:
        np.savez_compressed(outPath,mask=data)
        return
    
    defaultChans = ['BF','GFP','RFP','CFF','FR']
    if not chan:
        if seshDs:
            assert len(seshDs[4])<6,EM.sv4
            chan = defaultChans[:len(seshDs[4])]
        elif len(dims)==7:
            assert dims[-3]<6,EM.sv5
            chan = defaultChans[:dims[-3]]
        elif len(dims)==3:
            assert dims[0]<6,EM.sv5
            chan = defaultChans[:dims[0]]
        elif len(dims)==2:
            chan = ['BF']          
        else:
            raise Exception(EM.sv3)
    
    # reshape data according to seshDs if data is not 7D and seshDs provided
    if len(dims)!=7 and seshDs:
        seshDL = [len(s) for s in seshDs]
        nImMeta = np.product(seshDL)*dims[-1]*dims[-2]
        assert np.product(dims)==nImMeta,EM.sv2
        dims = seshDL + [dims[-2]] + [dims[-1]]
        data = data.reshape(dims)
    # reshape data if data has 3 axes and seshD not provided 
    # also set seshDs from dims
    elif len(dims)==3 and not seshDs:
        dims = [1,1,1,1]+[dims[-3]]+[dims[-2]]+[dims[-1]]
        data = data.reshape(dims)
        seshDs = [['None']*dims[0],
                  ['None']*dims[1],
                  ['None']*dims[2],
                  ['None']*dims[3],
                  ['None']*dims[4]]  
    # reshape data if data has 2 axes and seshD not provide    
    # also set seshDs from dims
    elif len(dims)==2 and not seshDs:
        dims = [1,1,1,1,1]+[dims[-2]]+[dims[-1]]
        data = data.reshape(dims)
        seshDs = [['None']*dims[0],
                  ['None']*dims[1],
                  ['None']*dims[2],
                  ['None']*dims[3],
                  ['None']*dims[4]]          
    # set seshDs if data has 7 axes and seshDs not provided
    elif len(dims)==7 and not seshDs:
        seshDs = [['None']*dims[0],
                  ['None']*dims[1],
                  ['None']*dims[2],
                  ['None']*dims[3],
                  ['None']*dims[4]] 
    elif len(dims)!=7 and not seshDs:
        raise Exception(EM.sv6)
    
    # this is the metadata to add:
    singleImQ = all([d==1 for d in dims[0:5]])
    meta = {'axes': 'TZCYX',
            'hyperstack':True,
            'ImageJ': '1.52a',
            'mode':'composite',
            'unit':'um',
            'spacing':zSize,
            'loop':False,
            'min':'0.0',
            'max':'256',
            'fps':1/tstep,
            'tw_NT':dims[0],'tw_NF':dims[1],'tw_NM':dims[2],
            'tw_NZ':dims[3],'tw_NC':dims[4],'tw_NY':dims[5],
            'tw_NX':dims[6],'tw_SeshT':str(seshDs[0]),
            'tw_SeshF':str(seshDs[1]),'tw_SeshM':str(seshDs[2]),
            'tw_SeshZ':str(seshDs[3]),'tw_SeshC':str(seshDs[4]),
            'tw_chan':str(chan)}
    if singleImQ:
        minV = np.percentile(np.ravel(data),minP)
        maxV = np.percentile(np.ravel(data),maxP)
        meta['min'] = str(minV)
        meta['max'] = str(maxV)
    if isinstance(sesh_meta_dict,dict):
        meta['session_meta_data'] = meta_dict_2_str(sesh_meta_dict)
    if isinstance(tdata_meta_dict,dict):
        meta['tdata_meta_data'] = meta_dict_2_str(tdata_meta_dict)
    
    # tifffile requires a dictionary ijmeta which it converts to binary for ij
    # set ranges for each channel:
    if autoscale:
        ranges = []
        for c,ch in enumerate(chan):
            minV = np.percentile(np.ravel(data[:,:,:,:,c]),minP)
            ranges.append(minV)
            maxV = np.percentile(np.ravel(data[:,:,:,:,c]),maxP)
            ranges.append(maxV)
    else:
        ranges = [x for i in range(dims[4]) for x in [0.0,65535.0]]
    # make the LUTs
    LUTs = [LUTMixer(LUTDic(c)) for c in chan]
    # package ranges and LUTs into imagej metadata dictionary 
    ijmeta = {'Ranges':tuple(ranges),'LUTs':LUTs}
        
    # do the save, reshaping the array for image j at the last moment
    dims = (dims[0]*dims[1]*dims[2],dims[3],dims[4],dims[5],dims[6])
    
    meta.update(ijmeta)
    
    tifffile.imsave(outPath,
                    data.reshape(dims),
                    imagej=True,
                    resolution=(1/pixSizeX, 1/pixSizeY, 'MICROMETER'),
                    metadata=meta)
    return meta
    
    
def trap2rect(points):
    ymin = min([p[0] for p in points])
    ymax = max([p[0] for p in points])
    xmin = min([p[1] for p in points])
    xmax = max([p[1] for p in points])
    return [[ymin,xmin],[ymin,xmax],[ymax,xmax],[ymax,xmin]]    

  
    
def getTagDic(fp,madeBy,customTags={}):
    """
    When the data along a certain dimension Q is split between different files 
    the file name is given a tag. This function returns that tag in a 
    dictionary. The keys of the dictionary are the 'Q' of any found tags and 
    the values are the tags. 
    
    Our code in general is going to assume that all files with the same tag 
    contain equivalent Q-points. That is, all files with time tag '_t0004' 
    have the same set of time points even if they contain different subsets 
    of other dimensions, e.g. you could have _t0004_m0000 and _t0004_m0001 
    but both contain time points e.g. t=6,7,8.
    
    Parameters
    -----------
    fp : str
        The filepath.
    madeBy : {'Andor','MicroManager'}
        The software that made the image.
    
    Returns
    ----------
    tagDic : dict
        Keys can be any of ('T','F','M','Z','C')
        Values are the found tags.
    """
    from .Classes import XFold

    tagDic = {}
    if madeBy=='Andor':
        fp = fp[:-4]
        endTagReg = r'(_(t|f|m|z|w)\d{3,9})$'
        # takes tags off end one by one
        tag = re.search(endTagReg,fp)
        foundTags = []
        while tag:
            foundTags.append(tag.group(1))
            fp = fp[:-len(tag.group(1))]
            tag = re.search(endTagReg,fp)
        for t in foundTags:
            if t[1]=='w':
                tagDic.update({'C':t})
            else:
                tagDic.update({t[1].upper():t})
    
    elif madeBy=='MicroManager':
        fTag = fp.split('MMStack_')[1][:-4]
        fTag = re.sub('.ome','',fTag)
        fTag = re.sub(r'_\d\d\d_\d\d\d','',fTag)
        #fTag = re.search(r'(MMStack_\d)',fp) # old way
        if fTag:
            tagDic.update({'F':fTag})
        mTag = re.search(r'(Pos_\d\d\d_\d\d\d)',fp)
        if mTag:
            tagDic.update({'M':mTag.group(1)})

    elif madeBy=='Incucyte':
        # there are only 2 tags so far and these should be on every file
        tagRegex = re.compile(r"_([A-Z]\d+_\d+)_(\d{4}y\d{2}m\d{2}d_\d{2}h\d{2}m).tif$")   
        tagRegex2 = re.compile(r"_([A-Z]\d+_\d+)_(\d{2}d\d{2}h\d{2}m).tif$")
        if re.search(tagRegex,fp):
            tagDic.update({'T':re.search(tagRegex,fp).group(2)})
            tagDic.update({'F':re.search(tagRegex,fp).group(1)})
        elif re.search(tagRegex2,fp):
            tagDic.update({'T':re.search(tagRegex2,fp).group(2)})
            tagDic.update({'F':re.search(tagRegex2,fp).group(1)})
        else:
            raise Exception('unrecognised Incucyte filename in getTagDic')

    elif madeBy=='multisesh': 
        tagRegex = re.compile(r'_t(\d{4}|None)(_C(_Ch[^\W_]+){1,10})?(_m(\d{4}|None))?.tif?f$')    
        if re.search(tagRegex,fp).group(1):
            tagDic.update({'T':re.search(tagRegex,fp).group(1)})
        if re.search(tagRegex,fp).group(3):
            tagDic.update({'C':re.search(tagRegex,fp).group(3)})
        if re.search(tagRegex,fp).group(5):
            tagDic.update({'M':re.search(tagRegex,fp).group(4)})            
        if os.path.split(fp)[0]: 
            path1,_ = os.path.split(fp)
            _,FTag = os.path.split(path1)    
            tagDic.update({'F':FTag.replace(XFold.FieldDir,'')})

    elif madeBy=='multisesh_compressed': 
        tagRegex = re.compile(r'_t(\d{4}|None)(_C(_Ch[^\W_]+){1,10})?(_m(\d{4}|None))?.npz$')    
        if re.search(tagRegex,fp).group(1):
            tagDic.update({'T':re.search(tagRegex,fp).group(1)})
        if re.search(tagRegex,fp).group(3):
            tagDic.update({'C':re.search(tagRegex,fp).group(3)})
        if re.search(tagRegex,fp).group(5):
            tagDic.update({'M':re.search(tagRegex,fp).group(4)})            
        if os.path.split(fp)[0]: 
            path1,_ = os.path.split(fp)
            _,FTag = os.path.split(path1)    
            tagDic.update({'F':FTag.replace(XFold.FieldDir,'')})            
        
    elif madeBy=='Opera':
        tagRegex = re.compile(r'[\s\S]*(r\d+c\d+)f(\d+)(?:p(\d+))?-ch(\d+)sk1fk1fl1.tif(?:f)?$')
        tags_found = re.search(tagRegex,fp)

        if tags_found:
            if not tags_found.group(1)==None:
                tagDic.update({'F':tags_found.group(1)})
            if not tags_found.group(2)==None:
                tagDic.update({'M':tags_found.group(2)})
            if not tags_found.group(3)==None:
                tagDic.update({'Z':tags_found.group(3)})
            if not tags_found.group(4)==None:
                tagDic.update({'C':tags_found.group(4)})
        else:
            raise Exception('%s not found in %s' % (tagRegex,fp))
    elif madeBy=='aicsimageio_compatible' and customTags:
        for k,v in customTags.items():
            tagDic.update({k:re.search(v,fp).group(0)})
    elif madeBy=='Leica':
        pass
    elif madeBy=='aicsimageio_compatible':
        pass
    elif madeBy=='Zeiss':
        # haven't seen file splitting in czi yet
        pass
    elif madeBy=='Nikon':
        # haven't seen file splitting in nd2 yet
        pass        
    else:
        raise Exception(EM.md1)
    
    return tagDic



def unravel(T,F,M,Z,C,NT,NF,NM,NZ,NC):
    """
    this function converts 7D indices to their flat equivalent index
    given the shapes of those dimensions of course
    actually it only does 5D of the 7D, XY ignored here    
    """
    return T*NF*NM*NZ*NC + F*NM*NZ*NC + M*NZ*NC + Z*NC + C



def UDRL2LRUD(I,NY,NX):
    """
    Give this a list of indices for tiles in left-right-up-down ordering and 
    it returns the list of where you find those tiles in a up-down-right-left 
    ordered list of tiles. Hence it helps to convert UDLR -> LRUD.
    
    Parameters
    -----------
    I : 1d list
        the list of indices in LRUD for which you would like to find the 
        corresponding UDRL indices.
    NY,NX : int
        The sizes of the full montages.
    
    Returns
    -------
    I2 : 1d list
        Each element of the original list I converted to UDRL indices.
    """
    
    R = [i//NX for i in I]
    C = [i%NX for i in I]
    
    return [(NX-1-c)*NY + r for i,r,c in zip(I,R,C)]


def removePads(im,pads=None,retPad=False):
    """
    Removes padding from a 2D image by either detecting the blank (0) edges 
    or according to user provided padding. Padding is defined as 
    rows/columns of 0. It returns the image without padding and (optionally) 
    the padding that was removed so you can add it back later if you like.
    
    Parameters
    ----------
    im : 2D numpy.array
        The image.
    pads : None or [int,int]
        If None then it will find the padding by detecting blank rows/columns 
        at edges. Otherwise it gives the total number of blank pixels that are 
        to be removed from the [y,x]-axes. See code for how this total number 
        is divided between the two sides.
    retPad : bool
        Whether to return the pad along with cropped image.
    
    Returns
    -------
    im : 2D numpy.array
        The image without padding.
    pad : tuple of tuple of ints
        If returnPad then how many pixels were removed is returned in: 
        (top,bottom,left,right). In this case im and pad are a tuple.
    """
    ny,nx =im.shape
    if pads is None:
        for t in range(ny):
            if any(im[t,:]):
                break
        for b in reversed(range(ny)):
            if any(im[b,:]):
                break
        for l in range(nx):
            if any(im[:,l]):
                break
        for r in reversed(range(nx)):
            if any(im[:,r]):
                break
    else:
        t,b = [pads[0]//2,ny-math.ceil(pads[0]/2)-1]
        l,r = [pads[1]//2,nx-math.ceil(pads[1]/2)-1]
    if retPad:
        return (im[t:b+1,l:r+1],(t,ny-b-1,l,nx-r-1))
    else:
        return im[t:b+1,l:r+1]
    
    
def pad2Size(im,sizes):
    """
    Pads the image to the specified size, using the convention of 
    sessionStitcher to decide where to put extra pixels.
    
    Parameters
    ----------
    im : array-like
        The image you want to pad. Can be any dimension, always the last two 
        axes are the ones that will be padded.
    sizes : tuple
        The (ysize,xsize) that you pad to.
    
    Returns
    -------
    im : array like
        The padded image.
    """
    pY = sizes[0] - im.shape[-2]
    pX = sizes[1] - im.shape[-1]
    pY = ((pY//2,math.ceil(pY/2)),)
    pX = ((pX//2,math.ceil(pX/2)),)
    pad = ((0,0),)*(im.ndim - len(sizes)) + pY + pX
    return np.pad(im,pad)  


def takeWindow(im,win):
    """
    Extracts window from an image when window is given in format of 
    XFold.WindowDic, which is [[y,x],[y,x],[y,x],[y,x]]. Image is assumed to 
    be size of the template that the window was drawn on, no padding 
    considered.
    
    Parameters
    ----------
    im : numpy.array
        2D image that will be cropped.
    win : list of lists of ints
        The window in XFold.WindowDic format.
    """
    return im[win[0][0]:win[2][0],win[0][1]:win[2][1]]


def makeuint8(img,quantH=0.95,quantL=False):
    """
    Converts uint16 to uint8 with re-scaling to the max value. Can also do 
    clipping (max or max and min) to stop detail loss from anomalous high 
    values.
    """
    maxV = np.quantile(img,quantH)
    img[img>maxV] = maxV
    if quantL:
        minV = np.quantile(img,quantL)
        img[img<minV] = minV
    if quantL:
        return (255*((img-minV)/(maxV-minV))).astype('uint8')
    else:
        return (255*(img/maxV)).astype('uint8')
    
    
def ExtractProfileRectangle(data,outDir,outName=False,masks=None,
                            NXSegs=10,downsizeY=2,overwrite=False,
                            returnData=False):
    """
    This saves csvs containing averaged pixel values taken from the image 
    data. NXSegs and downsizeY define a grid in which the mean of each 
    square is returned.

    Parameters
    ----------
    data : array-like, shape (NIm,sizeY,sizeX)
        The image data. One csv will be saved for each image along axis 0.
    outDir : str or bool
        Path of directory to save csvs to. This directory will be created if 
        needed. If False then csvs aren't saved
    outName : str or list of str
        The name of the csv(s). If False then it will create a name and number 
        according NIm. If you provide a list then the names correspond to the 
        images along axis 0 of the data.
    masks : array-like, shape (NIm,sizeY,sizeX) or (sizeY,sizeX)
        Image(s) which define(s) a region of arbitrary shape to decide which 
        pixels to include. This image should be binary with included regions 
        as white.
    NXSegs : int
        The number of segments to return along the x-axis. All pixel
        values are averaged within each segment. The X position in pixels
        of the middle of each segment will be given in the CSV row heading
        (measured relative to the window).
    downsizeY : int
        Factor to downsize y-axis length by. Pixels are averaged. Position
        put in CSV headings too.
    overwrite : bool
        Whether to allow csvs to be overwritten.
    returnData : bool
        Whether to return the extracted arrays of data.

    Returns
    -------
    The extracted data are numpy arrays with NXSegs columns and NY//downsizeY 
    rows. You can either save these as csvs or return as a list of arrays (one 
    array for each image along axis). One separate csv for each image along 
    data axis 0.
    """
    if len(data.shape)==2:
        data = data[np.newaxis]
        
    NIm,sizeY,sizeX = data.shape

    if returnData:
        outList = []

    for i in range(NIm):
        # initiate np array and steps for collecting data
        dy = downsizeY
        NY = int(sizeY/dy)
        dx = int(sizeX/NXSegs)
        _csv = np.zeros((NY,NXSegs))
        for y in range(NY):
            for x in range(NXSegs):
                # input the data
                blockD = data[i,y*dy:(y+1)*dy,x*dx:(x+1)*dx].copy()
                if len(masks.shape)==3:
                    blockM = masks[i,y*dy:(y+1)*dy,x*dx:(x+1)*dx].copy()
                else:
                    assert len(masks.shape)==2,'masks not of shape we can handle'
                    blockM = masks[y*dy:(y+1)*dy,x*dx:(x+1)*dx].copy()
                result = blockD[blockM==1]
                if result.size==0:
                    _csv[y,x] = 'NaN'
                else:
                    _csv[y,x] = np.mean(result)
                del blockD
                del blockM
        if returnData:
            outList.append(_csv)

        if outDir:
            # add x-slice and y-distance headings
            xlab = [str(dx*x+dx//2) for x in range(NXSegs)]
            xlab = np.array(['x='+x+' pixels' for x in xlab])
            ylab = ['y='+str(dy*y+dy//2)+'pixels' for y in range(NY)]
            ylab = np.array(['Y distance']+ylab)
            _csv = np.vstack((xlab,_csv))
            _csv = np.hstack((ylab.reshape((NY+1,1)),_csv))
        
            if not outName:
                outName1 = 'ExtractedValues_Image_'+str(i).zfill(4)+'.csv'
            else:
                if isinstance(outName,list):
                    assert len(outName)==NIm,EM.ae9
                    outName1 = outName[i]
                else:
                    outName1 = outName
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            outPath = os.path.join(outDir,outName1)
            if not overwrite:
                assert not os.path.exists(outPath),EM.ae5

            # 10/24 changed this to pandas to avoid csv import
            #with open(outPath,'w',newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerows(_csv)
            
            pd.DataFrame(_csv).to_csv(outPath)
            
    if returnData:
        return outList

        
            
def adjustFilterEdges(filt,DF,tileA,tileB,molap):
    """
    This improves the homogenisation filter using the knowledge that the 
    bottom fraction of tileA overlaps with the top section of tileB, with a 
    fractional overlap size of molap.
    
    Parameters
    ----------
    filt : array-like shape (NY,NX)
        The original filter, e.g. calculated by BaSiC.
    DF : array-like shape (NY,NX)
        The original dar-fileld filter, e.g. calculated by BaSiC.        
    tileA : array-like shape (NY,NX)
        A tile in the montage where there is good signal all at the bottom.
    tileB : array-like shape (NY,NX)
        The tile in the montage directly below tileA.
    molap : float
        The decimal fraction of tile A and tileB that overlap.
    """
    
    # extract section of tiles and filter that should overlap
    ny,nx = tileA.shape
    segA_B = (tileA-DF)[ny-math.floor(ny*molap)-1:-1]
    segB_T = (tileB-DF)[:math.floor(ny*molap)]
    segF_B = filt[ny-math.floor(ny*molap)-1:-1]
    segF_T = filt[:math.floor(ny*molap)]
    
    ny_seg = segA_B.shape[0]
    
    # y is essentially the equation that says that the corrected image sections 
    # should be the same... solve that equation with best fit straight line
    # that gives you the factor to correct the filter with
    x = np.array(range(ny_seg))
    y = np.mean((segA_B/segB_T)/(segF_B/segF_T),axis=1)
    a,b = np.polyfit(x,y,1)
    
    A = np.ones((ny))
    for i in range(ny):
        if i > ny - ny_seg:
            A[i] = math.sqrt(a*(i - (ny - ny_seg)) + b)
        elif i < ny_seg:
            A[i] = 1/math.sqrt(a*i + b)            
    
    start_T = int(3*ny_seg/4)
    end_T = int(5*ny_seg/4)
    max_T = A[start_T]
    for i in range(start_T,end_T):
        A[i] = max_T + (i-start_T)*((1-max_T)/(ny_seg/2))
        
    start_B = int(ny-ny_seg-(ny_seg/4))
    end_B = int(ny-ny_seg+(ny_seg/4))
    max_B = A[end_B]
    for i in range(start_B,end_B):
        A[i] = 1 + (i-start_B)*((max_B-1)/(ny_seg/2))        
        
    A = A[:,np.newaxis]
    A = np.tile(A,(1,nx))
    
    filt = filt*A
    filt = filt/np.mean(filt)
    
    return (A,filt)



def signalNoiseMasks(mask,mask2=None):
    """
    You provide a mask and it returns two masks, one of definite signal 
    and one of definite noise. I.e. it erodes/dilates/inverts the provided 
    mask of signal to be sure.
    
    mask : array-like
        The mask of a segmentation of signal.
    mask2 : False or array-like or str
        A further mask to restrict the region from which signal is taken.
    
    Returns
    -------
    sig : array-like
        The mask of where signal definitely is.
    noise : array-like
        The mask of where noise definitely is.
    """
    # nuclei and noise masks
    kernel = np.ones((3,3),np.uint8)
    sig = cv.erode(mask,kernel)
    mask2 = cv.erode(mask.astype('uint16'),kernel)
    nucleiMask = np.logical_and(nucleiMask,mask2)
    noise = 1-cv.dilate(mask,kernel)
    noise = np.logical_and(noise,mask2)
    
    return sig,noise




def groupByTag(filepaths,regex):
    """
    This goes through all filepaths and groups them according to the part 
    returned by the regex (i.e. the tag). 
    
    Parameters
    ----------
    
    filepaths : list of strings
        The filepaths
    regex : regex
        A compiled regex that you search for in each file path

    Returns
    --------
    groupedFPs : dict
    In format {tag:{list of files}}
    """
    groupedFPs = {}
    
    for fp in filepaths:
        
        match = re.search(regex, fp)
        if match:
            match = match.group(0) 
            if not match in groupedFPs.keys():
                groupedFPs.update({match:[]})
            groupedFPs[match].append(fp)
        else:
            raise Exception('regex wasn\'t found in {fp}')    
            
    return groupedFPs


def sort_grid_points(x_coords, y_coords):
    """
    If you have a grid of x,y coordinates it arranges it into LRUD order.
    It returns the indices that will reorder your list.
    """
    # Combine the x and y coordinates into a list of tuples
    points = list(zip(x_coords, y_coords))
    
    # Sort the points first by y-coordinate in descending order, then by x-coordinate in ascending order
    sorted_points = sorted(points, key=lambda point: (-point[1], point[0]))

    indices = [points.index(sp) for sp in sorted_points]
    
    ## Separate the sorted points back into x and y coordinates
    #sorted_x_coords, sorted_y_coords = zip(*sorted_points)
    
    return indices#sorted_x_coords, sorted_y_coords


def find_grid_indices(ysize, xsize, N=3):
    """
    If you have an image that you want to split into an N-by-N grid, this 
    gives you the start and end pixel indices.

    Parameters 
    ----------
    ysize,xsize : int
        The size of the image you are splitting.
    N : int 
        It will be a N-by-N grid.

    Returns
    -------
    ((y_min,y_max),(x_min,x_max))
    """
    # Define the step sizes for y and x directions
    y_step = ysize / N
    x_step = xsize / N

    # Initialize the list to store the centers
    indices = []

    # Loop over the grid
    for i in range(N):
        for j in range(N):
            # Calculate the center of the current rectangle
            y_min = int(i * y_step)
            x_min = int(j * x_step)
            y_max = int((i+1) * y_step)
            x_max = int((j+1) * x_step)

            # Append the center to the list
            indices.append(((y_min,y_max),(x_min,x_max)))

    return indices


def clip_and_rescale_image(image, low_percentile, high_percentile):
    """
    It returns the image at the same scale but with the low_ and high_ 
    percentile pixel values clipped and the whole histogram rescaled so it 
    uses the full range of the image's dtype.
    """
    dtype = image.dtype
    # Calculate the percentile values
    low_value = np.percentile(image, low_percentile)
    high_value = np.percentile(image, high_percentile)
    
    # Clip the image
    clipped_image = np.clip(image, low_value, high_value).astype('float32')
    
    # Rescale the image
    old_range = high_value - low_value  # Old range of pixel values
    new_range = np.iinfo(dtype).max - np.iinfo(dtype).min  # New range of pixel values
    
    # Rescale the pixel values to the new range
    rescaled_image = (((clipped_image - low_value) * new_range) / old_range) + np.iinfo(dtype).min
    
    # Convert the image to the desired data type
    rescaled_image = rescaled_image.astype(dtype)
    
    return rescaled_image


def get_modes(data,exclude=False):
    """
    This just returns the modes (in statistical average sense of the word) 
    from a list. Needed to write this function 
    because most python equivalents seem to only return one of the modes when 
    there are several.
    """
    if exclude:
        data = [d for d in data if d!=exclude]
    counter = Counter(data)
    max_count = max(counter.values())
    modes = [item for item, count in counter.items() if count == max_count]
    return modes


def moving_average(arr, window_size):
    """
    This does moving average of a numpy array. np.nan are handled properly by 
    ignoring them and properly calculating means with the remaining values. It 
    works on 1D arrays and otherwise works along axis 0, e.g. treating 
    different positions along axis 1 as independent.
    """
    ret = np.cumsum(np.where(np.isnan(arr), 0, arr), axis=0)
    n_nan = np.cumsum(np.isnan(arr), axis=0)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    n_nan[window_size:] = n_nan[window_size:] - n_nan[:-window_size]
    return ret[window_size - 1:] / (window_size - n_nan[window_size - 1:])


def meta_dict_2_str(meta_dict):
    """
    This converts a multisesh metadata dictionary to a string, allowing 
    conversion between common datatypes that we have in multisesh metadata.

    Note how imagej metadata doesn't like = signs so we replace them!!
    """
    meta_dict2 = meta_dict.copy()

    for k,v in meta_dict2.items():
        if isinstance(v,datetime):
            meta_dict2[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(v,timedelta):
            meta_dict2[k] = str(v)
        elif isinstance(v,np.ndarray):
            list_k = list(v)
            meta_dict2[k] = [int(i) for i in list_k]           
        elif k=='metadata':
            if not isinstance(v,str) and not isinstance(v,int) and not isinstance(v,float):
                meta_dict2[k] = 'full metadata in non-supported format'            
        elif isinstance(v,dict):
            change = False
            dict2 = v.copy()
            for kk,vv in dict2.items():
                if isinstance(vv,datetime):
                    change = True
                    dict2[kk] = vv.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(vv,timedelta):
                    change = True
                    dict2[kk] = str(vv)
            if change:
                meta_dict2[k] = dict2.copy()
    return json.dumps(meta_dict2).replace('=',' equals ')


def str_2_timedelta(meta_str):
    """
    When str() is used to convert a timedelta to a str, this should convert 
    it back.
    """
    
    if 'days' in meta_str:
        days, meta_str = meta_str.split(' days, ')
    else:
        days = 0
        
    hours, minutes, seconds = map(int, meta_str.split(':'))
    
    return timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)


def timedeltaStrQ(timedelta_str):
    """
    This tests if the provided string fits any of the possible formats of a 
    timedelta.
    """
    delta_pat1 = r"^\d+ days, \d+:\d{2}:\d{2}.\d+$"
    delta_pat2 = r"^\d+:\d{2}:\d{2}.\d+$"
    delta_pat3 = r"^\d+:\d{2}:\d{2}$"
    delta_pat4 = r"^\d+ days, \d+:\d{2}:\d{2}$"
    test1 = bool(re.match(delta_pat1,timedelta_str))
    test2 = bool(re.match(delta_pat2,timedelta_str))
    test3 = bool(re.match(delta_pat3,timedelta_str))
    test4 = bool(re.match(delta_pat4,timedelta_str))
    return any([test1,test2,test3,test4])



def meta_str_2_dict(meta_str):
    """
    This converts a string representing a multisesh metadata dictionary back 
    into a meatadata dictionary.
    """
    date_pat = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    
    meta_dict = json.loads(meta_str)

    # json converts tuples into lists so need to convert back
    if 'Shape' in meta_dict.keys():
        meta_dict['Shape'] = tuple(meta_dict['Shape'])

    if 'LRUPOrdering' in meta_dict.keys():
        meta_dict['LRUPOrdering'] = np.array(meta_dict['LRUPOrdering'])

    for k,v in meta_dict.items():
        if isinstance(v,str) and bool(re.match(date_pat,v)):
            meta_dict[k] = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        elif isinstance(v,str) and timedeltaStrQ(v):
            meta_dict[k] = str_2_timedelta(v)
        elif isinstance(v,dict):
            for kk,vv in meta_dict[k].items():
                if isinstance(vv,str) and bool(re.match(date_pat,vv)):
                    meta_dict[k][kk] = datetime.strptime(vv, "%Y-%m-%d %H:%M:%S")
                elif isinstance(vv,str) and timedeltaStrQ(vv):
                    meta_dict[k][kk] = str_2_timedelta(vv)
                
    return meta_dict


def orderFilePathsByField(TPs,madeBy,xfold):
    """
    This reorders the session filepaths to make sure they are ordered 
    correctly according to SeshF (since SeshF will be assigned according to 
    this ordering in Session.makeTFiles().

    In multisesh it is doing this by the saved SeshF metadata.

    Parameters
    ---------
    TPs : list of str
        The list of file paths that must all be from the same session.
    madeBy : str
        One of the accepted madeBy formats.
    xfold : ms.XFold
        If madeBy=='multisesh_compressed' then there is no metadata inside the 
        files so you take it from the OriginalXFold.
    """
    if madeBy=='multisesh':
        SeshFs = []
        for seshtp in TPs:
            with tifffile.TiffFile(seshtp) as tif:
                I = tif.imagej_metadata
                SeshFs.append(meta_str_2_dict(I['tdata_meta_data'])['SeshF'])

        return [seshTP for SeshF,seshTP in sorted(zip(SeshFs,TPs))]
    elif madeBy=='multisesh_compressed':
        tagRegex = re.compile(r'_s(\d{4})')    
        if re.search(tagRegex,TPs[0]).group(1):
            seshN = int(re.search(tagRegex,TPs[0]).group(1))
        else:
            raise Exception('no SessionN found')         
        FIDs = xfold.SessionsList[seshN].FieldIDMap
        SeshFs = []
        for seshtp in TPs:
            path1,_ = os.path.split(seshtp)
            _,FTag = os.path.split(path1)
            FTag = FTag.replace(defs.FieldDir,'')              
            SeshFs.append(FIDs.index(FTag))
        return [seshTP for SeshF,seshTP in sorted(zip(SeshFs,TPs))]
    else:
        return TPs
                
        
        
def nucl_detect_yolo(img, model, clip=[1, 99], augment=False, conf=0.05, iou=0.3, max_det=3000):
    """
    Using YOLO to detect nuclei.
    
    Args:
    img (2D, NumPy array): input image of shape (H, W)
    """
        
    # Nornalize image & Detect nuclei
    pmin, pmax = np.percentile(img, clip)
    img = img.clip(pmin, pmax)
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min)/(img_max - img_min)
    img *= 255.0
    img = img.astype('uint8')
    image = np.stack([img]*3, axis=2)
    results = model(image, augment=augment, conf=conf, iou=iou, max_det=max_det)

    # Return bboxes as DataFrame
    Boxes = results[0].boxes.cpu().numpy()
    if len(Boxes) == 0:
        return None
    else:
        df = pd.DataFrame(columns=['X1', 'Y1', 'X2', 'Y2', 'Conf', 'Cls'])
        df[['X1', 'Y1', 'X2', 'Y2']] = Boxes.xyxyn
        df['Conf'] = Boxes.conf
        df['Cls'] = Boxes.cls.astype(int)
        return df


def nucl_segment_sam(img, bboxes, predictor):
    """
    Using SAM to segment nuclei that have been detected and whose positions are provided by bboxes.
    """
    predictor.set_image(img)

    transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, img.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.cpu().numpy()

    return masks


def patchify2(image, tile_size, min_olap):
    """
    This breaks the image into tile of size tile_size. It chooses the number 
    of tiles N that makes tiles fit perfectly with an overlap >= min_olap 
    (corrections for non-integer olap are made in right-most tile).
    """
    if image.shape[0]==tile_size:
        return [image],1,0,0
    NY_im = image.shape[0]

    N_tiles = int(np.ceil((NY_im - min_olap) / (tile_size - min_olap)))
    #print('N_tiles: ',N_tiles)

    olap = int(np.floor( ((N_tiles*tile_size) - NY_im)/ (N_tiles - 1) ))
    #print('olap: ',olap)

    olap_f = (N_tiles-2)*(tile_size - olap) + 2*tile_size - NY_im
    #print('olap_f: ',olap_f)

    patches = np.zeros((N_tiles,N_tiles,tile_size,tile_size))
    for i in range(N_tiles):
        for j in range(N_tiles):
            #print(i,j)
            if i==(N_tiles-1):
                sliceY = slice((i - 1)*(tile_size - olap) + tile_size - olap_f,
                               (i+1)*tile_size - (i - 1)*olap - olap_f)
            else:
                sliceY = slice(i*(tile_size - olap),
                               (i+1)*tile_size - i*olap)
            
            if j==(N_tiles-1):
                sliceX = slice((j - 1)*(tile_size - olap) + tile_size - olap_f,
                               (j+1)*tile_size - (j - 1)*olap - olap_f)
            else:
                sliceX = slice(j*(tile_size - olap),
                               (j+1)*tile_size - j*olap)                
                
            patches[i,j] = image[sliceY,sliceX]
    return patches,N_tiles,olap,olap_f
            

def unpatchify2(patches,N_tiles,olap,olap_f,outsize,tile_size):
    """
    This puts images back together after patchify has taken them apart.
    """
    if patches.shape[0]*patches.shape[1]==1:
        return patches[0]
        
    olap2 = olap/2
    olap_f2 = olap_f/2
    
    unpatch = np.zeros(outsize)
    for i in range(N_tiles):
        for j in range(N_tiles):
            #print("i: ",i,"j: ",j)
            if i==0:
                sliceY_out = slice(None,int(tile_size - np.ceil(olap2)))
                sliceY_patch = slice(None,int(tile_size - np.ceil(olap2)))
            elif i==(N_tiles-2):
                sliceY_out = slice(int(tile_size*i - olap*(i-1) - np.ceil(olap2)),
                                   int(tile_size*(i+1) - olap*i - np.ceil(olap_f2)))
                sliceY_patch = slice(int(np.floor(olap2)),
                                     int(tile_size - np.ceil(olap_f2)))
            elif i==(N_tiles-1):
                sliceY_out = slice(int(-(tile_size - np.floor(olap_f2))),None)
                sliceY_patch = slice(int(np.floor(olap_f2)),None) 
            else:
                sliceY_out = slice(int(tile_size*i - olap*(i-1) - np.ceil(olap2)),
                                   int(tile_size*(i+1) - olap*i - np.ceil(olap2)))
                sliceY_patch = slice(int(np.floor(olap2)),
                                     int(tile_size - np.ceil(olap2)))
                
            if j==0:
                sliceX_out = slice(None,int(tile_size - np.ceil(olap2)))
                sliceX_patch = slice(None,int(tile_size - np.ceil(olap2)))
            elif j==(N_tiles-2): 
                sliceX_out = slice(int(tile_size*j - olap*(j-1) - olap2),
                                   int(tile_size*(j+1) - olap*j - np.ceil(olap_f2)))
                sliceX_patch = slice(int(np.floor(olap2)),
                                     int(tile_size - np.ceil(olap_f2)))
            elif j==(N_tiles-1):            
                sliceX_out = slice(int(-(tile_size - np.floor(olap_f2))),None)
                sliceX_patch = slice(int(np.floor(olap_f2)),None)
            else:
                sliceX_out = slice(int(tile_size*j - olap*(j-1) - np.ceil(olap2)),
                                   int(tile_size*(j+1) - olap*j - np.ceil(olap2)))
                sliceX_patch = slice(int(np.floor(olap2)),
                                     int(tile_size - np.ceil(olap2)))
            
            unpatch[sliceY_out,sliceX_out] = patches[i,j,sliceY_patch,sliceX_patch]
    return unpatch


def rotate_mask(im,degrees,centre,increase_outsize_to_fit_rotation=True,recentre_first=False,crop_to_object=False):
    """
    This is a very general function for rotating images with lots of options 
    for control. It is designed for rotating masks and so converts back to 
    bool to clean up interpolation.

    Parameters
    ----------
    im : 2D numpy array
        The mask to be rotated
    degrees : {int,float}
        The degrees you rotate anti-clockwise.
    centre : tuple of {int,float}
        The centre that you rotate about in format (y,x)
    increase_outsize_to_fit_rotation : bool
        Whether to increase the output image size to fit the full original 
        image rotated. Directly using skimage for this. I.e. happens during 
        rotation.
    recentre_first : bool
        Whether to transform so the centre you provide is at the centre of the 
        image before rotation is applied. This is always done by adding to the 
        image, i.e. pads on sides that need to push specified centre to new 
        output image centre.
    crop_to_object : bool
        Whether to crop the output image to perfectly fit the mask, without 
        border.
    """
    cy0,cx0 = tuple(int(c) for c in centre)
    
    if recentre_first:
        ny0,nx0 = im.shape
        
        if cy0>=ny0-cy0:
            ny1 = 2*cy0
            cy1 = int(ny1/2)
            im = np.pad(im,((0,cy1-(ny0-cy0)),(0,0)))
        else:
            ny1 = 2*(ny0-cy0)
            cy1 = int(ny1/2)
            im = np.pad(im,((cy1-cy0,0),(0,0)))    
        if cx0>=nx0-cx0:
            nx1 = 2*cx0
            cx1 = int(nx1/2)
            im = np.pad(im,((0,0),(0,cx1-(nx0-cx0))))
        else:
            nx1 = 2*(nx0-cx0)
            cx1 = int(nx1/2)
            im = np.pad(im,((0,0),(cx1-cx0,0)))  
        cy0,cx0 = (cy1,cx1)
    
    im = transform.rotate(im, degrees, resize=increase_outsize_to_fit_rotation, center=(cx0,cy0)).astype('bool').astype('uint8')

    if crop_to_object:
        region = regionprops(im)[0]
        im = im[region.slice]
    
    return im.astype('bool')


def alpha_composite_over(imA,imB):
    """
    Alpha compositing is the formal way to combine images that have alpha 
    channels. The over operation is placing A on top of B. I.e. a transparency 
    overlay.

    ImA,B : numpy array shape (NY,NX,4)
        Should be dtype float and in [0,1]. uint8 are converted by division 
        by 255.
    """
    if imA.dtype=='uint8' or imA.dtype=='uint16':
        # Ensure the alpha channel is normalized to [0, 1]
        imA = imA / 255.0
        imB = imB / 255.0

    A_alpha = imA[..., 3:4]
    B_alpha = imB[..., 3:4]
    
    # Compute the composite alpha
    out_alpha = A_alpha + B_alpha*(1 - A_alpha)

    # Compute the composite RGB values
    out_rgb = (imA[..., :3]*A_alpha + imB[..., :3]*B_alpha*(1 - A_alpha)) / out_alpha

    # Concatenate the RGB and alpha channels back
    out_image = np.concatenate((out_rgb, out_alpha), axis=-1)

    # Convert back to 8-bit data
    out_image = (out_image * 255).astype(np.uint8)

    return out_image   



def labelMask2Edges(mask,outwards=False,thickness=2):
    """
    This takes a labelled mask and returns a new mask that is just the edges 
    of that mask. The edges still have the value of the original label.

    Parameters
    ----------
    mask : numpy array
        A labelled mask of dimensions(NY,NX).
            outwards : bool
                Whether to expand the mask outwards to form the edge or inwards 
                (i.e. dilate or erode).        
    thickness : int
        How thick the edges in returned mask will be in pixels.
    """
    d1 = disk(thickness)
    labs = np.unique(mask)[1:]
    out_mask = np.zeros(mask.shape,dtype='uint16')

    if outwards:
        out_mask = dilation(mask,footprint=d1)
        out_mask[mask.astype('bool')] = 0    
        return out_mask
    else:
        for lab in labs:
            lab_mask = np.zeros(mask.shape,dtype='bool')
            lab_mask[mask==lab] = True
            edge_mask = np.logical_and(lab_mask,np.invert(erosion(lab_mask,footprint=d1)))
            out_mask[edge_mask] = lab
    
        return out_mask
    

def csv2dataFrame(csv_path):
    """
    This reads a csv into a pandas DataFrame just like pd.read_csv() but adds 
    things like correctly reading numpy slices.
    """
    df = pd.read_csv(csv_path)

    # sort slice column
    if 'slice' in df.columns:
        sliceReg = re.compile(r'\(slice\((\d+), (\d+), (\w+)\), slice\((\d+), (\d+), (\w+)\)\)')
        extracted_groups = df['slice'].str.extract(sliceReg)
        df['slice'] = extracted_groups.apply(lambda x: (slice(int(x[0]),int(x[1]),None if x[2]=='None' else int(x[2])),slice(int(x[3]),int(x[4]),None if x[5]=='None' else int(x[5]))),axis=1) 

    return df
        

def safelyAddImsUINT16(A,B):
    """
    A and B are 2 numpy arrays of the same shape and both dtype=uint16. This 
    adds them together while taking care not to have integer overflow errors. 
    I.e. if the addition result is greater than the max value for uint16 then 
    it is set to the max value.
    """
    result = A.astype(np.uint32) + B.astype(np.uint32)
    return np.clip(result, 0, 65535).astype(np.uint16)  


def combine_images(out_im,add_im,channel,active_channels=[]):
    """
    This function is designed specifically for use in TData.Plot(). out_im is 
    an image of dimensions (NY,NX,3). add_im is an image of dimensions (NY,NX) 
    that you want to combine into out_im according to a colour specified by 
    channel. When channel is indeed a colour, the images are combined by 
    simple additative blending. Note that we aren't careful about integer 
    overflow or clipping here because that is done in TData.Plot(). 
    
    If channel is an HSV channel it is different because I don't understand 
    image blending in HSV so well. Here are some examples of why it is 
    complicated... 1. You can't simply translate to RGB and add because 
    HSV->RGB is many->one (e.g. V=0 is black in HSV but you need R=G=B=0 for 
    black in RGB). 2. How does the H channel blend? Take the average maybe? 
    3. There isn't obvious way to think about a pure S or V channel, always it 
    has a colour (red if H=0). So a pure S,V channel will alter colour even 
    though we're not thinking about colour there. 4. If you add S and V it 
    tends towards bright saturated colour rather than white. ...so in all 
    there is probably a way to do it but we haven't figured it out so we cheat 
    here. We check which other HSV channels are selected and fill missing ones 
    with 1s for SV and 0s for H.

    Parameters
    ----------

    out_im : numpy array (NY,NX,3)
        add_im will be blended into this image.
    add_im : numpy array (NY,NX)
        Will be blended into out_im
    channel : str
        One of: 'Red', 'Green', 'Blue','Cyan','Magenta','Yellow','Greys','RGB',
        'Hue (HSV)','Saturation (HSV)','Value (HSV)'. Specifies how add_im 
        will be blended.
    active_channels : list of str
        What other channels are also going to be put into out_im. May or may 
        not already be in out_im. Just used for HSV.
    """
    if channel=='Red' or channel=='Yellow' or channel=='Magenta' or channel=='Greys':
        out_im[:,:,0] += add_im
    if channel=='Green' or channel=='Yellow' or channel=='Cyan' or channel=='Greys':
        out_im[:,:,1] += add_im       
    if channel=='Blue' or channel=='Cyan' or channel=='Magenta' or channel=='Greys':
        out_im[:,:,2] += add_im
    if channel=='RGB':
        out_im += add_im
        
    if channel=='Hue (HSV)' or channel=='Saturation (HSV)' or channel=='Value (HSV)':
        out_im = out_im.astype('float32')

    if channel=='Hue (HSV)':
        out_im[:,:,0] = add_im/255
        if not 'Saturation (HSV)' in active_channels:
            out_im[:,:,1] = 1
        if not 'Value (HSV)' in active_channels:
            out_im[:,:,2] = 1 
            
    if channel=='Saturation (HSV)':
        out_im[:,:,1] = add_im/255
        if not 'Value (HSV)' in active_channels:
            out_im[:,:,2] = 1  
        
    if channel=='Value (HSV)':
        out_im[:,:,2] = add_im/255
        if not 'Saturation (HSV)' in active_channels:
            out_im[:,:,1] = 1        
        
    return out_im



def normalise_image(im,perc=1,method='percentile'):
    """
    Normalises im by various methods (see method parameter) so that pixel 
    values lie between 0 and 1.
    
    Parameters
    -----------
    im : numpy array shape (NY,NX) or (NC,NY,NX)
        If 3D then 3rd axes are normalised separately. 
    perc : int or float
        If method=='percentile' then the max value in the normalisation will 
        be the (100-perc)th percentile and the min value will be the percth 
        percentile.
    method : {'percentile','otsu'}
        'percentile' rescales im so the percth and (100-perc)th percentile are 
        0 and 1 and clips anything outsie. 'otsu' first does an otsu threshold 
        and finds what fraction 'frac' of the image is found True - then perc 
        is set to frac/4.
    """
    im = im.astype('float32')
    expanded = False
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=0)
        expanded = True
    for c in range(im.shape[0]):
        perc_c = perc
        if method=='otsu':
            thr = filters.threshold_otsu(im[c,:,:])
            mask = (im[c,:,:] > thr) 
            perc_c = (np.sum(mask)/(mask.shape[0]*mask.shape[1]))/4    

        low1 = np.percentile(im[c,:,:],perc_c)
        high1 = np.percentile(im[c,:,:],100 - (perc_c))
        im[c,:,:] = exposure.rescale_intensity(im[c,:,:],
                                           in_range=(low1,high1),
                                           out_range=(0,1))
    if expanded:
        im = im[0,:,:]

    return im


def align_nuc_cyto_labels(nuc_seg,cyto_seg):
    """
    Give this two labelled segmentations, one for nuclei and one for 
    cytoplasm and this returns a new version of each so that the labels 
    are matching. The cytoplasm label is changed so that it has the same 
    label as nuclei that overlaps with it the most. Cytoplasm masks with 
    no overlapping nuclei are deleted, as are nuclei that aren't assigned 
    to a cytoplasm. This is currently a basic method, errors may occur 
    where nuclei overlap multiple cytoplasms.

    Parameters
    ----------
    nuc_seg,cyto_seg : numpy array (NY,NX)
        Labelled segmentations of nuclei and cytoplasms.
    """

    # these will be the new segmentations with matching labels
    nuc_seg = nuc_seg.copy()
    cyto_seg = cyto_seg.copy()
    
    new_nuc_seg = np.zeros_like(nuc_seg)
    new_cyto_seg = np.zeros_like(cyto_seg)
    
    for c_lab in np.unique(cyto_seg)[1:]:
        
        cyto_lab_im = (cyto_seg == c_lab)
        nuc_labs = nuc_seg[cyto_lab_im]
        nuc_labs = nuc_labs[nuc_labs!=0]
        
        if len(nuc_labs)!=0:
            n_lab = stats.mode(nuc_labs,keepdims=False)[0]
            nuc_lab_im = (nuc_seg == n_lab)
            new_nuc_seg[nuc_lab_im] = n_lab
            new_cyto_seg[cyto_lab_im] = n_lab
    
            # delete that nuclei in the original segmentation so it 
            # can't be assigned again
            nuc_seg[nuc_seg==n_lab] = 0

    return (new_nuc_seg,new_cyto_seg)


def subtract_masks(mask1,mask_sub):
    """
    Give this two labelled segmentations and this returns a new version with 
    the regions of the second removed in the corresponding regions of the 
    first. Note that if e.g. label 2 in mask_sub is overlapping label 1 in 
    mask1 it will not remove this overlap section - only corresponding labels.    

    Parameters
    ----------
    mask1,mask_sub : numpy array (NY,NX)
        Labelled segmentations of regions.
    """

    # these will be the new segmentations with matching labels
    mask1 = mask1.copy()
    mask1[mask1==mask_sub] = 0
    
    return mask1
    

def can_write_to_folder(folder_path):
    """
    Checks in advance whether you will be able to write to a directory with 
    the provided path by checking it exists and that you have access.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return False
    # Check if the folder is writable
    if not os.access(folder_path, os.W_OK):
        return False
    return True


def isLabelledMaskQ(seg):
    """
    This sort of decides whether or not a segmentation is labelled. More 
    precisely it decides whether or not you can safely relabel a segmentation 
    without ruining any existing labelling (but reverse of this, because it 
    returns False if it is safe to relabel).

    It doesn't yet check that pixels of same value are indeed connected, i.e. 
    it is just checking it doesn't look like a boolean (either True/False or 
    1/0).

    Parameters
    ----------
    seg : 2D numpy array
        The segmentation that you want to decide whether or not it is labelled.
    
    """
    if seg.dtype=='bool':
        return False
    
    the_unique = np.unique(seg)

    if len(the_unique)==1:
        return False

    if len(the_unique)==2 and 1 in the_unique:
        return False

    return True



def label2rgb2(array):
    """
    I think skimage.color.label2rgb() doesn't have a constant mapping between 
    int and rbg colour, e.g. a pixel/label value 5 can be converted to 
    different RGB. I think probably it depends on the set of pixel values that 
    exist in the array, i.e. if the pixel values in one array are 1,2,3,4 and 
    in another they are 4,8,16,20, - the 4 will be converted to a different 
    colour in each. 

    In this function the colour of pixel value v is given by (v//9)+1 so the 
    mapping is always constant.
    """

    colour_array = np.array([[0.0,0.0,0.0],
                              [0.0,0.0,1.0],
                              [0.0,0.502,0.0],
                              [0.0,1.0,1.0],
                              [0.294,0.0,0.51],
                              [0.604,0.804,0.196],
                              [1.0,0.0,0.0],
                              [1.0,0.0,1.0],
                              [1.0,0.549,0.0],
                              [1.0,1.0,0.0],])

    n_colours = 9

    array_L = np.zeros_like(array)
    array_L = np.mod(array,n_colours) + 1
    array_L[array==0] = 0
    
    return colour_array[array_L]
