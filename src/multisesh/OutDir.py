import os
import re
import numpy as np
from skimage.transform import downscale_local_mean

import tifffile

from . import generalFunctions as genF
from . import errorMessages as EM

class OutDir:
    """ This class defines a directory structure containing image files that 
    have been output by xfold so that you can do so further basic processing.
    
    Attributes
    ----------
    outDir : str
        The path of the main directory.
    name : str
        Just the name of the directory (i.e. not the whole path).
    pDir : str
        The path of the parent directory of the main directory.
    fDirs : list of str
        List of paths to each sub-directory to be analysed, each of which 
        representing one field.
    fNames : list
        Same as fDirs but just the names of the directories.
    
    Notes
    -----
    The directory you give must be a directory with directories inside, 
    one for each field.
    """
    def __init__(self,outDirPath,fieldNameList=None,excludeFieldNameList=None):
        """
        Parameters
        ----------
        outDirPath : str
            The path of the directory.
        fieldNameList : str or list or None
            If None then fDirs will be a list of all directories in outDir. 
            Ones beginning with '.' are filterd out.
            If list then that list should contain all directory names that you 
            want to include to build fDir and only those will be included.
            If str then fDirs will be a list with that single string in.
        excludeFieldNameList : list of str
            Removes any of these paths from the fDirs that have been made.
        """
        
        self.outDir = outDirPath
        assert os.path.exists(self.outDir),EM.od5
        assert not os.path.split(self.outDir)[1] == '',EM.od6
        
        self.name = os.path.split(self.outDir)[1]
        self.pDir = os.path.split(self.outDir)[0]
        
        # set up field directory list. Default is all dirs in outDir. 
        # allow single string input. Don't allow entries that aren't real dirs!
        self.fDirs = fieldNameList
        if not self.fDirs:
            fd = os.listdir(self.outDir)
            fd = [d for d in fd if os.path.isdir(os.path.join(self.outDir,d))]
            self.fDirs = [d for d in fd if d[0]!='.']
        if isinstance(self.fDirs,str):
            self.fDirs = [self.fDirs]
        self.fDirs = [os.path.join(self.outDir,fl) for fl in self.fDirs]

        for fl in self.fDirs:
            assert os.path.exists(fl),EM.od7
        
        self.xfDirs = excludeFieldNameList
        if not self.xfDirs:
            self.xfDirs = []
        self.xfDirs = [os.path.join(self.outDir,fl) for fl in self.xfDirs]
        self.fDirs = [fl for fl in self.fDirs if fl not in self.xfDirs]
        
        # let's be restrictive no point in creating empty OutDir
        assert self.fDirs != [],EM.od8
        
        # very restrictive: don't allow OutDirs with empty field dirs
        # this saves us time in function writing
        for fd in self.fDirs:
            fs = [f for f in os.listdir(fd) if f[-4:]=='.tif']
            assert fs!=[],EM.od9
        
        self.fNames = [os.path.split(fl)[1] for fl in self.fDirs]
        
        
    def Concatenate(self,newDirName=None,CancelIntensityJumps=False,
                    threshDiff=0.25,autoscale=True):
        """ 
        This concatenates all the .tif files in each of your OutDir's field
        directories. It saves the output in a directory that you provide the 
        name of (which will be created in the parent of the OutDir).
        
        It creates a new directory name if you don't provide one. If you 
        provide one that already exists it is ok unless there are field 
        directories in your OutDir that match the ones in the pre-existing 
        new directory. I.e. it doesn't fail unless you try to overwrite any 
        field directories.
        """
        # make directory paths for output
        if not newDirName:
            newDirName = 'Concat_'+self.name
        newDir = os.path.join(self.pDir,newDirName)
        newFDirs = [os.path.join(newDir,fn) for fn in self.fNames]
        
        # create that directory and the field directories
        for nd in newFDirs:
            assert not os.path.exists(nd),EM.od1
        for nd in newFDirs:
            os.makedirs(nd)
        
        # do the concatenations
        for fdi,fdo in zip(self.fDirs,newFDirs):
            fs = [fp for fp in os.listdir(fdi) if fp[-4:]=='.tif']
            # need to sort according to s0000-tag first, then alphabetical
            fs2 = [re.search(r'_(s\d\d\d\d)',f).group(1) for f in fs]
            filePs = [s[1] for s in sorted(zip(fs2,fs))]
            outP = os.path.join(fdo,filePs[0])
            filePs = [os.path.join(fdi,fp) for fp in filePs]
            dims = []
            with tifffile.TiffFile(filePs[0]) as tif:
                allData = tif.asarray()
                dim = genF.tif2Dims(tif)
                allData = allData.reshape(tuple(dim))
                dims.append(dim)
            # concatenate the rest:
            if len(filePs)>1:
                for tp in filePs[1:]:
                    with tifffile.TiffFile(tp) as tif:
                        _data = tif.asarray()
                        dim = genF.tif2Dims(tif)
                        if dim[1:]!=dims[0][1:]:
                            raise Exception(EM.od4)
                        _data = _data.reshape(tuple(dim))
                        allData = np.concatenate((allData,_data))
                        dims.append(dim)            
                
            if CancelIntensityJumps:
                nt = sum([d[0] for d in dims])
                nc = dims[0][4]
                totals = [np.mean(allData[0,:,:,:,c]) for c in range(nc)]
                for t in range(nt-1):
                    for c in range(nc):
                        tot = np.mean(allData[t+1,:,:,:,c])
                        if tot==0 or totals[c]==0:
                            totals[c] = tot
                        else:
                            frac = tot / totals[c]
                            if frac > 1 + threshDiff or frac < 1 - threshDiff:
                                allData[t+1,:,:,:,c]=allData[t+1,:,:,:,c]/frac
                            else:
                                totals[c] = tot

            # try to find channel names in the file...
            # try to find from tw_metadata
            chanList = []
            for tp in filePs:
                with tifffile.TiffFile(tp) as tif:
                    d = dir(tif)                 
                    # for image j ready files that we saved:
                    if ('imagej_metadata' in d and 
                            tif.imagej_metadata != None and 
                            'tw_chan' in tif.imagej_metadata.keys()):
                        chanList.append(tif.imagej_metadata['tw_chan'])
            
            # set chan according to those results:
            if len(chanList)==len(filePs) and len(set(chanList))==1:
                chan = genF.listStr2List(chanList[0])
            else:
                raise Exception(EM.od3)
            
            seshDs = ['Multiple sessions' for i in range(5)]
            genF.saveTiffForIJ(outP,allData,chan,seshDs,autoscale=autoscale)
    
    
    def Downsize(self,downsize,newDirName=None,overwrite=False):
        """
        Simple method to downsize all the contents of an OutDir.
        """
        # make directory paths for output directories
        if not newDirName:
            newDirName = 'Downsize_'+self.name
        newDir = os.path.join(self.pDir,newDirName)
        newFDirs = [os.path.join(newDir,fn) for fn in self.fNames]
                
        # create that directory and the field directories
        if not overwrite:
            for nd in newFDirs:
                assert not os.path.exists(nd),EM.od1
        for nd in newFDirs:
            if not os.path.exists(nd):
                os.makedirs(nd)        

        # do the downsizing
        for fdi,fdo in zip(self.fDirs,newFDirs):
            fps = [os.path.join(fdi,f) for f in os.listdir(fdi) if '.tif' in f]
            ops = [os.path.join(fdo,f) for f in os.listdir(fdi) if '.tif' in f]
            for f,g in zip(fps,ops):
                with tifffile.TiffFile(f) as tif:
                    data = tif.asarray()
                    d = dir(tif)                 
                    # find the chan so we can save in metadata
                    if ('imagej_metadata' in d and 
                            tif.imagej_metadata != None and 
                            'tw_chan' in tif.imagej_metadata.keys()):
                        chan = tif.imagej_metadata['tw_chan']
                        chan = genF.listStr2List(chan)
                    elif 'LUTs' in tif.imagej_metadata.keys():
                        chan = genF.LUTInterpreter(tif.imagej_metadata['LUTs'])
                    else:
                        chan = 'Chan_unknown'
                    dim = genF.tif2Dims(tif)
                    if chan=='Chan_unknown':
                        chan = ['Chan_unknown' for c in range(dim[4])]
                data = data.reshape(tuple(dim))
                data = downscale_local_mean(data,(1,1,1,1,1,downsize,downsize))
                data = data.astype('uint16')
                seshDs = ['Multiple sessions' for i in range(5)]
                genF.saveTiffForIJ(g,data,chan,seshDs,overwrite=overwrite)
                
                
    def Clip(self,maxV,newDirName=None,overwrite=False):
        """
        very simple method to just clip pixel intensities according to max.
        Needed because ExtractProfileRectangle needs the mask outline you draw 
        to be the maximum pixel value in the image and that can be annoying 
        for images with overexposed regions or random high pixels.
        """
        # make directory paths for output directories
        if not newDirName:
            newDirName = 'Clip_'+self.name
        newDir = os.path.join(self.pDir,newDirName)
        newFDirs = [os.path.join(newDir,fn) for fn in self.fNames]
        
        # create that directory and the field directories
        if not overwrite:
            for nd in newFDirs:
                assert not os.path.exists(nd),EM.od1
        for nd in newFDirs:
            if not os.path.exists(nd):
                os.makedirs(nd)
                
        # do the clippings
        for fdi,fdo in zip(self.fDirs,newFDirs):
            fps = [os.path.join(fdi,f) for f in os.listdir(fdi) if '.tif' in f]
            ops = [os.path.join(fdo,f) for f in os.listdir(fdi) if '.tif' in f]
            for f,g in zip(fps,ops):
                with tifffile.TiffFile(f) as tif:
                    data = tif.asarray()
                    d = dir(tif)                 
                    # find the chan so we can save in metadata
                    if ('imagej_metadata' in d and 
                            tif.imagej_metadata != None and 
                            'tw_chan' in tif.imagej_metadata.keys()):
                        chan = tif.imagej_metadata['tw_chan']
                        chan = genF.listStr2List(chan)
                    elif 'LUTs' in tif.imagej_metadata.keys():
                        chan = genF.LUTInterpreter(tif.imagej_metadata['LUTs'])
                    else:
                        chan = 'Chan_unknown'
                    dim = genF.tif2Dims(tif)
                    if chan=='Chan_unknown':
                        chan = ['Chan_unknown' for c in range(dim[4])]
                data = data.reshape(tuple(dim))
                data[data>maxV] = maxV
                seshDs = ['Multiple sessions' for i in range(5)]
                genF.saveTiffForIJ(g,data,chan,seshDs,overwrite=overwrite)
                
                
    def AutoLevel(self,newDirName=None,minP=2,maxP=98,overwrite=False):
        """For each field, this looks at all files and timepoints and applies 
        autolevelling. That is, all pixels below/above the minP/maxP 
        percentile pixel value are set to 0/max value (where max is 
        the max value for the data type) and all pixel values in between 
        scale linearly. 
        
        If overwrite is True AND newDirName=None, it directly overwrites the 
        files as they are in the field directories. If newDirName is not None 
        it will raise an exception because we don't want to be able to 
        overwrite files in an external file. 
        
        If overwrite=False and newDirName=None then it makes up it's own name.
        
        """
        
        # only allowed to overwrite the files of OutDir, nothing external
        assert not (overwrite and newDirName),EM.od2
        
        # make newDirName and paths
        if not newDirName:
            if overwrite:
                newDirName = self.name
            else: 
                newDirName = 'AutoLev_'+self.name
        newDir = os.path.join(self.pDir,newDirName)
        newFDirs = [os.path.join(newDir,fn) for fn in self.fNames]
        
        # make directories or raise exception if they exist already
        if not overwrite:
            for nd in newFDirs:
                assert not os.path.exists(nd),EM.od1
            for nd in newFDirs:
                os.path.mkdirs(nd)         
        
        for fdi,fdo in zip(self.fDirs,newFDirs):
            fps = [fp for fp in os.listdir(fdi) if fp[-4:]=='.tif']
            # this is going to be 2 loops: 
            # 1 to concatenate everything and find global autolevel values 
            # 2 open and take data and autolevel it and save it
            
            with tifffile.TiffFile(fps[0]) as tif:
                NC = genF.tif2Dims(tif)[4]
            Cs = [np.zeros(0) for c in range(NC)]
            for fp in fps:
                with tifffile.TiffFile(fp) as tif:
                    _d = tif.asarray()
                Cs = [np.hstack(a,np.ravel(_d[:,:,:,:,i])) 
                      for i,a in enumerate(Cs)]
            minVs = [np.percentile(c,minP) for c in Cs]
            maxVs = [np.percentile(c,maxP) for c in Cs]
            # now just do the transposition....
            
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
        
                    