import numpy as np
import cv2 as cv
import math
from skimage.morphology import disk
from skimage.filters import sobel
from scipy.ndimage import generic_filter,gaussian_filter
from skimage.transform import downscale_local_mean,resize


def maxProj(stack):
    """Normal maximum projection."""
    return np.max(stack,axis=0)

def avProj(stack):
    """Normal average projection."""
    return np.mean(stack,axis=0)

def minProj(stack):
    """Normal minimum projection."""
    return np.min(stack,axis=0)

def signalF(vals):
    """A measure of signal."""
    if len(vals.shape)==2:
        return vals.mean()*vals.std()
    elif len(vals.shape)==3:
        return vals.mean(axis=(1,2))*vals.std(axis=(1,2))
    else:
        raise Exception('bad array shape given to signalF')

def signalF_sobel(arr):
    """A measure of signal based on sobel."""    
    if len(arr.shape)==2:
        return np.sum(sobel(gaussian_filter(arr,1)))
    elif len(arr.shape)==3:
        result = np.zeros(arr.shape,dtype='float64')
        for i in range(arr.shape[0]):
            result[i] = sobel(gaussian_filter(arr[i], sigma=1))
        return np.sum(result,axis=(1,2))
    else:
        raise Exception('bad shape of arr in signalF_sobel')
        

def tenengrad(image, ksize=3, mask=False,closeSize=40):
    """
    The tenegrad is another sobel based sharpness measure. Here we use cv 
    though because faster and can control kernel size. We allow masking of 
    big bright objects by user supplied mask or by multiplicative factor 
    (see parameters).

    !!This method only works for sliceBySlice signal quantification. By 
    definition it can't work for pixel-by-pixel slice selection because it's 
    made for cases where you want to mask a bad bit of the image!!!

    Parameters
    ----------
    image : 2D or 3D numpy array
        The image(s).
    ksize : int
        The size of the sobel kernel
    mask : np.array or float or int or False
        If supplied then this mask is used to omit the sobel values in final 
        caluculation of mean sobel. If a float or int is given then the median 
        pixel value is calculated and median*mask is taken as the threshold 
        value for a new mask (which is then morphologically closed and eroded 
        to make final mask).
    closeSize : int
        The size of the kernel when you are morphologically closing the mask 
        that you have calculated from a float/int multiplicative factor. 
        Having it about 1/3 the size of features (e.g. nuclei) is about right.
    """
    if image.ndim==2:
        image = image[np.newaxis].copy()

    measures = []
    for i in range(image.shape[0]):
        gx = cv.Sobel(image[i], cv.CV_64F, 1, 0, ksize=ksize)
        gy = cv.Sobel(image[i], cv.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        if isinstance(mask,np.ndarray):
            measures.append(np.mean(magnitude[mask]))
        elif isinstance(mask,float) or isinstance(mask,int):
            mask_im = np.ones_like(image[i])
            mask_im[image[i]>(np.median(image[i])*mask)] = 0
            mask_im = mask_im.astype('uint8')
            footprint = np.ones((closeSize,closeSize),np.uint8)
            mask_im = cv.morphologyEx(mask_im,cv.MORPH_CLOSE,footprint)
            mask_im = cv.erode(mask_im,footprint,iterations = 1)    
            measures.append(np.mean(magnitude[mask_im.astype('bool')]))
        else:
            measures.append(np.mean(magnitude))
    return np.array(measures)


def LapVar(image, 
            ksize=11,
            normalise=False):
    """
    Another common sharpness measure. 
    
    Parameters
    ----------
    image : 2D or 3D numpy array
        The image(s).
    ksize : int
        The size of the sobel kernel
    normalise : False
        You may or may not want intensity to contribute to the focus measure. 
        If True, then each slice will first be normalised to itself so that 
        intensity doesn't contribute.
    """
    if image.ndim==2:
        image = image[np.newaxis].copy()

    measures = []
    for i,im in enumerate(image):
        if normalise:
            im = im/np.max(im)  
        laplacian = cv.Laplacian(im,cv.CV_64F,ksize=ksize)
        measures.append(laplacian.var())
        
    return np.array(measures)      

    measures = []
    for i in range(image.shape[0]):
        gx = cv.Sobel(image[i], cv.CV_64F, 1, 0, ksize=ksize)
        gy = cv.Sobel(image[i], cv.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx**2 + gy**2)

    return np.array(measures)    


def signalProj(stack,
               pixelSize,
               dscale=1,
               slices=1,
               proj=True,
               furthest=False,
               sliceBySlice=False,
               meth_fun=signalF,
               **kwargs):
    """
    A 2D projection which applies a filter to select the slice of most signal 
    for each pixel. The measure of signal is mean * std. 
    
    Parameters
    ----------
    stack : numpy array
        The image stack. (NZ,YSIZE,XSIZE)
    pixelSize : float
        The size in um of the pixel in the image. The structure element of the 
        filter is a circle of 20um because that is a bit bigger than a nucleus.
    dscale : int
        The factor by which to dscale the images during analysis to make 
        analysis faster.
    slices : int
        The number of slices to include in the output. These are taken 
        alternatingly from above-below the highest signal slice. NA is put 
        in if the number goes beyond the slices of the image stack.
    proj : bool
        Whether to do a mean projection of the final array.
    furthest : bool
        Whether to simply return the pixels (currently only does it with 1 
        slice) furthest from the signal. I.e. can be used to return a 
        background image.
    sliceBySlice : bool
        If True then the same z-slice is chosen across all x-y. I.e. we just 
        pick the slice with most signal rather than the filter method that 
        picks the best z for each pixel based on its surroundings.
    meth_fun : python function
        The function to use for the signal sharpness quantification.
    **kwargs : variable
        The arguments to be passed to meth_fun
    """
    NZ,YSIZE,XSIZE = np.shape(stack)  

    if sliceBySlice:
        sigs = meth_fun(stack,**kwargs)
        slice1 = np.argmax(sigs)
        
        if furthest:  
            if slice1>=NZ/2:
                slice1 = 0
            else:
                slice1 = NZ-1
            
        if slices==1:
            return stack[slice1]
        else:
            NAN = np.empty((YSIZE,XSIZE))
            NAN.fill(np.nan)            
            stack = np.concatenate((stack,NAN[np.newaxis]))
            out = np.zeros((slices,YSIZE,XSIZE))
            for s in range(slices):
                ii = ((s+1)//2)*((-1)**s)   
                sel = slice1 + ii
                sel[sel<0] = NZ
                sel[sel>=NZ] = NZ
                sel = sel.astype(int) 
                out[ii+(slices//2)] = stack[sel]                
                
            if proj:
                return np.nanmean(out,axis=0)
            else:
                return out

    assert isinstance(dscale,int),'dscale must be an int'
    assert isinstance(slices,int),'slices must be an int'
    
    # radius wants to be about 20um so it is definitely bigger than nuclei
    radius = np.ceil(20/(pixelSize*dscale))
    selem = disk(radius)
    
    #initiate mask:
    stack2 = np.zeros((NZ,math.ceil(YSIZE/dscale),math.ceil(XSIZE/dscale)))
    for i,im in enumerate(stack):
        stack2[i] = generic_filter(downscale_local_mean(im,(dscale,dscale)),
                                   meth_fun,
                                   **kwargs,
                                   footprint=selem,
                                   mode='reflect')
    stack2 = np.argmax(stack2,axis=0)
    stack2 = resize(stack2,(YSIZE,XSIZE),preserve_range=True)
    stack2 = np.rint(stack2).astype(int)
    I,J = np.ogrid[:YSIZE,:XSIZE]      
    
    if furthest:
        stack2 = np.where(stack2>=NZ/2,0,NZ-1)
        assert slices==1,'slices must be 1 when using furthest'
    
    if slices==1:
        return stack[stack2,I,J]
    else:        
        #out = stack[stack2,I,J][np.newaxis]
        out = np.zeros((slices,YSIZE,XSIZE))        
        NAN = np.empty((YSIZE,XSIZE))
        NAN.fill(np.nan)
        stack = np.concatenate((stack,NAN[np.newaxis]))
        for s in range(slices):
            ii = ((s+1)//2)*((-1)**s)
            sel = stack2 + ii
            sel[sel<0] = NZ
            sel[sel>=NZ] = NZ
            sel = sel.astype(int)
            #out = np.concatenate((out,stack[sel,I,J][np.newaxis]))
            out[ii+(slices//2)] = stack[sel,I,J]
        if proj:
            return np.nanmean(out,axis=0)
        else:
            return out

        
def findSliceSelection(stack,
                       pixelSize,
                       dscale=1,
                       furthest=False,
                       sliceBySlice=False,
                       meth_fun=signalF,
                       **kwargs):
    """
    This is just the first part of signalProj, where you choose which slices 
    will form the projection.
    """
    assert isinstance(dscale,int),'dscale must be an int'
    NZ,YSIZE,XSIZE = np.shape(stack)

    if sliceBySlice:
        sigs = meth_fun(stack,**kwargs)
        stack2 = np.argmax(sigs)
        
        if furthest:  
            if slice>=NZ/2:
                stack2 = 0
            else:
                stack2 = NZ-1
        return int(stack2)
    
    # radius wants to be about 30um so it is definitely bigger than nuclei
    radius = np.ceil(20/(pixelSize*dscale))
    selem = disk(radius)
    
    #initiate mask:
    stack2 = np.zeros((NZ,math.ceil(YSIZE/dscale),math.ceil(XSIZE/dscale)))
    for i,im in enumerate(stack):
        stack2[i] = generic_filter(downscale_local_mean(im,(dscale,dscale)),
                                   meth_fun,
                                   **kwargs,
                                   footprint=selem,
                                   mode='reflect')
    stack2 = np.argmax(stack2,axis=0)
    stack2 = resize(stack2,(YSIZE,XSIZE),preserve_range=True)
    stack2 = np.rint(stack2).astype(int)
    
    if furthest:
        stack2 = np.where(stack2>=NZ/2,0,NZ-1)
        
    return stack2
        
    
        
def takeSlicesSelection(selection_stack,
                        data_stack,
                        slices=1,
                        proj=True,
                        sliceBySlice=False):
    """
    This is just the last part of signalProj, you have already found which 
    slices have the signal you want and now just take them from your stack.

    Parameters
    ----------
    selection_stack : numpy array (1,NY,NX) or int
        This is the stack giving which z-slices to select. For sliceBySlice it 
        should be int.
    data_stack : 3D numpy array shape (NZ,NY,NX)
       This is your data that you're selecting slices from. 
    slices : int
        The number of slices to include in the output. These are taken 
        alternatingly from above-below the highest signal slice. NA is put 
        in if the number goes beyond the slices of the image stack.
    proj : bool
        Whether to do a mean projection of the final array.        
    sliceBySlice : bool
        If True then the same z-slice is chosen across all x-y. I.e. we just 
        pick the slice with most signal rather than the filter method that 
        picks the best z for each pixel based on its surroundings.        
    """
    NZ,YSIZE,XSIZE = np.shape(data_stack)
    
    if sliceBySlice:
        if slices==1:
            return data_stack[selection_stack]
        else:
            NAN = np.empty((YSIZE,XSIZE))
            NAN.fill(np.nan)            
            data_stack = np.concatenate((data_stack,NAN[np.newaxis]))
            out = np.zeros((slices,YSIZE,XSIZE))
            for s in range(slices):
                ii = ((s+1)//2)*((-1)**s)   
                sel = selection_stack + ii
                sel[sel<0] = NZ
                sel[sel>=NZ] = NZ
                sel = sel.astype(int) 
                out[ii+(slices//2)] = data_stack[sel]                
                
            if proj:
                return np.nanmean(out,axis=0)
            else:
                return out
    
    I,J = np.ogrid[:YSIZE,:XSIZE]
        
    if slices==1:
        return data_stack[selection_stack,I,J]
    else:
        out = data_stack[selection_stack,I,J][np.newaxis]
        NAN = np.empty((YSIZE,XSIZE))
        NAN.fill(np.nan)
        data_stack = np.concatenate((data_stack,NAN[np.newaxis]))
        for s in range(slices-1):
            sel = selection_stack + ((s+1)//2)*((-1)**s)
            sel[sel<0] = NZ
            sel[sel>=NZ] = NZ
            sel = sel.astype(int)
            out = np.concatenate((out,data_stack[sel,I,J][np.newaxis]))
        if proj:
            return np.nanmean(out,axis=0)
        else:
            print(out.shape)
            return out        



def selectSlices_byMeanSTD_takeN(stack,N):
    """
    This selects N adjacent slices to take out. 
    Selection is based on mean*STD signal measure. 
    It does a rolling average and selects N slices with min rolling average.
    """
    #this is the signal measure, smaller for more signal:
    measures = [MAXSIGNALMEASURE/(im.mean()*im.std()) for im in stack]
    measures = np.convolve(measures, np.ones((N,))/N, mode='valid')
    
    # find min position and return the appropriate slice of stack
    minIndex = np.argmin(measures)  
    return stack[minIndex:minIndex + N]



def selectSlices_byMeanSTD_thresh(stack,thresh):
    """Returns any slices where the measure is below a threshold."""
    # just do list comprehension filtering:    
    return [im for im in stack if MAXSIGNALMEASURE/(im.mean()*im.std()) < thresh]


def sectioniseStack(stack,N,NZ):
    """Divides a stack into N*N smaller stacks."""
    # first divide it up
    # look how np.array_split returns a list not a numpy array! (I guess because it can be jagged)
    sections = []
    for im in stack:
        sections.append(np.array_split(im,N))
        for n in range(N):
            sections[-1][n] = np.array_split(sections[-1][n],N,axis=1)
   
    # now need to reshape it so that each element of the list is a section 
    # with all z-slices
    # i.e. the thing returned is N*N long
    resections = [ [] for i in range(N*N) ]
    for n in range(N*N):
        for z in range(NZ):
            resections[n].append(sections[z][n%N][n//N])
    return resections



def reassembleSections(imageList):
    """
    Takes a list of images which is N*N long and together form a N*N square 
    image puts it back together watch out it doesn't work with a list of N*N 
    image stacks! They have to be single images.
    """
    N = int(math.sqrt(len(imageList)))
    columns = []
    for n in range(N):
        columns.append(np.concatenate(imageList[n*N:(n+1)*N]))
    image = np.concatenate(columns,axis=1)
    return image