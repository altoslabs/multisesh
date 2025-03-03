import numpy as np
from skimage.filters import sobel
from skimage.filters import gaussian
# 12/6/23 this line not working:
#from skimage._shared.fft import fftmodule as fft
#so changing to this: (hoping numpy is identical to old skimage one, have written warning)
from numpy import fft as fft
import cv2 as cv
from skimage import filters,measure


def findCentres(ims,NX,imageOverlap,method,parameters,printWarnings=True):
    """
    Aligns montage tiles using cross-correlation, returning the centres of the 
    tiles within the aligned full image.
    
    Parameters
    ----------
    ims : numpy array
        The images to be aligned, dimensions (NM,NC,NY,NX).
    NX : int
        Number of tiles in x dimension.
    imageOverlap : float
        The tile overlap set in the imaging software.
    
    Returns
    --------
    cenList_M : tuple
        All the indices of the centre pixel of each image.
    
    Notes
    ------
    In the code _M means the contents are in the final montage's coordinate 
    system.
    It assumes all images are the same size.
    
    The parameters in the code are in the form: 
    [threshold,ampPower,maxShiftFraction,boxsize,sdt]
    
    threshold - threshold for 'enough signal', see lower functions
    ampPower - image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShiftFraction - maximum detected that is applied, as a fraction of 
                    image size b/c if a big shift is detected it is probably
                    wrong if detected shift is bigger it does 'auto' aligning
    boxsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring
    minSize - for images smaller than this it won't do cross-correlation
                alignment because the small size risks stupid results.
    """
    
    # unpack parameters
    # they get packaged again right away b/c they're only used in lower
    # functions but I unpackage them just so you can see them
    thresh = parameters[0]
    aPower = parameters[1]
    maxShift = parameters[2]
    bsize = parameters[3]
    sdt = parameters[4]
    minSize = parameters[5]
    cs = parameters[6]
    
    ysize,xsize = np.shape(ims[0,0])
    ycen = np.floor((ysize-1)/2).astype(int)
    xcen = np.floor((xsize-1)/2).astype(int)
    
    # package these parameters together so they don't bloat the script
    pars = (ysize,xsize,imageOverlap,method,thresh,
            aPower,maxShift,bsize,sdt,minSize)
    
    # initiate the main list we want to return
    cenList_M = []
    
    # main loop over all images:
    for j,im in enumerate(ims):
        # first image has centre local centre
        if j == 0:
            cenList_M.append(np.array([ycen,xcen]))
        # now do the top row where only the sides are compared:
        elif j < NX:
            cenList_M.append(findCen_S(ims[j-1],cenList_M[j-1],im,pars,cs))
        # then if it is the first colomn of a row you only compare the top:
        elif j % NX == 0:  
            cenList_M.append(findCen_T(ims[j-NX],cenList_M[j-NX],im,pars,cs,
                                       printWarnings=printWarnings))
        # or general case you compare the top and side:
        else:
            cenList_M.append(findCen_TS(ims[j-1],cenList_M[j-1],ims[j-NX],
                                        cenList_M[j-NX],im,pars,cs))
        
    return tuple(cenList_M)


def findCen_S(imageL,centreL_M,imageR,parameters,counts):
    """This function finds the centre of an imageR, knowing that it should sit 
    directly to the right of imageL and not aligning with any other tiles, 
    i.e. it is for the top row of montages, i.e. the _S stands for 'side only'
    imageL has centre pixel index given by centreL.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring
    """
    
    extra = 1.5
    
    # unpack parameters from parameters:
    ysize,xsize = [parameters[0],parameters[1]]
    imageOverlap = parameters[2]
    method = parameters[3]
    thresh = parameters[4]
    aPow = parameters[5]
    maxShiftFrac = parameters[6]
    minSize = parameters[9]
    
    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = parameters[7]
    sdt = parameters[8]
    pars = (boxsize,sdt)
    
    # initiate centreR so image R is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreR_M = np.array([centreL_M[0], xsize + centreL_M[1]])    
    
    # the xsize of the section we will compare, 'extra' fraction bigger than 
    # the suggested overlap 
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the y, and a section of the x
    testxSize = int(np.floor(extra*imageOverlap*xsize))
    
    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secL = imageL[:,:,xsize - testxSize:].copy()
    secR = imageR[:,:,:testxSize].copy()
    
    # calculate sobel signal in both sections, in all channels
    sigL = np.array([measureSig(secL[c],pars) for c in range(len(secL))])
    sigR = np.array([measureSig(secR[c],pars) for c in range(len(secR))])
    # test whether those signals are big enough and combine to make one 
    # channel selection: 
    chan = (sigL > thresh)*(sigR > thresh)
    
    # remove bad channels from signal b/c we use is as a multiplier next":
    sigL = sigL[chan]
    sigR = sigR[chan]
    
    # now delete channels which don't have enough signal in them  
    # amplify the channels by their signal measure so that the ones 
    # with signal are favoured during cross-correlation:
    secL = np.swapaxes(np.swapaxes(secL[chan],0,2)*sigL**aPow,0,2)
    secR = np.swapaxes(np.swapaxes(secR[chan],0,2)*sigR**aPow,0,2)
    
    # test if the sections contain signal, if not then just set the shift 
    # to what you'd expect from overlap:
    # also don't bother aligning if the images are small, you'll just 
    # get errors
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift = np.array([0,int(xsize*imageOverlap)])
    elif any(chan)==False:
        shift = np.array([0,int(xsize*imageOverlap)])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secL,secR)
        shift = shift.astype(int)
        
        #if shifts are too big then ignore them and set default:
        if abs(shift[0]) > maxShiftFrac*ysize:
            shift[0] = 0
            counts[1] += 1
        else:
            # the shift in y doesn't depend on the section you took:
            shift[0] = -shift[0]
        # extra*xsize is the expected shift (equates to matching overlap) so 
        # shift[1]-extra*xsize compares fairly left and right shifts:
        if abs(shift[1]-(extra-1)*testxSize) > maxShiftFrac*xsize:
            shift[1] = int(xsize*imageOverlap)
            counts[1] += 1
        else:        
            # the x shift of the actual image depends on the section you took!
            shift[1] = testxSize - shift[1] 
            
    return centreR_M - shift



def findCen_T(imageT,centreT_M,imageB,parameters,counts,printWarnings=True):
    """
    This function finds the centre of an imageB, knowing that it should sit 
    directly below of imageT and not aligning with any other tiles,
    i.e. it is for 2nd from top, far left tile only, _T stands for top.
    imageT has centre pixel index given by centreT.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.    
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring    
    """    
    
    extra = 1.5
    
    # unpack parameters from parameters:
    ysize,xsize = [parameters[0],parameters[1]]
    imageOverlap = parameters[2]
    method = parameters[3]
    thresh = parameters[4]
    aPow = parameters[5]
    maxShiftFrac = parameters[6]
    minSize = parameters[9]

    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = parameters[7]
    sdt = parameters[8]
    pars = (boxsize,sdt)      
    
    # initiate centreB_M so imageB is directly against imageT (i.e. no overlap) 
    # this way we will just add the shift to this centre later
    centreB_M = np.array([centreT_M[0] + ysize, centreT_M[1]])  
    
    # the ysize of the section we will compare, 'extra' fraction bigger than 
    # the suggested overlap
    # this will be used as a number of pixels, i.e. a size not an index
    # remember we are using all of the x, and a section of the y
    testySize = int(np.floor(extra*imageOverlap*ysize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secT = imageT[:,ysize - testySize:,:].copy()
    secB = imageB[:,:testySize,:].copy()
    
    # calculate sobel signal in both sections
    sigT = np.array([measureSig(secT[c],pars) for c in range(len(secT))])
    sigB = np.array([measureSig(secB[c],pars) for c in range(len(secB))])
    # test whether those signals are big enough and combine to make 
    # one channel selection: 
    chan = (sigT > thresh)*(sigB > thresh)    

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigT = sigT[chan]
    sigB = sigB[chan]    

    # now amplify the channels by their signal measure so that the ones with 
    # signal are favoured during cross-correlation:
    secT = np.swapaxes(np.swapaxes(secT[chan],0,2)*sigT**aPow,0,2)
    secB = np.swapaxes(np.swapaxes(secB[chan],0,2)*sigB**aPow,0,2)

    # test if the sections contain signal, if not then just set the shift to 
    # what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        if printWarnings:
            print('findCen_T used default position1')
        shift = np.array([int(ysize*imageOverlap),0])
    elif any(chan)==False:
        if printWarnings:
            print('findCen_T used default position2, chan= ',chan,
                  ' sigT= ',sigT,
                  ' sigB= ',sigB)
        shift = np.array([int(ysize*imageOverlap),0])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secT,secB)
        shift = shift.astype(int)

        #if shifts are too big then ignore them and set default:
        # couldn't understand original condition here, so changed to simpler thing
        # i.e. we're just checking that final shift (=testqSize - shift[q]) isn't bigger than maxShiftFraction of total image size??
        #if  abs(shift[0]-(1-extra)*testySize) > maxShiftFrac*ysize:
        if  testySize - shift[0] > maxShiftFrac*ysize:    
            shift[0] = int(ysize*imageOverlap)
            counts[1] += 1
            if printWarnings:
                print('findCen_T used default position3')
        else:
            # the y shift of the actual image depends on the section you took!
            shift[0] = testySize - shift[0] 
        if abs(shift[1]) > maxShiftFrac*xsize:
            shift[1] = 0
            counts[1] += 1
            if printWarnings:
                print('findCen_T used default position4')
        else:        
            # the shift in x doesn't depend on the section you took:
            shift[1] = -shift[1]    

    return centreB_M - shift


def findCen_TS(imageL,centreL_M,imageT,centreT_M,imageBR,pars,counts):
    """This function finds the centre of an imageBR, knowing it should sit
    directly below of imageT and to the right of imageL, 
    i.e. is for all tiles starting from 2nd tile of 2nd row. 
    _TS stands for top and side.
    imageL has centre pixel index given by centreL.
    imageT has centre pixel index given by centreT.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.    
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring    
    """ 
    
    extra = 1.5
    
    # unpack parameters from parameters:
    ysize,xsize = [pars[0],pars[1]]
    imageOverlap = pars[2]
    method = pars[3] 
    thresh = pars[4]
    aPow = pars[5]
    maxShiftFrac = pars[6]
    minSize = pars[9]

    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = pars[7]
    sdt = pars[8]
    pars = (boxsize,sdt)    
    
    # start by aligning imageBR to imageL: 
    # (afterwards we will align to imageT and take an average)
    # initiate centreR so imageB is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreBR_M = np.array([centreL_M[0], xsize + centreL_M[1]])  
    
    # the xsize of the section we will compare, 'extra' fraction bigger than 
    # the suggested overlap
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the x, and a section of the y
    testxSize = int(np.floor(extra*imageOverlap*xsize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secL = imageL[:,:,xsize - testxSize:].copy()
    secBR = imageBR[:,:,:testxSize].copy()

    # calculate sobel signal in both sections
    sigL = np.array([measureSig(secL[c],pars) for c in range(len(secL))])
    sigBR = np.array([measureSig(secBR[c],pars) for c in range(len(secBR))])

    # test whether those signals are big enough and combine to 
    # make one channel selection: 
    chan1 = (sigL > thresh)*(sigBR > thresh)      

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigL = sigL[chan1]
    sigBR = sigBR[chan1]    
    
    # now amplify the channels by their signal measure so that the ones 
    # with signal are favoured during cross-correlation:
    secL = np.swapaxes(np.swapaxes(secL[chan1],0,2)*sigL**aPow,0,2)
    secBR = np.swapaxes(np.swapaxes(secBR[chan1],0,2)*sigBR**aPow,0,2)    
    
    # test if the secs contain signal, if not then just set the shift 
    # to what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift = np.array([0,int(xsize*imageOverlap)])
    elif any(chan1)==False:
        shift = np.array([0,int(xsize*imageOverlap)])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secL, secBR)
        shift = shift.astype(int)
    
        # if shifts are too big then ignore them and set default:
        if abs(shift[0]) > maxShiftFrac*ysize:
            shift[0] = 0
            counts[1] += 1           
        else:
            # the shift in y doesn't depend on the section you took:
            shift[0] = -shift[0]  
        if abs(shift[1]-(1-extra)*testxSize) > maxShiftFrac*xsize:
            shift[1] = int(xsize*imageOverlap)
            counts[1] += 1           
        else:
            # the x shift of the actual image depends on the section you took!
            shift[1] = testxSize - shift[1]       
    
    #shift the first estimation of centreBR_M:
    centreBR_M = centreBR_M - shift
    
    # now do everything again but aligning to imageT:
    # initiate centreR so imageB is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreBR_M2 = np.array([ysize + centreT_M[0], centreT_M[1]])  
    
    testySize = int(np.floor(extra*imageOverlap*ysize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secT = imageT[:,ysize - testySize:,:].copy()
    secBR = imageBR[:,:testySize,:].copy()
    
    # calculate sobel signal in both sections
    sigT = np.array([measureSig(secT[c],pars) for c in range(len(secT))])
    sigBR = np.array([measureSig(secBR[c],pars) for c in range(len(secBR))])
    # test whether those signals are big enough and combine 
    # to make one channel selection: 
    chan2 = (sigT > thresh)*(sigBR > thresh)     

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigT = sigT[chan2]
    sigBR = sigBR[chan2]        
    
    # now amplify the channels by their signal measure so that 
    # the ones with signal are favoured during cross-correlation:
    secT = np.swapaxes(np.swapaxes(secT[chan2],0,2)*sigT**aPow,0,2)
    secBR = np.swapaxes(np.swapaxes(secBR[chan2],0,2)*sigBR**aPow,0,2)    
    
    # test if the secs contain signal, if not then just set the 
    # shift to what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift2 = np.array([int(ysize*imageOverlap),0])
    elif any(chan2)==False:
        shift2 = np.array([int(ysize*imageOverlap),0])
        counts[0] += 1
    else:    
        shift2 = register_translation(secT, secBR)
        shift2 = shift2.astype(int)

        #if shifts are too big then ignore them and set default:
        if abs(shift[0]-(1-extra)*testySize) > maxShiftFrac*ysize:
            shift2[0] = int(ysize*imageOverlap)
            counts[1] += 1
        else:
            # the y shift of the actual image depends on the section you took!
            shift2[0] = testySize - shift2[0] 
        if abs(shift2[1]) > maxShiftFrac*xsize:
            shift2[1] = 0
            counts[1] += 1
        else:        
            # the shift in x doesn't depend on the section you took:
            shift2[1] = -shift2[1]       
    
    #shift the second estimation of centreBR_M2:
    centreBR_M2 = centreBR_M2 - shift2
    
    #take average of the two centres:
    centreBR_M = (centreBR_M + centreBR_M2)/2

    return centreBR_M.astype(int)



def register_translation(src_image,target_im):
    """
    This is a version of the skimage register_translation. Like the skimage 
    version, it does cross correlation of the image fourier transforms 
    ('fast fourier transform' algorithms) and returns the shifts calculated 
    from the maxima of that. I've made it work for multi channel and removed 
    the subpixel and error  stuff for simplicity. This takes images of 
    dimensions: (channels,ysize,xsize) and sums the cross-correlations of 
    channels before finding the max.
    
    Parameters
    ----------
    src_image,target_im : numpy array
        The images to align. Must have dimensions (NC,NY,NX). Where NC is 
        the number of channels.
    
    Returns
    -------
    shift : list of floats
        The (y,x) that you need to shift target_im by to align with src_im.
    """
    
    print("12/6/23 warning: changed import skimage fft to numpy fft, check it is the same!")
    
    if src_image.shape != target_im.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")
    if len(src_image.shape)!=3:
        raise ValueError("Images must have 3 axes")
    
    src_freq = [fft.fftn(src_image[c]) for c in range(len(src_image))]
    target_freq = [fft.fftn(target_im[c]) for c in range(len(target_im))]

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq[0].shape
    im_product = [im1 * im2.conj() for im1, im2 in zip(src_freq,target_freq)]
    cross_correlation = [fft.ifftn(im) for im in im_product]
    
    cross_correlation = sum(cross_correlation)
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    # this is done because FFT are periodic, 
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq[0].ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts



def enough_sig_Q():
    """
    Determines if there is enough signal in the two images to get a reliable 
    overlap. This isn't very clever, it's just doing a sobel filter and 
    seeing how much edge there is compared to a threshold that you give it.
    """

    
    
    return enoughQ
    

def measureSig(image,pars):
    """
    Returns a measure of amount of 'signal'. It normalises image by the mean, 
    gaussian smooths, applies a sobel filter and finds the mean. Therefore 
    edges are what constitute signal.
    
    Parameters
    ----------
    image : numpy.array, 2D
        The image you want to quantify signal for.
    pars : (int1,int2)
        int1 is the size of the square boxes that ou divide the images into 
        (see notes below). int2 is the standard deviation of the gaussian blur 
        that you apply. Note how this sets the scale of features that you 
        count as signal.
    
    Returns
    -------
    mean : float
        The measure of signal.
    
    Notes
    -----
    The method has a slight dependence on size so we break into boxes of 
    roughly constant size boxsize.
    """
    
    # unpack tha parameters:
    boxsize = pars[0]
    sdt = pars[1]
    
    ysize,xsize = image.shape
    # find the number of divisions that will give sizes closest to boxsize:
    NY = round(ysize/boxsize)
    if NY == 0:
        NY = 1
    NX = round(xsize/boxsize)
    if NX == 0:
        NX = 1
    
    vals = []
    for ySplit in np.array_split(image,NY):
        for xySplit in np.array_split(ySplit,NX,axis=1):
            if np.mean(xySplit)==0:
                val = 0
            else:
                val = np.mean(sobel(gaussian(xySplit/np.mean(xySplit),sdt)))
            vals.append(val)
    
    return np.mean(vals)



def splitSignalDetect(image,N):
    """This function isn't used in the main script it is just 
    used for testing out the sobel signal measure
    image has shape (1,ysize,xsize)... i.e. stupid 
    sectioniseStack needs a z-stack...
    ...but reassembleSections needs shape (N*N,ysize,xsize), so we need 
    to give as zstack size 1
    this splits it into N*N squares
    sdt is the size of the blur we apply, it shouldn't be changed 
    without changing thresh
    thresh is the value of the signal measure that we count as a 
    positive result...
    ...hoping it's a constant that only depends on sdt, since we normalise 
    everything else by means
    """
    sdt = 4
    thresh = 0.035
    image = zProj.sectioniseStack(image,N,1)
    sigList = []
    #ave  = np.mean(image[0])
    print('no. of sections: ',len(image),'  no. of z: ',len(image[0]),
          '  dimensions of sections: ',image[0][0].shape)
    for i,im in enumerate(image):
        sig = np.mean(sobel(gaussian(im[0]/np.mean(im[0]),sdt)))
        #sig = np.mean(sobel(gaussian(im[0]/ave,sdt)))
        if sig > thresh:
            image[i][0][:,:] = np.ones(im[0].shape)
        else:
            image[i][0][:,:] = np.zeros(im[0].shape)
        if i%10==0:
            sigList.append(sig)
    image = [im[0] for im in image]
    print(np.mean(sigList))
    return zProj.reassembleSections(image)


def findCentres2(ims,NX,imageOverlap,returnAll=False):
    """
    Aligns montage tiles using cross-correlation, returning the centres of the 
    tiles within the aligned full image.
    
    Parameters
    ----------
    ims : numpy array
        The images to be aligned, dimensions (NM,NC,NY,NX).
    NX : int
        Number of tiles in x dimension.
    imageOverlap : float
        The tile overlap set in the imaging software.
    
    Returns
    --------
    cenList_M : tuple
        All the indices of the centre pixel of each image.
    
    Notes
    ------
    In the code _M means the contents are in the final montage's coordinate 
    system.
    It assumes all images are the same size.
    """
    
    ysize,xsize = np.shape(ims[0,0])
    ycen = np.floor((ysize-1)/2).astype(int)
    xcen = np.floor((xsize-1)/2).astype(int)
    
    # initiate the main list we want to return
    cenList_M = []
    all_ccss = []
    all_shifts = []
    
    # main loop over all images:
    for j,im in enumerate(ims):
        # first image has centre local centre
        if j == 0:
            cenList_M.append(np.array([ycen,xcen]))
            all_ccss.append(np.zeros((10,10)))
            all_shifts.append([np.nan,np.nan,np.nan,np.nan])
        # now do the top row where only the sides are compared:
        elif j < NX:
            cen,out_ccs,shifts = findCen_S2(ims[j-1],cenList_M[j-1],im,imageOverlap,returnAll=True)
            cenList_M.append(cen)
            all_ccss.append(out_ccs)
            all_shifts.append(shifts)
        # then if it is the first colomn of a row you only compare the top:
        elif j % NX == 0:  
            cen,out_ccs,shifts = findCen_T2(ims[j-NX],cenList_M[j-NX],im,imageOverlap,returnAll=True)
            cenList_M.append(cen)
            all_ccss.append(out_ccs)
            all_shifts.append(shifts)
        # or general case you compare the top and side:
        else:
            cen,out_ccs,shifts = findCen_TS2(ims[j-1],cenList_M[j-1],ims[j-NX],cenList_M[j-NX],im,imageOverlap,returnAll=True)
            cenList_M.append(cen)
            all_ccss.append(out_ccs)
            all_shifts.append(shifts)
        
    if returnAll:
        return tuple(cenList_M),all_ccss,all_shifts

    else:
        return tuple(cenList_M)


def findCen_S2(imageL,centreL_M,imageR,imageOverlap,overlap_error_percent=50,y_error_percent=3,returnAll=False):
    """
    This one takes a different approach to finding the best overlap position. 
    It doesn't use FFT. It takes a range of pairs of images where each pair 
    is the last N cols of the left image and the first N cols of the right. 
    For the right it trims a bit off the ends of y too. Then it 
    cross-correlations the pair together and finds the best cross-correlation 
    among all pairs. The values of N are chosen to produce image sections that 
    range from a bit smaller to a bit larger than the overlap suggests. 
    We call the resulting matrix: the 'section sliding cross-correlation matrix'.
    Note that since we are comparing different images of different sizes the 
    normalisation was very important and has been well tested.

    Parameters
    ----------
    imageL : numpy array
        The whole left image to be aligned. Dimensions (NC,NY,NX).
    centreL_M : (y,x)
        The coordinaes of the centre of imageL in the coordinates of the full 
        montage.
    imageR : numpy array
        The whole right image to be aligned. Dimensions (NC,NY,NX).
    imageOverlap : float
        Fraction image overlap that is expected. You will probably get 
        problems if the true overlap is very different to this.
    overlap_error_percent : int
        The sizes of the sections taken will have a maximum size that is this 
        percentage bigger than the size suggested by the overlap. And a 
        minimum size this percentage smaller.
    x_error_percent : int
        The size in x of the second image will be trimmed by this percentage 
        on both sides (so final size is reduced by twice this percentage). 
        The cross-correlation therefore allows for this size of error in x.
    returnAll : bool
        Whether to return the cross-correlation matrix for further checks and 
        the 'shifts'.
    """
    
    # unpack parameters from parameters:
    #ysize,xsize = [parameters[0],parameters[1]]
    NC,ysize,xsize = imageL.shape  

    start_x_ind = int(xsize*imageOverlap*(1-(overlap_error_percent/100)))
    end_x_ind = int(xsize*imageOverlap*(1+(overlap_error_percent/100)))
    
    delta_y = int(ysize*y_error_percent/100)
    
    # now build our 'section-slid cross-correlation matrix'
    out_ccs = np.zeros(((delta_y*2)+1,end_x_ind-start_x_ind),dtype='float32')
    for i in range(NC):
        cross_corrs = []
        imageL2,imageR2 = otsu_contrast([imageL[i].copy(),imageR[i].copy()])
        imageL2,imageR2 = best_guess_overlap_normalisation(imageL2,imageR2,imageOverlap,'LR')

        # first check if there is enough signal
        secL = imageL2[:,-(end_x_ind-1):].copy()
        secR = imageR2[delta_y:-delta_y,:end_x_ind-1].copy()    
        # if there isn't enough signal, try once more with multi_otsu in case the problem was a super high blob intensity ruining the segmentation
        if not enough_signalQ(secL,200) and not enough_signalQ(secR,200):
            imageL2,imageR2 = otsu_contrast([imageL[i].copy(),imageR[i].copy()],multi=True)
            imageL2,imageR2 = best_guess_overlap_normalisation(imageL2,imageR2,imageOverlap,'LR') 
            secL = imageL2[:,-(end_x_ind-1):].copy()
            secR = imageR2[delta_y:-delta_y,:end_x_ind-1].copy()              
            if not enough_signalQ(secL,200) and not enough_signalQ(secR,200):
                continue
        
        for r in range(start_x_ind,end_x_ind):
            secL = imageL2[:,-r:].copy()
            secR = imageR2[delta_y:-delta_y,:r].copy()
            secL = filters.gaussian(secL,sigma=3,preserve_range=True)
            secR = filters.gaussian(secR,sigma=3,preserve_range=True)     
            cross_corrs.append(cv.matchTemplate(secL.astype('float32'),secR.astype('float32'),eval('cv.TM_SQDIFF_NORMED')))
        cross_corrs = np.hstack(cross_corrs)
        cross_corrs = 1 - cross_corrs # because we always do cv.TM_SQDIFF_NORMED so needs inversion
        cross_corrs = cross_corrs - np.min(cross_corrs) # this kind of scaling is important, it minimises the contribution of low quality ccs etc
        
        if good_crosscorr_matrixQ(cross_corrs): # check if you have a nice clear peak
            out_ccs += cross_corrs     

    # check you have found some good ccs
    if np.any(out_ccs):
        max_y_ind,max_x_ind = np.unravel_index(np.argmax(out_ccs),out_ccs.shape)
    
        x_back_shift = range(start_x_ind,end_x_ind)[max_x_ind]

        y_shift = max_y_ind - delta_y

        cen = (centreL_M[0] + y_shift, centreL_M[1] + xsize - x_back_shift)

        shifts = [x_back_shift,y_shift,np.nan,np.nan]
    else:
        # no good peak found so use the imageOverlap shift instead of whatever random max it found
        cen = (centreL_M[0],centreL_M[1] + xsize - int(xsize*imageOverlap))
        
        shifts = [np.nan,np.nan,np.nan,np.nan]
        
    if returnAll:
        return cen,out_ccs,shifts
    else:
        return cen

        

def findCen_T2(imageT,centreT_M,imageB,imageOverlap,overlap_error_percent=50,x_error_percent=3,returnAll=False):
    """
    This one takes a different approach to finding the best overlap position. 
    It doesn't use FFT. It takes a range of pairs of images where each pair 
    is the last N rows of the top image and the first N rows of the bottom. 
    For the bottom it trims a bit off the ends of x too. Then it 
    cross-correlations the pair together and finds the best cross-correlation 
    among all pairs. The values of N are chosen to produce image sections that 
    range from a bit smaller to a bit larger than the overlap suggests. 
    We call the resulting matrix: the 'section sliding cross-correlation matrix'.
    Note that since we are comparing different images of different sizes the 
    normalisation was very important and has been well tested.

    Parameters
    ----------
    imageT : numpy array
        The whole top image to be aligned. Dimensions (NC,NY,NX).
    centreT_M : (y,x)
        The coordinaes of the centre of imageT in the coordinates of the full 
        montage.
    imageB : numpy array
        The whole bottom image to be aligned.
    imageOverlap : float
        Fraction image overlap that is expected. You will probably get 
        problems if the true overlap is very different to this.        
    overlap_error_percent : int
        The sizes of the sections taken will have a maximum size that is this 
        percentage bigger than the size suggested by the overlap. And a 
        minimum size this percentage smaller.
    x_error_percent : int
        The size in x of the second image will be trimmed by this percentage 
        on both sides (so final size is reduced by twice this percentage). 
        The cross-correlation therefore allows for this size of error in x.
    returnAll : bool
        Whether to return the cross-correlation matrix for further checks and the 'shifts'.
    """
    
    # unpack parameters from parameters:
    NC,ysize,xsize = imageT.shape
    
    start_y_ind = int(ysize*imageOverlap*(1-(overlap_error_percent/100)))
    end_y_ind = int(ysize*imageOverlap*(1+(overlap_error_percent/100)))

    delta_x = int(xsize*x_error_percent/100)

    out_ccs = np.zeros((end_y_ind-start_y_ind,(delta_x*2)+1),dtype='float32')
    for i in range(NC):
        cross_corrs = []
        imageT2,imageB2 = otsu_contrast([imageT[i].copy(),imageB[i].copy()])
        imageT2,imageB2 = best_guess_overlap_normalisation(imageT2,imageB2,imageOverlap,'TB')

        # first check if there is enough signal
        secT = imageT2[-(end_y_ind-1):,:].copy()
        secB = imageB2[:end_y_ind-1,delta_x:-delta_x].copy()      
        # if there isn't enough signal, try once more with multi_otsu in case the problem was a super high blob intensity ruining the segmentation
        if not enough_signalQ(secT,200) and not enough_signalQ(secB,200):
            imageT2,imageB2 = otsu_contrast([imageT[i].copy(),imageB[i].copy()],multi=True)
            imageT2,imageB2 = best_guess_overlap_normalisation(imageT2,imageB2,imageOverlap,'TB')    
            secT = imageT2[-(end_y_ind-1):,:].copy()
            secB = imageB2[:end_y_ind-1,delta_x:-delta_x].copy()              
            if not enough_signalQ(secT,200) and not enough_signalQ(secB,200):
                continue
        
        for r in range(start_y_ind,end_y_ind):
            secT = imageT2[-r:,:].copy()
            secB = imageB2[:r,delta_x:-delta_x].copy()
            secT = filters.gaussian(secT,sigma=3,preserve_range=True)
            secB = filters.gaussian(secB,sigma=3,preserve_range=True)    

            cross_corrs.append(cv.matchTemplate(secT.astype('float32'),secB.astype('float32'),eval('cv.TM_SQDIFF_NORMED')))

        cross_corrs = np.vstack(cross_corrs)

        cross_corrs = 1 - cross_corrs # because we always do cv.TM_SQDIFF_NORMED so needs inversion
        cross_corrs = cross_corrs - np.min(cross_corrs) # this kind of scaling is important, it minimises the contribution of low quality ccs etc
        
        if good_crosscorr_matrixQ(cross_corrs): # check if you have a nice clear peak
            out_ccs += cross_corrs        

    # check you have found some good ccs
    if np.any(out_ccs):
        max_y_ind,max_x_ind = np.unravel_index(np.argmax(out_ccs),out_ccs.shape)
    
        y_back_shift = range(start_y_ind,end_y_ind)[max_y_ind]

        x_shift = max_x_ind - delta_x

        cen = (centreT_M[0] + ysize - y_back_shift, centreT_M[1] + x_shift)

        shifts = [np.nan,np.nan,y_back_shift,x_shift]

    else:
        # no good peak found so use the imageOverlap shift instead of whatever random max it found
        cen = (centreT_M[0] + ysize - int(ysize*imageOverlap), centreT_M[1])
        
        shifts = [np.nan,np.nan,np.nan,np.nan]

    if returnAll:
        return cen,out_ccs,shifts
    else:
        return cen



def findCen_TS2(imageL,centreL_M,imageT,centreT_M,imageBR,imageOverlap,overlap_error_percent=50,side_error_percent=3,returnAll=False):
    """
    This one combines findCen_T() and findCen_S() to find best overlap of tile 
    which has a neighbour above and to the left. It does simplest thing of 
    finding position according to top and left separately and taking average 
    position.

    Parameters
    ----------
    imageT,imageL : numpy array
        The whole top/left image to be aligned. Dimensions (NC,NY,NX).
    centreT_M,centreL_M : (y,x)
        The coordinaes of the centre of imageT,imageL in the coordinates of the full 
        montage.
    imageBR : numpy array
        The whole bottom-right image to be aligned.
    imageOverlap : float
        Fraction image overlap that is expected. You will probably get 
        problems if the true overlap is very different to this.        
    overlap_error_percent : int
        The sizes of the sections taken will have a maximum size that is this 
        percentage bigger than the size suggested by the overlap. And a 
        minimum size this percentage smaller.
    side_error_percent : int
        The size in direction perpendicular to the overlap direction of the 
        second image will be trimmed by this percentage on both sides (so 
        final size is reduced by twice this percentage). The cross-correlation 
        therefore allows for this size of error in x.
    returnAll : bool
        Whether to return the cross-correlation matrices for further checks and the 'shifts'.
    """

    cen_S,out_ccs_S,shifts_S = findCen_S2(imageL,
                             centreL_M,
                             imageBR,
                             imageOverlap=imageOverlap,
                             overlap_error_percent=overlap_error_percent,
                             y_error_percent=side_error_percent,
                             returnAll=True)

    cen_T,out_ccs_T,shifts_T = findCen_T2(imageT,
                                 centreT_M,
                                 imageBR,
                                 imageOverlap=imageOverlap,
                                 overlap_error_percent=overlap_error_percent,
                                 x_error_percent=side_error_percent,
                                 returnAll=True)

    blank_shifts = [np.nan,np.nan,np.nan,np.nan]
    
    if shifts_S==blank_shifts and shifts_T==blank_shifts:
        cen = (np.array(cen_S) + np.array(cen_T))/2
        shifts = blank_shifts
    elif shifts_S==blank_shifts:
        cen = cen_T
        shifts = shifts_T
    elif shifts_T==blank_shifts:
        cen = cen_S
        shifts = shifts_S
    else:
        cen = (np.array(cen_S) + np.array(cen_T))/2
        shifts = [shifts_S[0],shifts_S[1],shifts_T[0],shifts_T[1]]

    if returnAll:
        return cen,[out_ccs_S,out_ccs_T],shifts
    else:
        return cen




def border_pixelsQ(im):
    """
    Just checks if there are any true pixels at the edges of an image
    """
    if any(im[0,:]):
        return True
    elif any(im[-1,:]):
        return True
    elif any(im[:,0]):
        return True
    elif any(im[:,-1]):
        return True
    else:
        return False



def good_crosscorr_matrixQ(corr):
    """
    This function decides if you have found a good match or not in your 
    cross-correlation matrix. So it is essentially peak detection. But I made 
    a bit of a weird way that I think is good. Methods based on mean/sdt/max 
    were unreliable because the cross-correlation matrices give a range of 
    weird results depending on image content. E.g. images with mainly noise 
    give slopes resulting from changing section size and imaging edge 
    abberations. Images with lots of signal give big broad peaks that increase 
    the mean and don't have the hugest peak amplitude in comparison. Image 
    segmentation seem like a good way since it somehow makes use of the 
    inherent peak shape. Histogram based segmentation was no good though. 
    What we care about is the maximum so the threshold is found as a fraction 
    of the maximum. Then we say there must be nothing at the edge because this 
    excludes slope signals and also helps with random noise. This may need 
    more work for a greater range of images, perhaps particularly small images.

    Since we now use just SQRDIFF_NORM method (which we invert) we know that 
    max possible is 1 and this max will be lower with reduced quality of match 
    so we also set a limit that the max must be above 0.2.
    """

    # smoothing is good but it helps the noisey images merge/consolidate peaks so keep to minimum (i.e. sigma=1)
    corr = filters.gaussian(corr.copy(),sigma=1,preserve_range=True)

    # since we now do cv.TM_SQDIFF_NORMED the max will always be 1 (we have already inverted with 1 - ccs)
    # haven't put min_peak as a parameter in function yet since seems very reasonable
    min_peak = 0.2
    if np.max(corr)<min_peak:
        return False

    # 0.75 is so reasonable here that we don't even have it as a function parameter
    peak_thresh = 0.75
    # it kinds of says peaks other than the main peak must have max smaller than 75% the max of the main peak
    maxx = np.max(corr)
    corr = corr>(maxx*peak_thresh)   

    # we don't allow any border pixels because so many noise cross-correlation matrices look a bit like downward constant slopes
    # i.e. you have to make sure you're probed big enough area to give a peak that isn't at the edge
    if border_pixelsQ(corr):
        return False
    # we only allow one peak, any other peaks must be smaller than peak_thresh of the main peak
    elif len(np.unique(measure.label(corr)))>2:
        return False
    else:
        return True



def safely_rescale_image(image,ratio):
    """
    Just rescales image by ratio making sure you don't get datatype overflow problems
    """
    data_type = image.dtype
    data_type_max = np.iinfo(data_type).max
    thresh = data_type_max/ratio
    image = image.copy()
    image[image>thresh] = thresh
    return (image*ratio).astype(data_type)
    

def best_guess_overlap_normalisation(image1,image2,overlap,LRorTB):
    """
    This looks at just the 'best guess' overlap regions of two images 
    and rescales image2 so that they have the same max intensity in those 
    sections.
    """

    image1 = image1.copy()
    image2 = image2.copy()

    data_type = image2.dtype
    
    ysize,xsize = image1.shape
    if LRorTB=='TB':
        crop_ind = int(ysize*overlap)
        secT = filters.gaussian(image1[-crop_ind:,:], sigma=3,preserve_range=True)
        secB = filters.gaussian(image2[:crop_ind,:], sigma=3,preserve_range=True)
        max_T = np.max(secT)
        max_B = np.max(secB)
        ratio = max_T/max_B        
    elif LRorTB=='LR':
        crop_ind = int(xsize*overlap)
        secL = filters.gaussian(image1[:,-crop_ind:], sigma=3,preserve_range=True)
        secR = filters.gaussian(image2[:,:crop_ind], sigma=3,preserve_range=True)
        max_L = np.max(secL)
        max_R = np.max(secR)
        ratio = max_L/max_R
    else:
        raise Exception('unrecognised LRorTB parameter input')

    return [image1,safely_rescale_image(image2,ratio)]
    
    

def otsu_contrast(image,min_fraction=0.75,max_centile=99,verbose=False,multi=False):
    """
    This is a histogram rescaling that chooses a pixel value near the otsu 
    threshold as the histogram minimum. This means the rescaling will clip 
    nicely around the signal region of the histogram. It is good for images 
    with a lot of of empty space. 

    Parameters
    -----------

    image : 2D numpy array or list
        The input image. If list then it finds one min and max to aplly to 
        all images.
    min_fraction : float [0-1]
        The otsu threshold found will be multiplied by this and the result 
        is the threshold where the minimum clipping will occur. I.e. values 
        lower than this are all zero.
    max_centile : int [0-100]
        The threshold for clipping the maximum will be the value a this 
        centile of the 'minimum-clipped image pixel values'. I.e. we do the 
        minimum clipping first then take the percentile. This avoids removing 
        everything if the otsu threshold is a very high pixel value (i.e. 
        very small signal region). This is better than just setting 
        to the max since it is less effected by stray high pixels.
    multi : bool
        Whether or not to use the first multi_otsu threshold. multi_otsu here 
        separates signal into 3 - we hope any super high signal mess will be 
        in the top threshold so hope that we get the main signal by using the 
        lower threshold.
    """
    if isinstance(image,list):
        data_type = image[0].dtype
        image2 = [im.copy() for im in image]
        im_combined = np.hstack(tuple(image))
    else:
        data_type = image.dtype
        image2 = [image.copy()]
        im_combined = image.copy()
    
    dtype_max = np.iinfo(data_type).max    
    
    blur = filters.gaussian(im_combined, sigma=3,preserve_range=True)

    if multi:
        thresh = filters.threshold_multiotsu(blur)
        thresh = thresh[0]
    else:
        thresh = filters.threshold_otsu(blur)

    min_thr = int(thresh*min_fraction)
    max_thr = int(np.percentile(im_combined[im_combined>min_thr],max_centile))

    if verbose:
        print('min_thr: ',min_thr)
        print('max_thr: ',max_thr)
        print('type(image2): ',type(image2))  

    for im in image2:
        if verbose:
            print('type(im): ',type(im))        
        im[im<min_thr] = min_thr
        im[im>max_thr] = max_thr

    image2 = [(im - min_thr)*(dtype_max/max_thr) for im in image2]

    if len(image2)==1:
        return image2[0].astype(data_type)
    else:
        return [im.astype(data_type) for im in image2]


def enough_signalQ(image,min_N_signal_pix=200):
    """
    This is the simplest check of whether there is enough signal to trust 
    a cross-correlation. It just checks the mean of the image.
    
    It relies on the fact that we have done good contrast things to the image 
    so any signal should have pixels of a value that is a reasonable fraction 
    of the image-type maximum. 

    Parameters
    ----------
    min_N_signal_pix : int
        How many signal pixels you want to say is the minimum to trust an 
        alignment.
    """
    Npix = np.product(image.shape)
    dtype_max = np.iinfo(image.dtype).max
    
    thresh = ((min_N_signal_pix*dtype_max)/Npix)

    if np.mean(image)<thresh:
        return False
    else:
        return True
    





    
