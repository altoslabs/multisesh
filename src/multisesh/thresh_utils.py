import os
import csv
import re
import math

import numpy as np
import scipy.optimize as opt
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import cv2 as cv


def getWin(root,ex,fold):
    """
    Gets the window of an experiment.
    
    Parameters
    ----------
    root : str
        The path to the directory where all data is.
    ex : dict
        The row of the data log xlsx file corresponding to your experiment. 
        Should have details of the Date and Exp no.
    fold : str
        The name of the directory containing the windows in the XPath parent 
        directory.
    
    Returns
    --------
    win : list of 2-lists
        [x,y] coordinates of each of the corners of the window. Note this is 
        different to xfold.WindowDic which has the formate [y,x].
    """
    
    path = os.path.join(root,str(ex['Date'])+'Gradients',fold)
    path = os.path.join(path,'Exp'+str(ex['Experiment']))
    winPath = [os.path.join(path,f) for f in os.listdir(path) if '.txt' in f][0]
    with open(winPath) as w:
        win = w.read()
    win = [[float(w) for w in w.split('\t')] for w in win.split('\n') if w!='']
    if ex['Flip'].lower()=='true':
        wy_min = min([w[1] for w in win])
        wy_max = max([w[1] for w in win])
        win = [[w[0],wy_min+wy_max-w[1]] for w in win]
    return win


def importCSVData(root,ex,csvDir,channel,t,clip=False,flip=True):
    """
    Imports csv data of pixel intensities.
    
    Parameters
    ----------
    root : str
        The path to the directory where all data is.    
    ex : dict
        The row of the data log xlsx file corresponding to your experiment. 
        Should have details of the Date and Exp no.
    csvDir : str
        The path to the directory containing the measurement csvs.
    channel : str
        e.g. 'YFP','BF' etc... should be in the name of the CSV file.
    t : int or str
        Which time point to take the data from. If int then it corresponds to 
        the time point of the final concatenated summary video. I.e. it starts 
        at 1 not 0 and skips blank time points so isn't aligned with session 
        time. It is assuming you have extracted data for every frame in the 
        concatenated video! If str then it is the time in min of the tdata you 
        are interested in, i.e. it's the str in the csv filename. It isn't 
        assuming you have calculated for all.
    clip : str
        Whether to clip the csv data. If it evaluates True then it should be a 
        string name of a directory in the xfold parent directory. If the 
        directory contains field directories containing txt files called 
        clip.txt then they will be loaded. The txt files should contain either 
        one or two ints. If one int then it clips that many pixels from the 
        end. If two ints separated by comma then that gives start and end clip 
        sizes. Clipped values are converted to nan. Note that the clipping 
        starts from the first non-nan number so the mask and trapezium windows 
        are dealt with properly.
    flip : bool
        Whether to flip the data (vertically) if ex says it is flipped data.
        
    Returns
    -------
    data : list of lists
        Each list represents one y-position in the chip. The first element of 
        each list is that y-position in pixels and the following elements are 
        pixels intensities for each segment along the x-direction of the chip.
    """
    floatReg = r'(\d+\.?\d*|\d*\.?\d+)'
    csvDir = os.path.join(root,str(ex['Date'])+'Gradients',csvDir)
    csvDir = os.path.join(csvDir,'Exp'+str(ex['Experiment']))
    csvFP = [os.path.join(csvDir,fp) for fp in os.listdir(csvDir)]
    csvFP = [c for c in csvFP if '.csv' in c and channel in c]
    if isinstance(t,str):
        csvFP = [c for c in csvFP if t in c][0]
    else:
        csvFP = csvFP[t-1]
    data = []   
    with open(csvFP,'r') as csvfile:
        rowGen = csv.reader(csvfile)
        next(rowGen)
        for row in rowGen:
            dist = [float(re.search(floatReg,row[0]).group(1))]
            data.append(dist+[float(x) for x in row[1:]])

    if flip and ex['Flip'].lower()=='true':
        data2 = []
        for i in range(len(data)):
            data2.append([data[i][0]]+data[-i-1][1:])
        data = data2.copy()
    
    if clip and isinstance(clip,str):
        clipDir2 = os.path.join(root,str(ex['Date'])+'Gradients',clip)
        clipDir2 = os.path.join(clipDir2,'Exp'+str(ex['Experiment']))
        clipFP = os.path.join(clipDir2,'Clip.txt')
        if os.path.exists(clipFP):
            with open(clipFP,'r') as file:
                clip = file.read()
            reg1 = re.search(r'(^\d+),(\d+)',clip)
            reg2 = re.search(r'(^\d+)',clip)
            assert reg1 or reg2,'No int found at start of Clip.txt.'
            if reg1:
                clipS = int(reg1.group(1))
                clipF = int(reg1.group(2))
                clipS = [clipS]*(len(data[0])-1)
                clipF = [clipF]*(len(data[0])-1)
                for i in range(len(clipS)):
                    for j in range(len(data)):
                        if clipS[i]>0:
                            if not math.isnan(data[j][i+1]):
                                data[j][i+1] = float('nan')
                                clipS[i]-=1
                        else:
                            break
                for i in range(len(clipF)):
                    for j in range(-1,-len(data),-1):
                        if clipF[i]>0:
                            if not math.isnan(data[j][i+1]):
                                data[j][i+1] = float('nan')
                                clipF[i]-=1
                        else:
                            break
            else:
                clip = int(reg2.group(1))
                clip = [clip]*(len(data[0])-1)
                for i in range(len(clip)):
                    for j in range(-1,-len(data),-1):
                        if clip[i]>0:
                            if not math.isnan(data[j][i+1]):
                                data[j][i+1] = float('nan')
                                clip[i]-=1
                        else:
                            break
    return data


def findPadding(im):
    """
    This returns the number of blank rows and columns at the borders of an 
    image. It for now actually only checks the first half of the columns since 
    we are using images with time labels put on after which overlap with 
    the padding.
    
    Parameters
    ----------
    im : np array
    
    Returns
    -------
    padT,padB,padL,padR : int
        The number of rows/columns of padding in pixels: Top,Bottom,Left,Right
    """
    padT = 0
    padB = 0
    padL = 0
    padR = 0
    for i in range(im.shape[0]):
        if np.max(im[i,:im.shape[1]//2].ravel())==0:
            padT += 1
        else:
            break
    for i in range(im.shape[0]):
        if np.max(im[-i-1,:im.shape[1]//2].ravel())==0:
            padB += 1
        else:
            break
    for i in range(im.shape[1]):
        if np.max(im[:im.shape[0]//2,i].ravel())==0:
            padL += 1
        else:
            break   
    for i in range(im.shape[1]):
        if np.max(im[:im.shape[0]//2,-i-1].ravel())==0:
            padR += 1
        else:
            break   
    checkY = padT<=im.shape[0]//2 and padB<=im.shape[0]//2
    checkX = padL<=im.shape[1]//2 and padR<=im.shape[1]//2
    er = 'A pad was > half the axis, will cause problems you haven\'t covered!'
    if not checkY or not checkX:
        print(er)
    return [padT,padB,padL,padR]


def removePadding(im,padT,padB,padL,padR):
    """
    Removes padding from img.
    
    Parameters
    ----------
    im : numpy.array
        The image
    padX : int
        The number of zero-valued at edge X. See findPadding().
    
    Returns
    -------
    im : array
        im but with the zeros removed.
    """
    
    im = im[padT:,padL:]
    if padB:
        im = im[:-padB,:]
    if padR:
        im = im[:,:-padR]
    return im


def padToSize(im,size):
    """
    Adds 0-valued padding to im to make it the specified size. Padding is 
    added evenly to opposite sides.
    
    Parameters
    ----------
    im : numpy.array
        Source image
    size : tuple of ints
        (ysize,xsize) of the desired output.
        
    Returns
    -------
    im : np.array
        Image of specified size.
    """
    h = im.shape[0]
    w = im.shape[1]

    a = (size[0] - h) // 2
    aa = size[0] - a - h

    b = (size[1] - w) // 2
    bb = size[1] - b - w

    return np.pad(im,pad_width=((a,aa),(b,bb)))


def calculateThresholds(data,
                        frac=0.5,
                        checkDataRange=False,
                        checkWin=False,
                        checkL=False,
                        plot=False):
    """
    Calculates the threshold position along y where intensity I transitions 
    from off to on. Does so via a logistic fit. 
    
    Parameters
    ----------
    data : list of lists
        Each element of list must be a list where the first element is the 
        y-value and the following elements are independent intensity measures.
        Y-value is relative to the top of the window.
    frac = int
        The increase in intensity within the fit logistic
        that defines the returned threshold, as a fraction of 
        L (= the logistic y-stretch).
    checkDataRange : bool
        Whether to check for each segment that the 'current mean y-threshold' 
        is within the range of valid y-values in that segment. The 'current 
        mean y-threshold' is being build sequentially during this check, 
        in which segments are ordered by the size of their range of y-values.
        This is used to remove irrelevant segments from triangle 
        chips. Don't use with trapezium!
    checkWin : False or win
        If not False then win must be the list of [x,y] points marking the 
        window and it will make y-thresh='NAN' if the calculated y-thresh 
        doesn't lie within the window.
    checkL : bool or int
        If not False then it must be an interger and the fit parameter L 
        (y-stretch of the logistic) must be bigger than that integer otherwise 
        the 'NAN' will be returned for the y-thresh. This is just a way to 
        filter out the noisy results that are likely to be wrong.
    plot : False or int
        If int then it will plot the data and fit of the segment with that 
        index.
    
    Returns
    --------
    y_threshes : list
        The calculated threshold position in y. One for each set of 
        independent intensity measures. 'NAN' if none found.
        
    Notes
    ------
    Requires definition of function sig!
    """
    
    if checkWin:
        wyMax = max([w[1] for w in checkWin]) - min([w[1] for w in checkWin])
        
    y_threshes = []
    threshFoundFlag = False
    rangs = []
    for seg in range(1,len(data[0])):
        y = np.array([d[0] for d in data if not math.isnan(d[seg])])
        if checkDataRange:
            if y.size==0:
                rangs.append([0,0])
            else:
                rangs.append([min(y),max(y)])
        I = np.array([d[seg] for d in data if not math.isnan(d[seg])])
        
        if not I.size>0:
            y_threshes.append('NAN')
            continue
        try:
            if not threshFoundFlag:
                guessParams = [np.max(I)-np.min(I),np.mean(I),0.04]
                guessParams += [guessThresh(y,I)]
            popt,pcov = opt.curve_fit(sig,y,I,guessParams,method="trf")
            y_t = popt[3] - math.log((1/frac)-1)/popt[2]
            goodThreshFlag = True
            if checkWin and y_t > wyMax or y_t < 0:
                goodThreshFlag = False
            if checkL and checkL > popt[0]:
                goodThreshFlag = False
            if goodThreshFlag:
                guessParams = popt
                y_threshes.append(y_t)
                threshFoundFlag = True
            else:
                y_threshes.append('NAN')
            if plot==seg:
                plt.plot(y,I)
                plt.plot(y,sig(y,*popt))
        except RuntimeError:
            if plot==seg:
                plt.plot(y,I)
            y_threshes.append('NAN')
            
    if checkDataRange:
        ymean = None
        iTR = zip(range(len(y_threshes)),y_threshes,rangs)
        iTR = sorted(iTR,key=lambda p:p[2][1]-p[2][0],reverse=True)
        for i,I in enumerate(iTR):
            if I[1]!='NAN':
                ymean = [I[1],1]
                del iTR[i]
                break
        if ymean:
            for i,yt,r in iTR:
                if ymean[0]<r[1]-20 and ymean[0]>r[0]+20 and yt!='NAN':
                    ymean[0] = (ymean[0]*ymean[1] + yt)/(ymean[1]+1)
                    ymean[1] += 1
                else:
                    y_threshes[i] = 'NAN'
                    
    return y_threshes


def guessThresh(y,I):
    """
    This is a rough and error-prone way to find the y-position of the 
    intensity threshold. It just searches for where the intensity is 
    closest to half way between the min and max values. Just use this 
    to initiate logarithmic fit.
    
    Parameters
    ----------
    y : np.array
        The positions along the concentration gradient in the chip.
    I : np.array
        The pixel intensities.
    
    Returns
    --------
    y_thresh : float
        The guess of y_thresh.
    """
    y2 = np.convolve(y,np.ones(15),'valid')/15 # = rolling mean
    I2 = np.convolve(I,np.ones(15),'valid')/15 # = rolling mean
    #IMin = I2.min()
    IMin = np.nanmin(I2)
    #IMax = I2.max()
    IMax = np.nanmax(I2)
    IHalf = (IMax+IMin)/2
    IMinInd = int(np.where(I2==IMin)[0][-1])
    IMaxInd = int(np.where(I2==IMax)[0][0])
    if IMinInd>=IMaxInd:
        IMinInd = 0
        IMaxInd = -2
    IHalfInt = np.argmin(np.absolute(I2[IMinInd:IMaxInd+1]-IHalf)) + IMinInd
    return y2[IHalfInt]

def sig(x,L,A,k,x_0):
    "logistic function"
    return L/(1.+np.exp(-k*(x-x_0))) + A 


def drawThreshOnIm(thresh,image,window,colour,thickness,downsize=1,flip=False):
    """
    Draws lines on image according to the thresholds provided in thresh.
    
    Parameters
    ----------
    thresh : list
        The values of the thresholds, in pixels from the top of the window. 
        Each value in the list is an independent measure from equally spaced 
        segments along the window.
    image : np.array
        The image you draw on. Currently RGB but others may work.
    window : list
        The [x,y] coordinates of the four corners of the rectangle marking the 
        boundaries relative to which the thresholds were measured and along 
        which the separate measures where made (see thresh).
    downsize : int
        The image and window may have been downsized relative to the thresh 
        data. If this is the case then we divide them by this factor.
    flip = bool
        Indicates whether the image is flipped wrt the thresholds you've 
        calculated.
        
    Returns
    -------
    lines : list of list of int
        The [[x0,y_thresh],[x1,y_thresh]] that draw the line of the thresholds 
        on image.
    """
    # make lines
    lines = []
    NSegs = len(thresh)
    wx_min = min([w[0] for w in window])
    wx_max = max([w[0] for w in window])
    segL = int((wx_max - wx_min)/NSegs)
    for i,y in enumerate(thresh):
        if y=='NAN': 
            y_t = 'NAN'
        elif flip:
            wy_max = max([w[1] for w in window])
            y_t = int(wy_max - y/downsize)
        else:
            wy_min = min([w[1] for w  in window])
            y_t = int(y/downsize + wy_min)
        x_s = int(wx_min + segL*i)
        lines.append([(x_s,y_t),(x_s+segL,y_t)])  
        
    # draw thresh lines
    for l in lines:
        if l[0][1]!='NAN':
            cv.line(image,l[0],l[1],colour,thickness)
    
    return lines


def dataLog2ConcFunc(chip,profile,concentration,win=False,flip=False,quad='simple'):
    """
    Creates a function that converts position to concentration for all the 
    various chips and profiles in our experiments.
    
    Parameters
    ----------
    chip/profile/concentration : str
        These are all experiment parameters as defined by the experiment log 
        spreadsheet.
    win : list of list of int
        The coordinates of the window for the experiment, as measured in 
        image j with a rectangle, using a processed and possibly region 
        extracted bright field image. Therefore these coordinates will be 
        relative to the region extracted and anything added during processing, 
        e.g padding. It is only required for trapezium. 
    flip : bool
        In the default case we chose low y has high concentration (somewhat 
        counter-intuitively!). Some of the images are upside down and need to 
        be flipped, if so then set this to true.
    quad : str {'simple','whiteside'}
        Which quadratic formula to use. 'simple' is without linear term. 
        'whiteside' is what the whiteside gradient device with one input at 
        concentration and the other 2 at zero.
        
    Returns 
    -------
    f : function
        The function which converts positions to concentrations.
        Parameters
        ----------
        y : float 
            The position in um relative to the window edge.
        x : float
            The position in um along the x-axis of the position reading y. 
            Relative to the left side of the window. Important e.g. in 
            trapezium where the concentration varies in x for constant y.
        Returns 
        -------
        c : float
            Concentration of threshold in units that the parameter 
            'concentration' is given in.
    """
    
    if chip=='Trapezium':
        assert win, 'You must provide win for Trapezium chips.'
    
    if chip=='Rectangle0p7':
        y_m = 880
    elif chip=='Rectangle1':
        y_m = 1000
    elif chip=='Rectangle2p5':
        y_m = 2500
    elif chip=='Triangle':
        y_m = 1000
    elif chip=='Trapezium':
        topLeft = sorted(sorted(win)[0:2],key=lambda p:p[1])[0]
        win = [[w[0]-topLeft[0],w[1]-topLeft[1]] for w in win]
        win = win[:4]
        topLeft,bottomLeft = sorted(sorted(win)[:2],key=lambda p:p[1])
        topRight,bottomRight = sorted(sorted(win)[2:],key=lambda p:p[1])
        lenY_0 = bottomLeft[1] - topLeft[1]
        slopeTop = (topRight[1] - topLeft[1])/(topRight[0] - topLeft[0])
        slopeBottom = (bottomRight[1] - bottomLeft[1])/(bottomRight[0] - bottomLeft[0])
    else:
        raise Exception('Unknown chip type')
    
    if profile=='Linear':
        if chip=='Trapezium':
            def f(y,x):
                if y=='NAN':
                    return 'NAN'
                lenY_x = (bottomLeft[1]+x*slopeBottom)-(topLeft[1]+x*slopeTop)
                y2 = lenY_0*(y - (topLeft[0] + x*slopeTop)) / lenY_x
                if flip:
                    return concentration*(y2/lenY_0)
                else:
                    return concentration*((lenY_0 - y2)/lenY_0)
        else:
            A = concentration/y_m
            def f(y,*args,**kwargs):
                if y=='NAN':
                    return 'NAN'
                if flip:
                    return A*y
                else:
                    return A*(y_m - y)
                
    elif profile=='Quadratic':
        if chip=='Trapezium':
            def f(y,x):
                if y=='NAN':
                    return 'NAN'
                lenY_x = (bottomLeft[1]+x*slopeBottom)-(topLeft[1]+x*slopeTop)
                y2 = lenY_0*(y - (topLeft[0] + x*slopeTop)) / lenY_x
                if flip:
                    return concentration*(y2/lenY_0)**2
                else:
                    return concentration*((lenY_0 - y2)/lenY_0)**2
        else:
            if quad=='simple':
                A = concentration/y_m**2
                def f(y,*args,**kwargs):
                    if y=='NAN':
                        return 'NAN'
                    if flip:
                        return A*y**2      
                    else:
                        return A*(y_m - y)**2    
                    
            elif quad=='whiteside':
                channel_width = y_m/9
                parabola_width = y_m - (3/2)*channel_width
                A = concentration/(parabola_width)**2
                B = channel_width
                
                def f(y,*args,**kwargs):
                    y2 = y.copy()
                    if y=='NAN':
                        return 'NAN'                    
                    if flip:
                        y2[y2 <= channel_width] = channel_width
                        y2[y2 > y_m - channel_width/2] = y_m - channel_width/2
                        return A*(y2-B)**2
                    else:
                        y2[y2 > 1000 - channel_width] = 1000 - channel_width
                        y2[y2 < channel_width/2] = channel_width/2
                        return A*(y_m - y2 - B)**2                            
            else:
                raise Exception('Unknown quad type')
    else:
        raise Exception('Unknown profile')
    
    return f 


def dataLog2FracFunc(chip,win):
    """
    Creates a function that converts position to positions as a fraction of 
    the total width for all the various chips and profiles in our experiments.
    
    Parameters
    ----------
    chip : str
        These are all experiment parameters as defined by the experiment log 
        spreadsheet.
    win : list of list of int
        The coordinates of the window for the experiment.
        
    Returns 
    -------
    f : function
        The function which converts positions to fractional position.
        Parameters
        ----------
        y : float 
            The threshold in um relative to the window edge.
        x : float
            The position in um along the x-axis of the threshold reading y. 
            Relative to the left side of the window. Important e.g. in 
            trapezium where the concentration varies in x for constant y.
            
        Returns 
        -------
        c : float {0,1}
            Decimal fraction giving how far the position is in the direction 
            minimum concentration to maximum.
    """
    if chip=='Rectangle0p7':
        y_m = 880
    elif chip=='Rectangle1':
        y_m = 1000
    elif chip=='Rectangle2p5':
        y_m = 2500
    elif chip=='Triangle':
        y_m = 1000
    elif chip=='Trapezium':
        topLeft = sorted(sorted(win)[0:2],key=lambda p:p[1])[0]
        win = [[w[0]-topLeft[0],w[1]-topLeft[1]] for w in win]
        win = win[:4]
        topLeft,bottomLeft = sorted(sorted(win)[:2],key=lambda p:p[1])
        topRight,bottomRight = sorted(sorted(win)[2:],key=lambda p:p[1])
        lenY_0 = bottomLeft[1] - topLeft[1]
        slopeTop = (topRight[1] - topLeft[1])/(topRight[0] - topLeft[0])
        slopeBottom = (bottomRight[1] - bottomLeft[1])/(bottomRight[0] - bottomLeft[0])
    else:
        raise Exception('Unknown chip type')
    
    if chip=='Trapezium':
        def f(y,x):
            if y=='NAN':
                return 'NAN'
            lenY_x = (bottomLeft[1]+x*slopeBottom)-(topLeft[1]+x*slopeTop)
            y2 = lenY_0*(y - (topLeft[0] + x*slopeTop)) / lenY_x
            return (lenY_0 - y2)/lenY_0
            return (y - (topLeft[0] + x*slopeTop)) / lenY_x
    else:
        def f(y,*args,**kwargs):
            if y=='NAN':
                return 'NAN'
            else:
                return (y_m - y)/y_m

    return f 


def makeuint8(img,quantH=0.9,quantL=False):
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
        return (255*((img-minV)/(maxV-minV))).astype('uint8')
    else:
        return (255*(img/maxV)).astype('uint8')
    
    
def segmentWatershed(img,
                     invert=False,
                     method='otsu',
                     mask=False,
                     clipMax=1,
                     clipMin=0,
                     removeSmall=True,
                     takeLargest=True,
                     takeJustMask=True,
                     cropMaskWidth=10,
                     removeTopDoubles=False):
    """
    This segments the image.
    
    Parameters
    -----------
    img : np.array
        The image to be segmented.
    invert : bool
        If True then it will high pixel values are background and low are 
        foreground.
    method = str
        Must be in {'otsu','triangle'}.
    mask : numpy.array
        An array showing which regions to include in the calculation of the 
        threshold.
    clipMax/Min : float {0-1}
        Percentile clip high and low values by.
    removeSmall : bool
        Whether to remove small connected components from the initial 
        segmentation.
    takeLargest : bool
        Whether to remove all but largest line from outline, i.e. to remove 
        small crap.
    takeJustMask : bool
        Remove any true pixels of outline that are outside of the mask, i.e. 
        you can't have a threshold outside the stimulated region!
    cropMaskWidth : bool or int
        Whether to remove cropMaskWidth% from each end of the width of the 
        mask to avoid those curved edges that can be a problem.
    removeTopDoubles : bool
        Remove the top of the watershed outline by looking above and below 
        pixels to check that signal is higher on the higher x side.
    
    Returns
    -------
    outline : np.array
        Pixel value 1 where the edge of regions are and 0 everywhere else.
    
    Notes
    -----
    Smallest segmented region area is image_area/20.
    """
    
    # make nice threshold
    pads = findPadding(img)
    img = removePadding(img.copy(),*pads)
    
    img = makeuint8(img,clipMax,clipMin)
    blur = cv.GaussianBlur(img,(9,9),0)
    if invert:
        op = cv.THRESH_BINARY_INV
    else:
        op = cv.THRESH_BINARY
    if method=='otsu':
        method = cv.THRESH_OTSU
    elif method=='triangle':
        method = cv.THRESH_TRIANGLE
    if isinstance(mask,np.ndarray):
        ret,thresh = cv.threshold(blur[mask],0,255,op+method)
        thresh = blur.copy()
        if invert:
            thresh[blur<ret] = 1
            thresh[blur>=ret] = 0
        else:
            thresh[blur<ret] = 0
            thresh[blur>=ret] = 1
    else:
        ret,thresh = cv.threshold(blur,0,255,op+method)
    kernel = np.ones((3,3),np.uint8)
    thresh = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=3)

    # remove small components
    if removeSmall:
        NComp,markers,stats,_ = cv.connectedComponentsWithStats(thresh)
        minA = np.product(markers.shape)/20
        for i in range(1,NComp):
            if stats[i,-1]<minA:
                markers[markers==i] = 0
        markers[markers>0] = 255
        markers = markers.astype('uint8')    

    # remove holes
    markers = cv.bitwise_not(markers)
    NComp,markers,stats,centroids = cv.connectedComponentsWithStats(markers)
    for i in range(1,NComp):
        if stats[i,-1]<minA:
            markers[markers==i] = 0
    markers[markers>0] = 255
    markers = markers.astype('uint8')
    markers = cv.bitwise_not(markers)  
    
    # make sure_fg and sure_bg
    kernel2 = np.ones((10,10),np.uint8)
    sure_bg = cv.dilate(markers,kernel2,iterations=3)
    sure_fg = cv.erode(markers,kernel2,iterations=3)
    unknown = cv.subtract(sure_bg,sure_fg)
    sure_fg[sure_fg>0] = 1
    sure_fg = sure_fg + 1
    sure_fg[unknown==255] = 0
    sure_fg = sure_fg.astype('int32')
    
    # calculate watershed
    img = img[:,:,np.newaxis]
    za = np.zeros(img.shape,dtype='uint8')
    img = np.concatenate((img,za,za),axis=2)
    markers = cv.watershed(img,sure_fg)
    markers[:,0] = 0 # getting rid of the border...
    markers[:,-1] = 0
    markers[0,:] = 0
    markers[-1,:] = 0
    markers[markers>0] = 0
    markers[markers<0] = 1
    markers = markers.astype('uint8')
    
    if takeLargest:
        _,markers,st,_ = cv.connectedComponentsWithStats(markers)
        biggestComp = np.argsort(st[:,-1])[-2]
        markers[markers!=biggestComp] = 0
        markers[markers==biggestComp] = 1
    
    if takeJustMask:
        if cropMaskWidth:
            cropMaskWidth = 100/cropMaskWidth
            width = mask.shape[1]
            # trick! armax stops at first True in a bool
            start = np.argmax(np.any(mask!=0,axis=0)) 
            stop = width - np.argmax(np.any(mask!=0,axis=0)[::-1])
            cropStart = int(start + (stop-start)/cropMaskWidth)
            cropStop = int(stop - (stop-start)/cropMaskWidth)
            mask[:,:cropStart] = 0
            mask[:,cropStop:] = 0
        markers[np.invert(mask)] = 0
        
    if removeTopDoubles:
        blurT = img.copy()
        blurT = cv.GaussianBlur(blurT,(201,5),0)
        
        for i in range(markers.shape[1]):
            if np.sum(markers[:,i])>1:
                ths = np.argwhere(markers[:,i]==1)
                for t in ths:
                    if t[0]-50<0:
                        t0 = 0
                    else:
                        t0 = t[0]-50
                    if t[0]+50 >= blurT.shape[0]:
                        t1 = blurT.shape[0]
                    else:
                        t1 = t[0]+50
                    if np.mean(blurT[t0:t[0],i])>np.mean(blurT[t[0]:t1,i]):
                        markers[t[0],i] = 0
    
    return np.pad(markers,((pads[0],pads[1]),(pads[2],pads[3])))


def filterBrightObjects(img,
                        thresh=False,
                        quan=0.9,
                        kernThresh=5,
                        kernRepl=21):
    """
    This is a direct copy of SessionStitcher.TData.filterBrightObjects() but 
    it acts on one image instead of a TData.It removes sparse high-signal 
    objects to aid viewing low signal of image. Any pixel where the local 
    (kernThresh) mean is above the Nth quantile intensity is replaced by the 
    mean of its neighbourhood (kernReplace) which is calculated after high 
    intensity pixels removed from this neighbourhood.
    
    Parameters
    ----------
    img : numpy.array()
        The 2D image to apply the function to
    thresh : False or int or 'otsu' or 'triangle'
        If int then it is the threshold pixel value above which pixels are 
        replaces by neighbourhood value. If false then thresh is 
        calculated from quan (see below). If 'otsu' then thresholding is done 
        with cv.threshold using the cv.THRESH_OTSU method. If 'triangle' then 
        cv again but with cv.THRESH_TRIANGLE.
    quan : float {0-1}
        The quantile intensity value that is found to set the filter 
        threshold if thresh isn't provided.
    kernThresh : int
        The size of the kernel which defines the neighbourhood which 
        decides whether a pixel will be replaced.
    kernRepl : int
        The size of the kernel which defines the neighbourhood which 
        creates the replacement pixel value.
    
    Returns
    -------
    img2 : numpy.array()
        The image with the high intensity objects filtered out.
    """
    
    if not thresh:
        thresh2 = np.quantile(img,quan)
    else: 
        thresh2 = thresh
    kern1 = cv.getGaussianKernel(kernThresh,-1)
    blur = cv.sepFilter2D(img,-1,kern1,kern1)
    if isinstance(thresh2,str):
        blur2 = makeuint8(blur,1,0)
        if thresh=='otsu':
            meth = cv.THRESH_BINARY+cv.THRESH_OTSU
        elif thresh=='triangle':
            meth = cv.THRESH_BINARY+cv.THRESH_TRIANGLE
        else:
            raise Exception('filterBrightObjects: unrecognised thresh string')
        ret,replaceMask = cv.threshold(blur2,0,255,meth)
    else:
        replaceMask = (blur > thresh2)
    kernDil = np.ones((7,7),np.uint8)
    replaceMask = replaceMask.astype('uint8')
    replaceMask = cv.dilate(replaceMask,kernDil,iterations=1)
    replaceMask = replaceMask.astype('bool')
    imHoles = blur.copy()
    imHoles[replaceMask] = 0
    rMaskNot = np.logical_not(replaceMask).astype('uint16')
    sums = cv.boxFilter(imHoles,-1,(kernRepl,kernRepl),normalize=False)
    Ns = cv.boxFilter(rMaskNot,-1,(kernRepl,kernRepl),normalize=False)
    k2 = kernRepl
    yKern = kernRepl          
    while np.any(Ns==0): # this grows the kernels until no blank spaces
        k2 += 10
        yKern += 2
        sums2 = cv.boxFilter(imHoles,-1,(k2,yKern),normalize=False)
        Ns2 = cv.boxFilter(rMaskNot,-1,(k2,yKern),normalize=False)
        sums[Ns==0] = sums2[Ns==0]
        Ns[Ns==0] = Ns2[Ns==0]
    bigBlur = sums/Ns
    img2 = img.copy()
    img2[replaceMask] = bigBlur[replaceMask]
    return img2


def fillLineGaps(im):
    """
    Completes gaps in a horizontal line so that it is unbroken from left to 
    right.
    
    Parameters
    ----------
    im : np.array
        Image with pixel value 1 for line and 0 elsewhere.
    
    Returns
    -------
    imF : np.array
        The image with gaps filled.
    """
    gapFlag = False
    gap_n = 0
    gaps = [] # each element represents a gap with [y0,x0,y1,x1] of line ends
    for i in range(im.shape[1]):
        lineQ = np.any(im[:,i])
        if not lineQ and not gapFlag:
            gapFlag = True
            gaps.append([np.argmax(im[:,i-1]),i-1])        
        elif lineQ and gapFlag:
            gapFlag = False
            gaps[gap_n] = gaps[gap_n] + [np.argmax(im[:,i]),i]
            gap_n =+ 1
    
    imF = im.copy()
    for g in gaps:
        if g[1]==-1:
            y0 = g[2]
            x0 = 0
        else:
            y0 = g[0]
            x0 = g[1]
        if len(g)==2:
            y1 = g[0]
            x1 = im.shape[1]-1
        else:
            y1 = g[2]
            x1 = g[3]
        
        for i in range(x0,x1+1):
            yi = int(y0 + (i-x0)*((y1-y0)/(x1-x0)))
            imF[yi,i] = 1
    return imF


def segmentBra(img,
               mask,
               YFPthresh,
               removeEdgeSig=True,
               threshFac=2,
               diagnostics=False,
               cropMaskWidth=False,
               clipMax=1,
               clipMin=0,
               minAFrac = 500):
    """
    A sobel-based segmentation optimised for Bra images.
    
    Parameters
    ----------
    img : np.array
        The image to segment.
    mask : np.array
        The mask of the experiment, cropped to the window.
    YFPthresh : np.array
        The image of the YFP threshold.
    removeEdgeSig : bool
        Where to remove signal that the algorithm classifies as edge rather 
        than threshold.
    threshFac : float
        Factor by which to multiple the mask outline distance transform. The 
        higher it is, the closer the signal has to be to the edge to be 
        considered edge signal.
    diagnostics : bool
        Whether to return loads of the intermediate images to help diagnose 
        problem.
    cropMaskWidth : bool or int
        Whether to remove cropMaskWidth% from each end of the width of the 
        mask to avoid those curved edges that can be a problem.
    clipMax,clipMin : float 
        Clipping of the sobel just before thresholding to help weak signals.
    minAFrac : int
        Minimum area in pixels of segmented regions as a fraction of the image 
        area. Smaller than this are removed.
        
    Returns
    -------
    markersCol : np.array
        Image with 0 as background, 1 as segmented signal to consider as edge, 
        2 as segmented signal to consider as not edge.
    markersSig : np.array
        Image of just non-edge segmented signal (value=1).
    """
    
    # find sobel and threshold it
    k = 11
    blur = cv.GaussianBlur(img, (k,k), 0) 
    sobx = cv.Sobel(blur,cv.CV_64F,1,0,ksize=k)
    soby = cv.Sobel(blur,cv.CV_64F,0,1,ksize=k)
    sob = np.sqrt(sobx**2 + soby**2)
    op = cv.THRESH_BINARY
    method = cv.THRESH_TRIANGLE
    ret,thresh = cv.threshold(makeuint8(sob,clipMax,clipMin),0,255,op+method)
    kernel = np.ones((7,17),np.uint8)
    thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations=1)
    thresh = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=1)

    # remove small components
    NComp,markers,stats,_ = cv.connectedComponentsWithStats(thresh)
    minA = np.product(markers.shape)/minAFrac
    for i in range(1,NComp):
        if stats[i,-1]<minA:
            markers[markers==i] = 0
    markers[markers>0] = 255
    markers = markers.astype('uint8') 
    
    # remove holes
    markers = cv.bitwise_not(markers)
    NComp,markers,stats,centroids = cv.connectedComponentsWithStats(markers)
    for i in range(1,NComp):
        if stats[i,-1]<minA:
            markers[markers==i] = 0
    markers[markers>0] = 255
    markers = markers.astype('uint8')
    markers = cv.bitwise_not(markers)   
    
    # delete things outside the mask
    markers[np.invert(mask)] = 0
    
    # skeletonise
    markersBin = markers.copy()
    markersBin[markersBin>0] = 1
    skeleton = skeletonize(markersBin)
    
    # fill gaps of YFPthresh
    YFPthresh = fillLineGaps(YFPthresh)        
    
    # get outline of mask
    mode = cv.RETR_EXTERNAL
    meth = cv.CHAIN_APPROX_NONE
    contour = cv.findContours(mask.astype('uint8'),mode,meth)
    outline = np.zeros(mask.shape)
    cv.drawContours(outline,contour[0],-1,1,1)
    
    # get distance transforms of YFPthresh and outline
    YthrI = np.invert(YFPthresh.astype('bool')).astype('uint8')
    threshDist = cv.distanceTransform(YthrI,cv.DIST_L2,cv.DIST_MASK_5)
    olI = np.invert(outline.astype('bool')).astype('uint8')
    outlineDist = cv.distanceTransform(olI,cv.DIST_L2,cv.DIST_MASK_5)
    
    # colour all skeleton pixels according to if they are closer to threshold 
    # line or mask outline, weight in favour of threshold by factor 2
    comp = np.where(threshDist>threshFac*outlineDist,1,2)
    skeleton = skeleton.astype('uint8')
    skeleton[skeleton.astype('bool')] = comp[skeleton.astype('bool')]
    
    # make dist. transf.s of the 2 separate colours of this coloured skeleton
    outlineSkel = skeleton.copy()
    outlineSkel[outlineSkel==2] = 0
    olsI = np.invert(outlineSkel.astype('bool')).astype('uint8')
    outlineSkelDist = cv.distanceTransform(olsI,cv.DIST_L2,cv.DIST_MASK_5)
    threshSkel = skeleton.copy()
    threshSkel[threshSkel==1] = 0
    thrskI = np.invert(threshSkel.astype('bool')).astype('uint8')
    threshSkelDist = cv.distanceTransform(thrskI,cv.DIST_L2,cv.DIST_MASK_5)
    
    # colour all seg. pix. according to which colour skel. they are closest to
    markersCol = markers.copy()
    comp2 = np.where(threshSkelDist>outlineSkelDist,1,2)
    markersCol[markersCol.astype('bool')] = comp2[markersCol.astype('bool')]
    
    # take just the signal near threshold
    markersSig = markersCol.copy()
    if removeEdgeSig:
        markersSig[markersSig==1] = 0
        markersSig[markersSig==2] = 1
    else:
        markersSig[markersSig>0] = 1
    
    if cropMaskWidth:
        cropMaskWidth = 100/cropMaskWidth
        width = mask.shape[1]
        # trick! armax stops at first True in a bool
        start = np.argmax(np.any(mask!=0,axis=0)) 
        stop = width - np.argmax(np.any(mask!=0,axis=0)[::-1])
        cropStart = int(start + (stop-start)/cropMaskWidth)
        cropStop = int(stop - (stop-start)/cropMaskWidth)
        markersSig[:,:cropStart] = 0
        markersSig[:,cropStop:] = 0
    
    
    if diagnostics:
        return [markers,skeleton,threshDist,outlineDist,markersCol,markersSig]
    else:
        return [markersCol,markersSig]