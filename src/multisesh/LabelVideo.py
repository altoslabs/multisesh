import cv2
import os
import re
import datetime
import numpy as np
import math

from . import generalFunctions as genF



def moment2Str(moment,roundM=30,roundMQ=True,style='hh:mm'):
    """
    Takes a datetime object and converts it to a string. 
    
    Parameters
    ----------
    roundM : int
        The value that you round to if roundMQ is True.
    roundMQ : bool
        Whether to round.
    style : str {'hh:mm','mm:ss'}
        Whether to print as hh:mm or mm:ss.
    
    You can do rounding in minutes by putting roundMQ=True and 
    roundM to the minutes you want to round to.
    """
    totsec = moment.total_seconds()
    if style == 'hh:mm':
        h = totsec // 3600
        m = (totsec%3600) // 60
        if roundMQ:
            m = math.floor(m/roundM)*roundM
        momentString = str(int(h)).zfill(2)+':'+str(int(m)).zfill(2)
    elif style == 'mm:ss':
        m = totsec // 60
        s = totsec%60
        momentString = str(int(m)).zfill(2)+':'+str(int(s)).zfill(2)        
        
    return momentString



def addTimeLabel(im,time,labelScale=None):
    """
    This uses opencv to add a string to the bottom right-hand corner.
    
    Parameters
    ----------
    im : numpy array
        Your image. Must be uint16.
    time : str
        The string that you want to print on the image.
    labelScale : int
        If provided then the height of the text will be roughly the size of 
        your image in the y-axis divided by labelScale. If not provided then 
        it automatically calculates a size. The idea is that it imagines the 
        image scaled to fit a presentation slide (which all have aspect ratio 
        of 1.75) and makes the text height 10 times smaller than the y-size 
        of the slide.
    
    Notes
    ------
    It assumes your image is uint16 and sets the colour to the maximum. 
    Labelscale is how manys times bigger the ysize is 
    compared to the text height, 10 is usually good.
    """
    # set and get parameters:
    ysize,xsize = np.shape(im)
    # fontFace 
    fF = 2
    # fontScale 
    fS = 4
    # theColor 
    c = 65535
    # lineType
    lT = 8
    # thickness
    th = 4
    
    # get a text size
    textSize = cv2.getTextSize(time,fontFace=fF,fontScale=fS,thickness=th)   
    
    # update the fontScale according to labelScale
    if not labelScale:
        if xsize/ysize<=1.75:
            fS = fS*((ysize/10)/textSize[0][1])
        else:
            fS = fS*((xsize/(10*1.75))/textSize[0][1])
    else:
        fS = fS*((ysize/labelScale)/textSize[0][1])
        
    # update the text size
    textSize = cv2.getTextSize(time,fontFace=fF,fontScale=fS,thickness=th)
    
    # set the position of the text to be kind of bottom right
    xorg = int(xsize - textSize[0][0] - xsize/200)
    yorg = int(ysize - textSize[1])
    
    # new thickness
    th = int(textSize[0][1]/10)
    
    # put the text on the image (note how this uses im being mutable)
    cv2.putText(im,time,org=(xorg,yorg),fontFace=fF,fontScale=fS,
                color=c,thickness=th,lineType=lT)
    return
    
