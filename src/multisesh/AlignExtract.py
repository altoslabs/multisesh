import cv2 as cv
import numpy as np
from skimage.transform import rotate
import math
from . import generalFunctions as genF
from . import errorMessages as EM


def extractRegion(im7D,ang,shift,ysizeT,xsizeT,ysizeOut=None,xsizeOut=None):
    """ 
    This function extracts a rectangular region from all slices of a 7D image 
    that you specify with the angle and translation of the region within the 
    image, as well as the final size of the region.
    
    Parameters
    ----------
    im7D : numpy.array
        The image stack to extract from. It should have 5,6 or 7 dimensions 
        with the y,x dimensions as the last two.
    ang : int
        The angle to rotate the image by in degrees.
    shift : [int,int]
        The [y,x] location of the top left corner of the <<rotated>> region 
        within im7D. Note however that we mean rotate in the sense that the 
        image is enlarged as you rotate so that the corners aren't cropped. 
        So shift is actually the top left hand corner of this enlarged image.
    ysizeT : int
        The size of the y axis of the region you extract.
    xsizeT : int
        The size of the x axis of the region you extract.
    ysizeOut : 
        Add padding to make the final size in the y-axis this.
    xsizeOut : 
        Add padding to make the final size in the x-axis this.
    
    Notes
    ------
    Here we explain the algorithm so it is easier to follow in the code.
    We call the region that you are extracting the 'template' and the image 
    you are extracting from the 'image'. By rotating point X 'in the frame 
    of I' we mean you imagine you are rotating the image I and it's size 
    changes so that corners aren't cropped. The new coordinates of the point 
    takes into account the rotation and the resizing.
    
    1. Rotate point [0,0] by ang in the frame of the template.
    2. Shift that rotated [0,0] point by shift so it will now sits on the top 
    left corner of the template but in the image coordinates.
    3. Rotate both the image and point by -ang in the frame of the image. Now 
    the template will be sat squarely in the rotated image.
    4. Now just add ysizeT,xsizeT to the point to find the limits of the 
    region to extract.
    """
    
    shapeT = (ysizeT,xsizeT)
    
    if len(im7D.shape)==5:
        im7D = im7D[np.newaxis,np.newaxis]
    if len(im7D.shape)==6:
        im7D = im7D[np.newaxis]
    
    times,regions,montages,zslices,channels,ysizeI,xsizeI = im7D.shape
    shapeI = (ysizeI,xsizeI)
    
    # the top left corner of the template, in the template coord system:
    corner = [0,0]
    
    # now rotated as template rotates and shift added 
    # so it is aligned into the image coordinate system:
    corner = [sum(x) for x in zip(rotCoord_NoCrop(corner,*shapeT,ang),shift)]
    
    # and now rotated as image rotates so it is square:
    corner = [int(pos) for pos in rotCoord_NoCrop(corner,*shapeI,-ang)]
    
    # now arrange the im7D so it can apply all rotations at once:
    prod5D = times*regions*montages*zslices*channels
    im7D = np.swapaxes(np.swapaxes(np.reshape(im7D,(prod5D,*shapeI)),0,1),1,2)
    
    # rotate the image so template can be extracted as a square
    im7D = rotate_image(im7D,-ang)
    
    # do extraction:
    im7D = im7D[corner[0]:corner[0] + ysizeT,corner[1]:corner[1] + xsizeT]
    
    # put it back in the original shape:
    # package first 5 dimensions to stop code bloating
    shape5D = (times,regions,montages,zslices,channels)
    assert np.prod(im7D.shape) == np.prod((*shape5D,*shapeT)), EM.ae10
    im7D = np.swapaxes(np.swapaxes(im7D,1,2),0,1).reshape(*shape5D,*shapeT)
    
    if not ysizeOut:
        ysizeOut = ysizeT
    if not xsizeOut:
        xsizeOut = xsizeT 
    
    return genF.pad2Size(im7D,(ysizeOut,xsizeOut))


def findRegion(image,template,anglePrecision,maxAngleD):
    """
    This matches a template to an image by masked normalised cross-correlation.
    
    Parameters
    ----------
    image : numpy.array
        The image in which we search for a region matching the template.
    template : numpy.array
        The image section which we search for within image.
    anglePrecision : float
        The size of the increments in angle of rotation of the template.
        
    Returns
    -------
    angle : float
        The angle of rotation of the template that returns the best match.
    shift : [int,int]
        The [y,x] shift of the top left corner of the rotated template from 
        the top left corner of the image.
    
    Notes
    -----
    The template is masked to exclude reions of black produced by resizing the 
    template during rotation (the resizing is to stop cropping of the cornes 
    during rotation).
    
    It also limits the rotations to rotation where the template fully fits in 
    the image.
        
    """
    
    ysizeI,xsizeI = image.shape
    ysizeT,xsizeT = template.shape
    
    maxAngleD = findMaxAngleD(ysizeT,xsizeT,ysizeI,xsizeI,maxAngleD)

    maxValues = []
    maxPos = []

    iRangeM = int(maxAngleD//anglePrecision)
    iRange = range(-iRangeM+1,iRangeM,1)

    for i in iRange:      
        # rotate the template:
        templateR = rotate_image(template,i*anglePrecision)
        # make a mask which will exclude areas of black produced by rotation
        theMask = np.zeros(templateR.shape)
        theMask[templateR!=0] = True
        theMask = theMask.astype('float32')
        # do the cross-correlation, 10 times faster in cv!
        res = cv.matchTemplate(image,templateR,3,mask=theMask)
        maxValues.append(np.amax(res))
        maxPos.append(np.unravel_index(np.argmax(res), res.shape))
        
    maxMaxPos = maxValues.index(max(maxValues))
    theAngle = iRange[maxMaxPos]*anglePrecision
    yShift,xShift = maxPos[maxMaxPos]
    
    return (theAngle,[yShift,xShift])




def findMaxAngleD(ysizeS,xsizeS,ysizeI,xsizeI,maxAngle):
    """
    Gives the maximum angle in degrees that you can rotate imageS within 
    imageI before it hits an edge.
    If the imageS can be rotated completely within imageI then it returns 
    maxAngle (angles in degrees).
    """
    # the length of the hypotenuse of the section
    hyp = math.sqrt(ysizeS**2 + xsizeS**2)
    
    # if the hypotenuse can fit then the section can fully rotate
    if hyp <= ysizeI:
        YmaxRotD = maxAngle
    # else do the trig to find the angles
    else:
        betaYR = math.atan(ysizeS/xsizeS)
        YmaxRotR = math.asin(ysizeI/hyp) - betaYR 
        YmaxRotD = 180*YmaxRotR/math.pi
    if hyp <= xsizeI:
        XmaxRotD = maxAngle
    else:        
        betaXR = math.atan(xsizeS/ysizeS)
        XmaxRotR = math.asin(xsizeI/hyp) - betaXR
        XmaxRotD = 180*XmaxRotR/math.pi
        
    return min([XmaxRotD,YmaxRotD,maxAngle])




def rotate_image(mat, angle):
    """
    Rotates image stack using opencv.
    
    Parameters
    ----------
    mat : numpy array 
        The image stack to rotate. Shape should be (NY,NX, no. of ims).
    angle : float
        The rotation angle in degrees. +ve direction in normal cartesian sense.
        
    Returns
    -------
    rotated_mat : numpy array
        The rotated image stack. NY and NX will have changed.

    Notes
    -----
    The image is padded so that the corners of the rotated image are not 
    cropped.
    """
    
    # angle in degrees
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    if mat.ndim == 2:
        rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    elif mat.shape[2]==1:
        rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        rotated_mat = rotated_mat[:,:,np.newaxis]
    else:
        rotated_mat = [cv.warpAffine(mat[:,:,i*512:(i+1)*512], rotation_mat, (bound_w, bound_h)) for i in range(mat.shape[2]//512)]
        rotated_mat.append(cv.warpAffine(mat[:,:,(mat.shape[2]//512)*512:], rotation_mat, (bound_w, bound_h)))
        rotated_mat = np.concatenate(rotated_mat,axis=2)
                           
    return rotated_mat


def rotCoord_NoCrop(coord,ysize,xsize,angle):
    """
    Rotates a coordinate in exactly the same way as rotate_image rotates an 
    image.
    
    Parameters
    ----------
    coord : [float,float]
        The [y,x] coordinate that you want to rotate.
    ysize,xsize : int
        Rotates the coord as if it is within an image of this size. Size is 
        important becuase it effects how the image is padded during rotation 
        so that corners are still included.
    angle : float
        The rotation angle in degrees. +ve direction in normal cartesian sense.
    """
    
    
    image_center = (xsize/2, ysize/2)
    
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
    
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    
    bound_x = int(ysize * abs_sin + xsize * abs_cos)
    bound_y = int(ysize * abs_cos + xsize * abs_sin)
    
    rotation_mat[0, 2] += bound_x/2 - image_center[0]
    rotation_mat[1, 2] += bound_y/2 - image_center[1]

    coordC = [coord[1],coord[0],1]

    coordC  = np.matmul(rotation_mat,coordC).round().astype('int')
    
    return [coordC[1],coordC[0]]