import os
import numpy as np
import pandas as pd
import cv2 as cv
import tifffile
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage import filters

from . import generalFunctions as genF

def measure_blobs(tdata,
                  tindices,
                  image,
                  mask,
                  size,
                  one_threshold=None,
                  saveBlobMask=None,
                  saveBlobEdgeMask=None,
                  thickness=2):
    """
    This detects blobs of specified size in the region and returns some 
    measurements. I.e. use for an image which has (possibly overlapping) blobs 
    of similar size.

    Use in RegionProps like this: 
    fun = [measure_blobs,
            [image_chan,mask_chan],
            [size,one_threshold,saveBlobMask,saveBlobEdgeMask,thickness]]

    Parameters
    ----------
    tdata : ms.TData
        The full TData that you took this region from. 
    tindices : (int,int,int,int,list of int,int)
        The (t,f,m,z,chans,lab) indices of the tdata from which the region 
        ('image') was taken and the label of that region ('lab'). Except chans 
        is a list of indices corresponding to tdata.Chan. In this function it should have two elements, 
        the first is the channel index of image and the second is the channel 
        index of the segmentation.
    image : numpy array
        An array of size (NY,NX) with the image data, cropped around the 
        region you are analysing.        
    mask : numpy array
        The same as image_data but a mask of the region you are analysing.
    size : float
        A rough size of blob in um.
    one_threshold : None or float or int or list
        The threshold to use if applying one threshold to everything. If it is 
        a list then it is assumed to be a threshold for each field in the 
        Session.
    saveBlobMask : None or str
        If you provide a string it will save the blob mask to a folder 
        specified by this string. Within that folder it will be saved in a 
        directory tree of SessionName/T/F/M/Z as done by 
        XFold.SaveDataFrameCrops(). The channel name is added to the file name 
        and will be Blob_Segmentation. A new entry in the returned dictionary 
        will record this filepath, it has key 'out path ch_Blob_Segmentation'.
    saveBlobEdgeMask : None or str
        Same as saveBlobMask but it converts the blob_mask to edges first.
    thickness : int
        The thickness of the edges in the saved BlobEdgeMask.

    Returns
    -------
    out_dic : dict
        Keys are measurement names (i.e. column names in resulting dataframe), 
        values are the measurements. '_Chan' is added to each one where Chan 
        is the name of the channel from which you are measureing blobs. This 
        is to avoid repeated column names if you run the function more than 
        once on different channels of the same tdata.
        The final measurements are:
            blob_otsu_threshold_Chan :
            blob_one_threshold_Chan : 
            blob_specified_size_Chan : 
            number_of_blobs_Chan : 
            blob_mask_Chan : 
            blob_area_mean_pixels_Chan : 
            blob_area_mean_um2_Chan : 
            blob_area_std_pixels_Chan : 
            mean_blob_intensity_at_cen_Chan  
            std_blob_intensity_at_cen_Chan : 
            blob_mean_intensity_Chan : 
            blob_std_intensity_Chan : 
            out_path_ch_Chan_Blob_Segmentation :
    """

    t,f,m,z,chans,lab = tindices
    T,F,M,Z = (tdata.SeshT[t],tdata.SeshF[f],tdata.SeshM[m],tdata.SeshZ[z])
    im_chan_str = tdata.Chan[chans[0]]
    
    all_props = ('area',
                'label',
                'centroid',
                'centroid_local')  
    
    # convert size to pixels
    size = size/(tdata.ParentSession.pixSizeX_um)
    
    mask = mask==lab
    
    # delete data outside mask
    image[np.invert(mask)] = 0

    # make Laplacian of Gaussian
    LoG = cv.Laplacian(image,cv.CV_32F)
    LoG = cv.GaussianBlur(LoG,(0,0),size/9)  

    # otsu the blobs
    blur = cv.GaussianBlur(image,(0,0),size/9)
    if isinstance(one_threshold,(int,float)):
        thr = one_threshold
    elif isinstance(one_threshold,list):
        thr = one_threshold[F]
    else:
        thr = filters.threshold_otsu(blur[mask])
    blob_mask = blur > thr

    # get peaks
    coords = peak_local_max(-LoG,
                            footprint=np.ones((int(size/2),int(size/2))),
                            labels=blob_mask)    
    
    markers = np.zeros(image.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers = measure.label(markers)
    water_labels = watershed(LoG, markers, mask=blob_mask)

    regions = measure.regionprops_table(water_labels,properties=all_props)
    df = pd.DataFrame(regions)

    water_labs = np.unique(water_labels)[1:]
    intensities = []
    for wlab in water_labs:
        intensities.append(np.mean(image[water_labels==wlab]))

    # centroids as indices
    cenY_i = df['centroid-0'].round(0).astype(int)
    cenX_i = df['centroid-1'].round(0).astype(int)
    cen_intensities = blur[cenY_i,cenX_i]

    out_dict = {'blob_otsu_threshold_'+im_chan_str:thr,
                'blob_one_threshold_'+im_chan_str:one_threshold,
                'blob_specified_size_'+im_chan_str:size*tdata.ParentSession.pixSizeX_um,
                'number_of_blobs_'+im_chan_str:len(df),
                'blob_area_mean_pixels_'+im_chan_str:df['area'].mean(),
                'blob_area_mean_um2_'+im_chan_str:df['area'].mean()*(tdata.ParentSession.pixSizeX_um**2),
                'blob_area_std_pixels_'+im_chan_str:df['area'].std(),
                'blob_area_std_um2_'+im_chan_str:df['area'].std()*(tdata.ParentSession.pixSizeX_um**2),
                'mean_blob_intensity_at_cen_'+im_chan_str:np.mean(cen_intensities),
                'std_blob_intensity_at_cen_'+im_chan_str:np.std(cen_intensities),
                'blob_mean_intensity_'+im_chan_str:np.mean(intensities),
                'blob_std_intensity_'+im_chan_str:np.std(intensities)
               }
    
    if saveBlobMask:
        if os.path.split(saveBlobMask)[0]=='':
            saveBlobMask = os.path.join(tdata.ParentSession.ParentXFold.XPathP,
                                        saveBlobMask)
        mask_path = os.path.join(saveBlobMask,
                            tdata.ParentSession.Name,
                            'T'+str(T).zfill(4),
                            'F'+str(F).zfill(4),
                            'M'+str(M).zfill(4),
                            'Z'+str(Z).zfill(4),
                            'L'+str(lab).zfill(4)+'_C'+im_chan_str+'.tif')
        
        out_dict['out_path_ch_'+im_chan_str+'Blob_Segmentation'] = mask_path

        if not os.path.isdir(os.path.split(mask_path)[0]):
            os.makedirs(os.path.split(mask_path)[0])
            
        tifffile.imwrite(mask_path,water_labels)

    if saveBlobEdgeMask:
        if os.path.split(saveBlobEdgeMask)[0]=='':
            saveBlobMask = os.path.join(tdata.ParentSession.ParentXFold.XPathP,
                                        saveBlobEdgeMask)
        mask_path = os.path.join(saveBlobEdgeMask,
                            tdata.ParentSession.Name,
                            'T'+str(T).zfill(4),
                            'F'+str(F).zfill(4),
                            'M'+str(M).zfill(4),
                            'Z'+str(Z).zfill(4),
                            'L'+str(lab).zfill(4)+'_C'+im_chan_str+'.tif')
        
        out_dict['out_path_ch_'+im_chan_str+'Blob_Edge_Segmentation'] = mask_path    
        
        if not os.path.isdir(os.path.split(mask_path)[0]):
            os.makedirs(os.path.split(mask_path)[0])
            
        tifffile.imwrite(mask_path,genF.labelMask2Edges(water_labels,thickness=thickness))
            
    return out_dict



def measure_channel_intensity(tdata,tindices,im,mask):
    """
    It returns the mean, max and std intensity of im after it has been masked 
    by mask.

    Use in RegionProps like this: 
    fun = [measure_channel_intensity,
            [im_chan,mask_chan],
            []]

    Parameters
    ----------
    tdata : ms.TData
        The full TData that you took this region from. 
    tindices : (int,int,int,int,list of int,int)
        The (t,f,m,z,chans,lab) indices of the tdata from which the region 
        ('im') was taken and the label of that region ('lab'). Except chans 
        is a list of indices corresponding to tdata.Chan. In this function it 
        should have two elements, the first is the channel index of image and 
        the second is the channel index of the segmentation.
    im : numpy array
        An array of size (NY,NX) with the image data, cropped around the 
        region you are analysing.        
    mask : numpy array
        The same as image_data but a mask of the region you are analysing.     

    Returns
    -------
    out_dict : dict of len 3
        Column names:
        'mean_of_'+im_chan_str+'_in_'+seg_chan_str 
        'max_of_'+im_chan_str+'_in_'+seg_chan_str
        'std_of_'+im_chan_str+'_in_'+seg_chan_str
    """

    t,f,m,z,chans,lab = tindices
    im_chan_str = tdata.Chan2[chans[0]]
    seg_chan_str = tdata.Chan2[chans[1]]

    mean_name = 'mean_of_'+im_chan_str+'_in_'+seg_chan_str
    max_name = 'max_of_'+im_chan_str+'_in_'+seg_chan_str
    std_name = 'std_of_'+im_chan_str+'_in_'+seg_chan_str
    
    out_dict = {}  

    mask = mask==lab
    
    if (not np.product(im.shape)>1) or (np.sum(mask)==0):
        out_dict[mean_name] = np.nan
        out_dict[max_name] = np.nan
        out_dict[std_name] = np.nan
    else:
        out_dict[mean_name] = np.mean(im[mask])
        out_dict[max_name] = np.max(im[mask])
        out_dict[std_name] = np.std(im[mask])

    return out_dict


def measure_channels_intensity_ratio(tdata,tindices,im1,im2,mask):
    """
    It returns the mean, max and std intensity of im1 divied by im2 (pixel by 
    pixel) after it has been masked by mask.

    Use in RegionProps like this: 
    fun = [measure_channels_intensity_ratio,
            [im1_chan,im2_chan,mask_chan],
            []]

    Parameters
    ----------
    tdata : ms.TData
        The full TData that you took this region from. 
    tindices : (int,int,int,int,list of int,int)
        The (t,f,m,z,chans,lab) indices of the tdata from which the region 
        ('image') was taken and the label of that region ('lab'). Except chans 
        is a list of indices corresponding to tdata.Chan. In this function it 
        should have two elements, the first is the channel index of image and 
        the second is the channel index of the segmentation.
    image : numpy array
        An array of size (NY,NX) with the image data, cropped around the 
        region you are analysing.        
    mask : numpy array
        The same as image_data but a mask of the region you are analysing.     

    Returns
    -------
    out_dict : dict of len 3
        Column names:
        'mean_of_'+im1_chan_str+'_over_'+im2_chan_str 
        'max_of_'+im1_chan_str+'_over_'+im2_chan_str
        'std_of_'+im1_chan_str+'_over_'+im2_chan_str
    """

    t,f,m,z,chans,lab = tindices
    im1_chan_str = tdata.Chan2[chans[0]]
    im2_chan_str = tdata.Chan2[chans[1]]
    seg_chan_str = tdata.Chan2[chans[2]]

    mean_name = 'mean_of_'+im1_chan_str+'_over_'+im2_chan_str
    max_name = 'max_of_'+im1_chan_str+'_over_'+im2_chan_str
    std_name = 'std_of_'+im1_chan_str+'_over_'+im2_chan_str
    
    out_dict = {}  

    mask = mask==lab
    
    if (not np.product(im1.shape)>1) or (np.sum(mask)==0):
        out_dict[mean_name] = np.nan
        out_dict[max_name] = np.nan
        out_dict[std_name] = np.nan
    else:
        ratio = im1[mask]/im2[mask]
        out_dict[mean_name] = np.mean(ratio)
        out_dict[max_name] = np.max(ratio)
        out_dict[std_name] = np.std(ratio)

    return out_dict    


def another_mask_region_props(tdata,tindices,mask):
    """
    TData.RagionProps() automatically gives you all the normal regionProps 
    measurements of the main mask that you give it (i.e Segmentation=x, this 
    also defines the crop that is sent to this function). Use this if you 
    want to get those same measurements of a different mask.

    Note how the combination of tdata.RegioProps with this is a bit awkward: 
    we want to use skimage.measure.regionprops_table() here so things are 
    identical to tdara.RegionProps. But this expects many labelled objects 
    forming a table with many rows - but we are passing one object at a time 
    in a slow python loop. Probably a much better way but for now we get rid 
    on the arrays of (one!) value in the last line here

    Parameters
    ----------
    tdata : ms.TData
        The full TData that you took this region from. 
    tindices : (int,int,int,int,list of int,int)
        The (t,f,m,z,chans,lab) indices of the tdata from which the region 
        ('image') was taken and the label of that region ('lab'). Except chans 
        is a list of indices corresponding to tdata.Chan. In this function it 
        should have two elements, the first is the channel index of image and 
        the second is the channel index of the segmentation.       
    mask : numpy array
        An array of size (NY,NX) - the mask of the region you are analysing.   

    Use as element in RegionProps 'fun'
    -----------------------------------
    fun = [ms.RegionPropFunctions.another_mask_region_props,
            [mask_chan],
            []]
            
    RegionProps 'fun' parameters
    ----------------------------
    
    
    
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

    t,f,m,z,chans,lab = tindices

    mask = (mask==lab).astype('uint8')    
    
    out_dict = measure.regionprops_table(mask,properties=all_props)

    chan_name = tdata.Chan2[chans[0]]
    
    out_dict = {k+'--'+chan_name:v[0] for k,v in out_dict.items() if k not in not_needed}

    return out_dict



