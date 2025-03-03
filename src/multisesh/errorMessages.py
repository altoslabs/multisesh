fm1 = 'We found a different number of sessions in your experiment folder to the number of fieldMaps you proved in fieldMapList.'

md1 = 'The image file {} you provided seems to have been made by an unsupported software.'

md2 = 'The image file that you are trying to find metadata for appears to be made by Andor but we couldn\'t find a separate metadata .txt file.' 

md3 = 'The time step found in the metadata of your data is not given in sec, hrs or min. We don\'t handle this yet.'

md4 = 'You can only provide customTags when madeBy is aicsimageio_compatible. If the software is known then for now we don\'t think it makes sense to add custom tags and it\'s compliated to implement.'

xf1 ='Your experiment directory can\'t be the root folder! Put it is a parent directory!' 

xf2 = 'Your experiment folder is empty!'

xf3 = 'You provided SaveXFoldPickle as a string that is just one directory level and doesn\'t end in .pkl. We don\'t know how to interpret that. Any other combination of +-directory levels and +- .pkl is fine.'

xf4 = 'This is going to error because you have provided a SaveXFoldPickle for which either the directory doesn\'t exist or you don\'t have permission. You might as well stop now.'

xf5 = 'warning: SaveXFoldPickle file exists already so xfold will not be pickled'

xf6 = 'warning: LoadFromPickle file not found so did normal __init__'

bs1 = 'To load multisesh_compressed XFolds you have to provide OriginalXFold when initiating XFold because .npz files have no metadata.'

bfml = 'Supplying FieldIDMapList as a list of str requires all sessions have the same number of fields. The list must have one string for each field as counted by Sessions.NF. Remember that there may be fewer than Session.NF fields in your Sessions TFiles if this is data analysed by multisesh and not all fields were saved. In that case you currently still need to provide the number of strings to match the original (raw microscopy data) Session.NF. To improve this we just need to write the code for supplying FieldIDMapList as list of list of str.'

st1 = 'You provided a string for the XFold\'s StartTimes attribute. We interpret that as being a path to a .txt file containing start times for each field in your experiment. We could not find a file withthe path you provided so processing is stopping. You gave: %s'

st2 = 'Warning: StartTimes file data isn\'t in correct format'

st3 = 'Warning: StartTimes fields don\'t match fieldMap fields'

stW = 'Warning: you didn\'t provide start times for the fields in your experiments. We will assume the field started at the beginning of the session that it first appeared.'

si1 = 'Warning from StitchIt: The calculated output size of the montage is not the same for all session in you experiment folder. This means either the number of montage tiles along one axis, or the montage overlap, or the image size in pixels along one dimension has changed. This situation probably isn\'t handled properly by this code.'

si2 = 'Warning: cross-correlation alignment was not used since the images were smaller than %s. Auto align was used instead.'

si3 = 'No fluorescent channels found in data. So we did noAlign.'

ae0 = 'clip must be a bool or an int (e.g. 90 for 90% clip)'

ae1 = 'You provided a string for the templateDic which we interpret as a path to a folder containing all the templates, with required fields\' templates separated into own directories. This doesn\'t seem to be the case with the templateDic that you provided.'

ae2 = 'There must only be one file in each field\'s template directory.'

ae3 = 'We couldn\'t find a template for all fields in your TData.'

ae4 = 'We couldn\'t find a channel called BF in your data. You need a bright field channel to extract from!'

ae5 = 'ExtractProfileRectangle tried to overwrite an existing csv file.'

ae6 = 'Data must not need stitching before align and extracting!'

ae7 = 'storedAlignments must be a string.'

ae8 = 'Template images must be smaller than data.'

ae9 = 'The length of the provided list of csv file names doesn\'t match the number of images in your data.'

ae10 = 'The calculated alignment has returned a shift that extends further than the input image! (i.e. you\'d end up with an extraction smaller than the template.)'

sd1 = 'You have not provided a name for the directory to save the data in. You have allowed for default naming of the directory but you haven\'t told us if this is the first TData to be saved in the batch or not. We can\'t find this from the data itself because sometimes the first time or field from the XFold will be excluded from analysis (e.g. deleted time point because it\'s blank), so you have to tell us. All processing is stopping.'

sd2 = 'You have not given a name for the directory to save in AND you have not set allowDefault=True so we have no way of deciding where to save the data. All processing is stopping.'

xy1 = 'The values you gave for the limits in TakeXYSection() were out of the range of your data array.'

cf1 = 'You gave a string to ConcatenateFiles which we interpret as a path to a folder with files in. But the string you provided didn\'t correspond to a directory that exists so it has failed.'

cf2 = 'Concatenate files encountered a problem in the format of the tifFileList you provided. They must all be paths (strings) or all TFiles.'

cf3 = 'ConcatenateFiles failed because some of the files had different dimensions. (size of time dimension can change but not the others).'

cf4 = 'Concatenate() failed either because the information on channel names couldn\'t be found or because not all the files had the same channel names. Channel name information is only understood here if it is stored in the metadata that we make (tw_chan) or in the TFile metadata.'

al1 = 'AutoLevel data not in required format. Must be TFile or TData or list of either.'

mt0 = 'You tried to makeTData with a field ID that isn\'t in your Session.'

mt1 = 'You tried to makeTData with a channel that isn\'t in your Session\'s channels.'

mt1b = 'You tried to takeSubStack with a channel that isn\'t in your TData\'s channels.'

mt2 = 'Changing the order of data while making a TData from a Session is not currently supported.'

mt3 = 'We could not find all the data points that you asked for within the session\'s TFiles.'

mt4 = 'The data points you requested in making a TData are not in the TFile.'

mt4s = 'The data points you requested in making a TData contain points that are beyond the Shape of the Session.'

mt4sfq = 'The data points you requested in making a TData contain points that, although within the Shape of the Session, were not actually found in the Session\'s TFiles. Probably this is a dataset analysed y multisesh and not all parts were analysed/saved? Maybe you have requested \'all\'.'

mt5 = 'You must makeTFiles() for the session before you can make a TData'

mt6 = 'Warning: we didn\'t find all the image data specified in the metadata in the files. You\'ve chosen to allow this so the data is left black.'

hf1 = 'No homog filters found.'

ch1 = 'WARNING: your channel name isn\'t in our channel dictionary. Please add it to the chanDic() function in XFH.py.'

ch2 = 'Your channel name isn\'t in our LUT dictionary. Please add it to the generalFunctions. LUTDic() function.'

LT1 = 'Your LUT didn\'t have the shape (n,3,256) so we couldn\'t interpret it!'

sm1 = 'Your session had a delay before imaging but metadata is in an unknown format.'

sm2 = 'The time step found in the metadata of your data is not given in sec, hrs or min. We don\'t handle this yet.'

sv1 = 'The directory that you are trying to save results to already exists and you haven\'t set overwrite=True.'

sv2 = 'Your data is not 7D and the number of pixels in the data doesn\'t match the number according to your seshD (with last 2 dimensions assumed to be NY,NX). So we don\'t know how to convert this.'

sv3 = 'Failed to set default channel names. Can only do this when you provide seshDs or your data has 3 axes (in which case axis 0 is assumed to be the channel axis).'

sv4 = 'Failed to set default channels from the seshDs that you provided. Max no. of default channels is 5.'

sv5 = 'Failed to set default channels from the data shape that you provided. Max no. of default channels is 5.'

sv6 = 'If you don\'t provide seshDs the number of axes must be 2, 3 or 7.'

it1 = 'Couldn\'t find what image type your tifffile is.'

it2 = 'Unrecognised image type (only 16 bit (or lower) recognised for now).'

cf1 = 'The tiff file {} appears to be corrupted. The dimensions in its metadata don\'t match the number of images in the file.'  

la1 = 'No startdatetimes file was provided! Can\'t do labelling.'

od1 = 'The directory you chose for output already exists and already contains at least one of the fields that you have in your OutDir. Please choose a new name.'

od2 = 'You have allowed overwrite and given a new directory name. Overwriting of an external directory is not allowed - only overwriting your OutDir is supported.'

od3 = 'Concatenate failed either because the info on channel names couldn\'t be found or because not all the files had the same channel names. Channel name info is only understood here if it is stored in the metadata that we make (tw_chan). '

od4 = 'Concatenate failed because some of the files had different dimensions. (size of time dim can change but not the others)'

od5 = 'The directory path you gave to analyse doesn\'t exist.'

od6 = 'The directory path you gave doesn\'t have a parent directory.'

od7 = 'The list of field names you gave contains names which aren\'t directories!'

od8 = 'You can\'t initiate an OutDir with no field directories inside it!'

od9 = 'You can\'t initiate an OutDir with field directories that don\'t have any tiffs in!'

zp1 = 'To use signalDetect you must provide an integer downscale factor which will increase the run speed (if>1).'

bw1 = 'Only one txt file permitted per field directory in buildWindows.'

bm1 = 'Only one tif file permitted per field directory in buildMaks.'

de1 = 'Using DeleteEmptyT on TData with > 1 fields could lead to unexpected behaviour. If any field of a time point is non-blank then no fields of that time point will be deleted (can\'t have jagged numpy arrays). But SaveData will separate the fields upon saving so you may end up with blank time points in your videos for some fields. Also, no blankedness information will be saved to the session stitcher metadata.'

co1 = 'Didn\'t find enough non-empty time points to reach the TP you provided'

bh1 = 'Can\'t resize filters, not all image sizes are equal.'

sf2 = 'You have set maskOutput to True but we couldn\'t find a Segmentation channel in the TData.'

pl1 = 'you put plotSegMask=True but channel \'Segmentation\' not found in self.Chan'

pl2 = 'you have put plotSegMask2=True but 2 Segmentation channels were not found in the tdata.'

ts1 = 'you left Segmentation=False but we couldn\'t find a channel named Segmentation in the TData. You should load a segmentation xfold with the Segmentation parameter.'

um1 = 'TData.um_2_pixels() doesn\'t handle your format yet: ' 

sw1 = 'Warning you are multiplying the size of your data by: '
sw2 = '. If you want to first reduce the image a bit then but the integer factor in initialDownsample.'

mg1 = 'tdatas must come from same Session'
mg2 = 'so far we can only merge tdatas when all axis are identical except for 1. To do more we would have to insert blank parts into the data array and then deal with those properly when merging more.'
mg3 = 'in the axis that requires merging, there some Sesh in tdata are also in tdata2. This overlap isn\'t handled yet, although it would be easy to exclude the overlapping bits.'

tr1 = 'warning: you are loading a Segmentation channel but one already exists in the tdata - the pre-existing one will be analysed'