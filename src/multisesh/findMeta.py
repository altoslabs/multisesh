"""
These are functions that extract metadata from an image file. 
Function work for all supported softwares that may have made the image 
data. 
"""

import os
import re
import numpy as np
from datetime import timedelta,datetime
import tifffile
from aicsimageio import AICSImage
import xml.etree.ElementTree as ET

if __package__ == '':
    import generalFunctions as genF
    import errorMessages as EM
else:
    from . import generalFunctions as genF
    from . import errorMessages as EM
    
imTypeReg = re.compile(r'Type : (\d+) bit') # image data type
NTReg = re.compile(r'Time : (\d*)') # number of time points
NFReg = re.compile(r'XY : (\d*)') # number of fields
# number of montage tiles, can extract x (group(1)) and y (group(2))
NMReg = re.compile(r'Montage Positions - \(\d* \( (\d*) by (\d*) \)\)')
NMReg2 = re.compile(r'Montage : (\d*)')
NZReg = re.compile(r'Z : (\d*)') # number of z-slices
NCReg = re.compile(r'Wavelength : (\d*)') # number of channels
NYReg = re.compile(r'y : (\d+) ') # y size
NXReg = re.compile(r'x : (\d+) ') # x size
pixSizeRegY = re.compile(r'y : \d+ \* (\d+.\d+)') # pixel size
pixSizeRegX = re.compile(r'x : \d+ \* (\d+.\d+)') # pixel size
pixUnitReg = re.compile(r'x : \d+ \* \d+.\d+ : (\w+)') # unit of pixel size
DTReg = re.compile(r'Repeat T - \d+ times? \((\d+) ([\s\w-]+)\)')
DZReg = re.compile(r'Repeat Z - (\d*) um in') # z-slice thickness
chanReg = re.compile(r'\t(?:Move )?Channel - (\w+)\n') # can get all names of chans
OlapReg = re.compile(r'Montage=(Region|Edge)\tOverlap (\d*)') 
startTimeReg = re.compile(r'Time=(\d\d:\d\d:\d\d)\n\[Created End\]')
startDateReg = re.compile(r'\[Created\]\nDate=(\d\d/\d\d/\d\d\d\d)')
startMomRegMM = re.compile(r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)')
# delay regex... i.e. time b/w start of protocol and start of imaging
delayReg = re.compile(r'Delay - (\d+) (\w+)')


def madeBy(tp):
    """
    This takes a tif path and determines what software the file was made by.
    
    Returns 
    --------
    Incucyte
    Andor
    Micromanager
    Exception (if not one of the above)
    """

    if tp[-4:]=='.lif':
        return 'Leica'
    elif tp[-4:]=='.czi':
        ## havent written allSeshMeta for Zeiss yet
        #return 'aicsimageio_compatible'
        return 'Zeiss'
    elif tp[-4:]=='.nd2':
        return 'Nikon'
    elif tp[-4:]=='.npz':
        return 'multisesh_compressed'      
    
    try:
        with tifffile.TiffFile(tp) as tif:
            m = tif.fluoview_metadata
            I = tif.imagej_metadata
            m2 = tif.micromanager_metadata
            
            # checking if it is made by incucyte
            for tag in tif.pages[0].tags:
                if 'Incucyte' in str(tag):
                    return 'Incucyte'
                if 'PerkinElmer' in str(tag):
                    return 'Opera'
                if 'Revvity' in str(tag):
                    return 'Opera'                    
            if I and 'tdata_meta_data' in I.keys():
                return 'multisesh'
            if I:
                # check if it's one we've saved to imagej
                I = 'tw_nt' in I.keys()                
            else:
                I = False
                
            if m or I:
                return 'Andor'
            elif m2:
                return 'MicroManager'
            else:
                try:
                    img = AICSImage(tp)
                    return 'aicsimageio_compatible'
                except:
                    raise Exception(EM.md1.format(tp))
    except tifffile.TiffFileError as TFE:
        print('Critical problem with tiff file: ',tp)
        raise TFE    
    

def allSeshMeta(tp,tps,customTags={},silenceWarnings=False):
    """
    gets all the main meta from an image filepath and returns in a dictionary
    (or a list of dictionaries if there are multiple sessions in one image file)
    
    Parameters
    -----------
    tp : str
        file path to the image file in question.
    tps : list of str
        the files paths of all tiff files in the session of tp
        (some softwares need this to figure out metadata)
    customTags : dict
        See XFold parameters.
    silenceWarnings : bool
        Whether to silence warnings.
    
    Accepted formats
    ----------------
    incucyte : ....
    
    Andor : ....
    
    Micromanager : ....

    Leica : ...

    Zeiss : ...

    Operra : ... 

    Metadata items it retrieves (each key : value of returned dict)
    ----------------------------------------------------------------
    NT,NF,NM,NZ,NC,NY,NX : int
        The number of elements along the specifies axis.
    ZStep : float
        The distance between z-slices in um.  
    .....    
    
    """
    
    allMeta = {}
    allMeta['aics_RGB'] = False
    
    if tp[-4:]=='.lif':
        allMeta['madeBy'] = 'Leica'    
    elif tp[-4:]=='.czi':
        allMeta['madeBy'] = 'Zeiss'     
    elif tp[-4:]=='.nd2':
        allMeta['madeBy'] = 'Nikon'   
    elif tp[-4:]=='.npz':
        allMeta['madeBy'] = 'multisesh_compressed'          
    else:  
        try:
            # set madeBy and extract everything we'll need from the file
            with tifffile.TiffFile(tp) as tif:
                m = tif.fluoview_metadata
                I = tif.imagej_metadata
                m2 = tif.micromanager_metadata
                
                # this is for checking if it is made by incucyte
                incucyte = False
                opera = False
                for tag in tif.pages[0].tags:
                    if 'PerkinElmer' in str(tag):
                        opera = True
                        break
                    if 'Revvity' in str(tag):
                        opera = True
                        break                             
                    if 'Incucyte' in str(tag):
                        incucyte=True
                        break
                if I and 'tdata_meta_data' in I.keys():
                    allMeta['madeBy'] = 'multisesh'
                elif I and 'tw_nt' in I.keys():
                    # check if it's one we've saved to imagej
                    allMeta['madeBy'] = 'Andor'
                elif m:
                    allMeta['madeBy'] = 'Andor'
                elif incucyte:
                    allMeta['madeBy'] = 'Incucyte'
                    aTag = tif.pages[0].tags.values()
                    incu_meta = tif.pages[-1].tags.values()[-1].value
                elif opera:
                    allMeta['madeBy'] = 'Opera'
                elif m2:
                    allMeta['madeBy'] = 'MicroManager'
                    t = tif.pages[0].tags['MicroManagerMetadata'].value
                    m = m2                
                else:
                    try:
                        aics = AICSImage(tp)
                        allMeta['madeBy'] = 'aicsimageio_compatible'
                    except:
                        raise Exception(EM.md1.format(tp))                
        except tifffile.TiffFileError as TFE:
            print('Critical problem with tiff file: ',tp)
            raise TFE
    
    if allMeta['madeBy']=='Leica':
        """
        the metadata for Leica is mainly in an xml that is put in 
        aics.metadata. It has an 'alternating hierarchical' structure for 
        which a level can be all 'Elements' but within an Element there is 
        'Data', 'Memory' and 'Children'. Data and Memory have some attributes 
        but no sub-elements. Children have sub-elements and they are all 
        'Elements'. It seems to make sense to call the first level of children 
        the Sessions and the next level of children the Fields.
        """

        allMeta = []

        aics = AICSImage(tp)
        
        sessions = aics.metadata.find('Element').find('Children')
        for ii,s in enumerate(sessions):

            aM = {}
            aM['aics_RGB'] = False
            aM['Name'] = os.path.split(tp)[1] + ' - ' + s.attrib['Name']
            # this remembers original ordering in file for if buildSessions changes Session order
            aM['FileSessionN'] = ii 
            aM['madeBy'] = 'Leica'
            aM['metadata'] = s
            fields = s.find('Children')
            aM['NF'] = len(fields)
            dims = []
            chanNs = []
            times = []
            bits = []
            XYLs = []
            startT = datetime(1601, 1, 1, 0, 0, 0)
            if len(fields)==1:
                imD = s.find('Data').find('Image').find('ImageDescription')
                dims0 = imD.find('Dimensions')
                XYLs.append(tuple([float(d.attrib['Length']) for d in dims0]))
                chans = imD.find('Channels')
                bits.append([int(c.attrib['Resolution']) for c in chans])
                chanNs.append(tuple([c.attrib['LUTName'] for c in chans]))
                dims.append(tuple([int(d.attrib['NumberOfElements']) for d in dims0]+[len(chans)]))
                time_stamps = s.find('Data').find('Image').find('TimeStampList')        
                time = int(time_stamps.text.split()[0],16)/10000000
                times.append(startT + timedelta(seconds=time))
            else:
                for f in fields:
                    imD = f.find('Data').find('Image').find('ImageDescription')
                    dims0 = imD.find('Dimensions')
                    XYLs.append(tuple([float(d.attrib['Length']) for d in dims0]))
                    chans = imD.find('Channels')
                    bits.append([int(c.attrib['Resolution']) for c in chans])
                    chanNs.append(tuple([c.attrib['LUTName'] for c in chans]))
                    dims.append(tuple([int(d.attrib['NumberOfElements']) for d in dims0]+[len(chans)]))
                    time_stamps = f.find('Data').find('Image').find('TimeStampList')        
                    time = int(time_stamps.text.split()[0],16)/10000000
                    times.append(startT + timedelta(seconds=time))

            if not len(tuple(set(dims)))==1:
                raise Exception('uh oh diff dims in diff fields of same session')
            else:
                dims = dims[0]
            if not len(tuple(set(XYLs)))==1:
                raise Exception('uh oh diff pix sizes in diff fields of same session')
            else:
                XYLs = XYLs[0]                
            if not len(tuple(set(chanNs)))==1:
                raise Exception('uh oh diff channel names in diff fields of same session')  
                
            aM['NY'] = dims[0]
            aM['NX'] = dims[1]
            aM['NZ'] = dims[2]
            aM['NC'] = dims[3]
        
            # haven't implented multiple T yet!
            aM['NT'] = 1
            aM['TStep'] = timedelta(hours=999)

            # haven't found this yet
            aM['ZStep'] = 1

            # montage not implemented yet but there is something called TileScanInfo so should be easy
            aM['NM'] = 1
            aM['MOlap'] = 0
            aM['NMX'] = 1
            aM['NMY'] = 1
            aM['FieldNMYs'] = [aM['NMY'] for f in range(aM['NF'])]
            aM['FieldNMXs'] = [aM['NMX'] for f in range(aM['NF'])]
            aM['MontageOrder'] = None
            aM['TilePosX'] = None
            aM['TilePosY'] = None              

            aM['Chan'] = chanNs[0]
            
            allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
            aM['Shape'] = tuple([aM[k] for k in allAxes])

            aM['pixSizeY'] = XYLs[0]/dims[0]
            aM['pixSizeX'] = XYLs[1]/dims[1]
            aM['pixUnit'] = 'm'  

            aM['startMom'] = times[0]

            aM['imType'] = bits[0][0]

            aM['SessionN'] = -1 # only used for multisesh

            allMeta.append(aM)
            
    elif  allMeta['madeBy']=='multisesh':
        
        mb = allMeta['madeBy']
        allMeta = genF.meta_str_2_dict(I['session_meta_data'])
        allMeta2 = genF.meta_str_2_dict(I['tdata_meta_data'])
        allMeta['madeByOriginal'] = allMeta['madeBy']
        allMeta['madeBy'] = mb
        allMeta['FileSessionN'] = None
        allMeta['SessionN'] = allMeta2['SessionN']
        
    elif  allMeta['madeBy']=='multisesh_compressed':
        allMeta['Chan'] = ['Segmentation']
        allMeta['NC'] = 1
        allMeta['NZ'] = 1
        
    elif allMeta['madeBy']=='Nikon':
        
        img = AICSImage(tp)
        
        allMeta['metadata'] = img.metadata

        allMeta['FileSessionN'] = None
        
        allMeta['NC'] = len(img.metadata['metadata'].channels)
        allMeta['Chan'] = [c.channel.name for c in img.metadata['metadata'].channels]
        allMeta['NY'] = img.metadata['attributes'].heightPx
        allMeta['NX'] = img.metadata['attributes'].widthPx
        
        allMeta['pixSizeY'] = img.metadata['metadata'].channels[0].volume.axesCalibration[0]
        allMeta['pixSizeX'] = img.metadata['metadata'].channels[0].volume.axesCalibration[1]
        allMeta['pixSizeZ'] = img.metadata['metadata'].channels[0].volume.axesCalibration[2]
        allMeta['pixUnit'] = 'um'  
        
        time_type = "<class 'nd2.structures.TimeLoop'>"
        fields_type = "<class 'nd2.structures.XYPosLoop'>"
        zstack_type = "<class 'nd2.structures.ZStackLoop'>"
        
        types = [str(type(ob)) for ob in img.metadata['experiment']]

        if time_type in types:
            time_index = types.index(time_type)
            allMeta['NT'] = img.metadata['experiment'][time_index].count
            allMeta['TStep'] = timedelta(milliseconds=img.metadata['experiment'][0].parameters.periodMs)
        else:
            allMeta['NT'] = 1
            allMeta['TStep'] = timedelta(hours=999)

        if fields_type in types:
            fields_index = types.index(fields_type)
            allMeta['NF'] = img.metadata['experiment'][fields_index].count
        else:
            allMeta['NF'] = 1
            
        if zstack_type in types:
            zstack_index = types.index(zstack_type)
            allMeta['NZ'] = img.metadata['experiment'][zstack_index].count
            allMeta['ZStep'] = img.metadata['experiment'][zstack_index].parameters.stepUm
        else:
            allMeta['NZ'] = 1
            allMeta['ZStep'] = 1
        
        allMeta['NM'] = 1
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        allMeta['MOlap'] = 0
        # for nikon we haven't done montages yet
        allMeta['NMX'] = 1
        allMeta['NMY'] = 1
        # don't think you can do varying montage size for Nikon? Need to check though
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]
        allMeta['MontageOrder'] = None
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None    

        allMeta['startMom'] = datetime.strptime(img.metadata['text_info']['date'], '%d/%m/%Y %H:%M:%S')       
        allMeta['imType'] = img.metadata['attributes'].bitsPerComponentInMemory

        allMeta['SessionN'] = -1 # only used for multisesh

        
    elif allMeta['madeBy']=='Zeiss':

        aics = AICSImage(tp)

        allMeta['FileSessionN'] = None
        
        # there's no big txt file, hopefully that's ok
        allMeta['metadata'] = aics.metadata

        meta_xml = aics.metadata.find('Metadata').find('Information')
        try: 
            dt = meta_xml.find('Image').find('AcquisitionDateAndTime').text
            dt = dt.split(".")[0]
            dt_format = '%Y-%m-%dT%H:%M:%S'
            allMeta['startMom'] = datetime.strptime(dt,dt_format) 
        except AttributeError:
            try:
                dt = meta_xml.find('Document').find('CreationDate').text
                dt_format = '%Y-%m-%dT%H:%M:%S'
                allMeta['startMom'] = datetime.strptime(dt,dt_format) 
            except AttributeError:
                # unknown startMom shouldn't be set to now because we sort 
                # first by startMom so don't want to impose artificial 
                # ordering here. Always use this datetime
                allMeta['startMom'] = datetime(1818,5,5,0,0)
                if not silenceWarnings:
                    print('warning: no start time found, set to default') 
        
        # haven't had example of time data in czi yet
        allMeta['TStep'] = timedelta(hours=999)

        # haven't found this yet
        allMeta['ZStep'] = 1        
        
        allMeta['Chan'] = []
        for chan in meta_xml.find('Image').find('Dimensions').find('Channels'):
            try:
                allMeta['Chan'].append(chan.find('Fluor').text)
            except AttributeError:
                try:
                    allMeta['Chan'].append(str(int(float(chan.find('ExcitationWavelength').text))))
                except AttributeError:
                    try:
                        allMeta['Chan'].append(chan.attrib['Name'])
                    except AttributeError:
                        allMeta['Chan'].append('Name not found')

        if meta_xml.find('Image').find('SizeT'):
            allMeta['NT'] = int(meta_xml.find('Image').find('SizeT').text)
        else:
            allMeta['NT'] = 1
        
        # for czi I haven't had multi F example yet
        allMeta['NF'] = 1

        # for czi I haven't had multi M example yet
        allMeta['NM'] = 1

        
        allMeta['NZ'] = int(meta_xml.find('Image').find('SizeZ').text)
        try:
             allMeta['NC'] = int(meta_xml.find('Image').find('SizeC').text)
        except:
             allMeta['NC'] = len(meta_xml.find('Image').find('Dimensions').find('Channels').findall('Channel'))        
        allMeta['NY'] = int(meta_xml.find('Image').find('SizeY').text)
        allMeta['NX'] = int(meta_xml.find('Image').find('SizeX').text)

        # haven't bothered searching for this in czi yet
        allMeta['imType'] = 16   
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        # haven't done this in czi yet
        allMeta['MOlap'] = False
        allMeta['NMY'] = 1
        allMeta['NMX'] = 1
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None        
        allMeta['MontageOrder'] = False

        for child in aics.metadata.find('Metadata').find('Scaling').find('Items'):
            if child.attrib['Id']=='X':
                xsize = float(child.find('Value').text)
            elif child.attrib['Id']=='Y':
                ysize = float(child.find('Value').text)        
        allMeta['pixSizeY'] = ysize
        allMeta['pixSizeX'] = xsize
        allMeta['pixUnit'] = 'm'      

        allMeta['SessionN'] = -1 # only used for multisesh
    
    elif allMeta['madeBy']=='Incucyte':

        allMeta['FileSessionN'] = None
        
        # there's no big txt file, hopefully that's ok
        allMeta['metadata'] = ''
        
        # find time of session start datetime from first alphabetically first filename
        # here we have to deal with the 2 ways incucyte labels files with time...
        # one is like 00d00h00m and the other 2023y09m22d_22h30m
        tp0 = sorted(tps)[0]

        # TStep is a timedelta object
        # we get it from minimum of all time differences b/w images        
        inc_DT_reg = re.compile(r'_\d{4}y\d{2}m\d{2}d_\d{2}h\d{2}m.tif$')
        DTformat = '_%Yy%mm%dd_%Hh%Mm.tif'
        if re.search(inc_DT_reg,tp0):
            date_string = re.search(inc_DT_reg,tp0).group(0)
            allMeta['startMom'] = datetime.strptime(date_string,DTformat)    
            allDTs = [datetime.strptime(re.search(inc_DT_reg,tpi).group(0),DTformat) for tpi in tps]
            allDTs2 = [abs(DTi - allDTs[0]) for DTi in allDTs if DTi!=allDTs[0]]
            if allDTs2==[]:
                allMeta['TStep'] = timedelta(hours=999)
            else:
                allMeta['TStep'] = min(allDTs2)            
        else:
            # here the situation is that we know time deltas but not start 
            # mom. So we have to set start mom to the standard unknown date 
            # and add to this in TStep
            # 8/1/25: actually we have found this now but not yet implemented... look for <TimeStamp> in incu_meta
            allMeta['startMom'] = datetime(1818,5,5,0,0)
            if not silenceWarnings:
                print('warning: no start time found, set to default') 
            inc_DT_reg2 = re.compile(r'_\d{2}d\d{2}h\d{2}m.tif$')
            allDTs = [re.search(inc_DT_reg2,tpi).group(0) for tpi in tps]
            allDTs = ['_2000y09m'+str(int(adt[1:3])+5)+'d_'+adt[4:] for adt in allDTs]
            allDTs = [datetime.strptime(adt,DTformat) for adt in allDTs]
            allDTs2 = [abs(DTi - allDTs[0]) for DTi in allDTs if DTi!=allDTs[0]]
            if allDTs2==[]:
                allMeta['TStep'] = timedelta(hours=999)
            else:
                allMeta['TStep'] = min(allDTs2)
            
        # there's no 'delay' to consider, incucyte start time in filename      

        # haven't found this yet
        allMeta['ZStep'] = 1
        
        # so far we only do single channel incucyte!
        # note there is 'channel_names' in aicsimageio though so check that in future
        allMeta['Chan'] = ['BF']
        
        # set the Session imaging parameters and dimensions etc
        # NT is number of unique timestamps in filenames
        allMeta['NT'] = len(set(allDTs))
        
        # for now NF is just no. of files / NT since we assume NZ=NC=NM=1
        allMeta['NF'] = int(len(allDTs)/len(set(allDTs)))

        allMeta['NM'] = 1
        allMeta['NZ'] = 1
        allMeta['NC'] = 1
        
        allMeta['NY'] = aTag[1].value
        allMeta['NX'] = aTag[0].value    

        allMeta['imType'] = aTag[2].value          
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        # don't think we'll ever need this in incucyte
        allMeta['MOlap'] = False
        allMeta['NMY'] = 1
        allMeta['NMX'] = 1
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]
        
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None        
        
        allMeta['MontageOrder'] = False


        pixSize_pattern = r'<FullResPixelSpacing>(\d+(\.\d+)?) (\w+)</Full'
        out_pixSize = re.search(pixSize_pattern,incu_meta)

        if out_pixSize:
            allMeta['pixSizeY'] = float(out_pixSize.group(1))
            allMeta['pixSizeX'] = float(out_pixSize.group(1))  
            allMeta['pixUnit'] = out_pixSize.group(3)
            if allMeta['pixUnit']=='µm':
                allMeta['pixUnit'] = 'um'
        else:
            allMeta['pixSizeY'] = 1
            allMeta['pixSizeX'] = 1
            allMeta['pixUnit'] = None

        allMeta['SessionN'] = -1 # only used for multisesh


        
    elif allMeta['madeBy']=='Opera':
        
        indexFP = os.path.join(os.path.split(tp)[0],'Index.xml')
        if not os.path.isfile(indexFP):
            indexFP = os.path.join(os.path.split(os.path.split(tp)[0])[0],'Index.xml')
        tree = ET.parse(indexFP)
        root = tree.getroot()
        
        # just put whole XML file here? Maybe no point, 
        # should only be a few MB though
        with open(indexFP,'rt') as file:
            allMeta['metadata'] = file.read()

        crap = False
        for child in root:
            if re.search('^({.*})(.*)$',child.tag):
                crap = re.search('^({.*})(.*)$',child.tag).group(1)
                break
        if not crap:
            raise Exception('in opera index file the starting crap wasnt found')   
            
        plate = root.find(crap+'Plates').find(crap+'Plate')
        allMeta['startMom'] = plate.find(crap+'MeasurementStartTime').text 
        allMeta['startMom'] = datetime.fromisoformat(allMeta['startMom'])

        allMeta['Name'] = plate.find(crap+'PlateID').text

        allMeta['FileSessionN'] = None
            
        # haven't done multiple time steps yet
        
        # TStep is a timedelta object
        # not implemented multiple time steps yet
        allMeta['TStep'] = timedelta(hours=999)
        
        # have only tested 2 datasets and they both had chan names stored 
        # differently! So this covers both of them but feels like it may 
        # need more for the next data...
        chN_reg = r"ChannelName: ([\w\s()-]+),"
        chN_reg2 = r"([\w\s()-]+)"
        allMeta['Chan'] = []
        for i,map1 in enumerate(root.find(crap+'Maps').findall(crap+'Map')):
            entries = map1.findall(crap+'Entry')
            if len(entries)==0:
                continue
            else:
                ChannelNamesFF = [entry.find(crap+'FlatfieldProfile') for entry in entries]
                ChannelNames = [entry.find(crap+'ChannelName') for entry in entries]               
                if not any([ch==None for ch in ChannelNamesFF]):
                    allMeta['Chan'] = [re.search(chN_reg,ch.text).group(1) for ch in ChannelNamesFF]
                elif not any([ch==None for ch in ChannelNames]):
                    allMeta['Chan'] = [re.search(chN_reg2,ch.text).group(1) for ch in ChannelNames]
                else:
                    continue
        assert allMeta['Chan']!=[],'couldn\'t find channels in metadata'
        
        allMeta['NC'] = len(allMeta['Chan'])        
        
        # haven't had example of multiple timepoints yet so might break
        TPs = []
        for im in root.find(crap+'Images'):
            TPs.append(int(im.find(crap+'TimepointID').text))
        allMeta['NT'] = len(set(TPs))        
        
        # haven't had example of multiple z yet so might break
        ZPs = []
        ZAbsPs = []
        for im in root.find(crap+'Images'):
            ZPs.append(int(im.find(crap+'PlaneID').text))  
            ZAbsPs.append(float(im.find(crap+'AbsPositionZ').text))
        indexFileNZ = len(set(ZPs))  
        if re.search(r'\d\d(p\d\d)-',os.path.split(tp)[1]):
            allMeta['NZ'] = len(set(ZPs))  
        else:
            allMeta['NZ'] = 1

        diffs = []
        if allMeta['NZ']>1:
            for ii,(idd,pos) in enumerate(zip(ZPs[:-1],ZAbsPs[:-1])):
                if (ZPs[ii+1] - idd == 1):
                    diffs.append(ZAbsPs[ii+1] - pos) 
            allMeta['ZStep'] = np.mean(diffs)*1000000
        else:
            allMeta['ZStep'] = 1
            
        # this may fail because we are assuming one F per 'Well'
        # I think it's good though, I think multiple tiles within 
        # a well are always called montage tiles even if they don't 
        # form a contiguous square
        mapF = root.find(crap+'Plates').find(crap+'Plate')
        allMeta['NF'] = len(mapF.findall(crap+'Well'))
        
        # !!the division in line 5 has not been tested for multiple T,Z
        # this is old way that failed in case of opera output z-projection
        #... it counted the original z even though images weren't there!
        # this doesn't work when some files are missing! so see 26/11/24 change
        #allNM = []
        #for well in root.find(crap+'Wells').findall(crap+'Well'):
        #    allNM.append(len(well.findall(crap+'Image')))
        #assert len(set(allNM))==1, 'Not all fields had same NM!'
        #allMeta['NM'] = int(allNM[0]/(allMeta['NC']*allMeta['NT']*allMeta['NZ']))
        #allMeta['NM'] = int(len(root.find(crap+'Images').findall(crap+'Image'))/(allMeta['NF']*allMeta['NC']*allMeta['NT']*indexFileNZ))

        # 26/11/24 now trying just from file names since this should get the 
        # max NM then we deal with missing later
        Ms = [re.search(r'r\d\dc\d\df(\d+)',tpi).group(1) for tpi in tps]
        allMeta['NM'] = len(set(Ms))        
        
        # this also looks like it could fail in the future because we are 
        # relying on it being the index 2 map that pixel size is in. 
        # i.e. Index 0 and 1 have rubbish
        map2 = root.find(crap+'Maps').findall(crap+'Map')[2].find(crap+'Entry')
        allMeta['NY'] = int(map2.find(crap+'ImageSizeY').text)
        allMeta['NX'] = int(map2.find(crap+'ImageSizeX').text)
        
        if int(map2.find(crap+'MaxIntensity').text) == 65536:
            allMeta['imType'] = 16
        else: 
            raise Exception('Unknown imType for Opera')
            
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        mapI = root.find(crap+'Images').findall(crap+'Image')
        R1 = int(mapI[0].find(crap+'Row').text)
        C1 = int(mapI[0].find(crap+'Col').text)
        Z1 = int(mapI[0].find(crap+'PlaneID').text)
        Ch1 = int(mapI[0].find(crap+'ChannelID').text)
        T1 = int(mapI[0].find(crap+'TimepointID').text)
        posMX = []
        posMY = []
        for i,im in enumerate(mapI):
            R1i = int(im.find(crap+'Row').text)
            C1i = int(im.find(crap+'Col').text)
            Z1i = int(im.find(crap+'PlaneID').text)
            Ch1i = int(im.find(crap+'ChannelID').text)
            T1i = int(im.find(crap+'TimepointID').text)            
            if R1i==R1 and C1i==C1 and Z1i==Z1 and Ch1i==Ch1 and T1i==T1:
                posMX.append(float(im.find(crap+'PositionX').text))
                posMY.append(float(im.find(crap+'PositionY').text))       

        diffsMY = [abs(p - posMY[0]) for p in posMY if not p - posMY[0]==0]
        if diffsMY==[]:
            DY = 1
        else:
            DY = min(diffsMY)
        allMeta['NMY'] = round((max(posMY)-min(posMY))/DY)+1
        
        diffsMX = [abs(p - posMX[0]) for p in posMX if not p - posMX[0]==0]
        if diffsMX==[]:
            DX = 1
        else:
            DX = min(diffsMX)        
        print(posMX)
        allMeta['NMX'] = round((max(posMX)-min(posMX))/DX)+1                
        
        allMeta['TilePosX'] = posMX
        allMeta['TilePosY'] = posMY

        allMeta['pixSizeY'] = float(map2.find(crap+'ImageResolutionY').text)
        allMeta['pixSizeX'] = float(map2.find(crap+'ImageResolutionX').text)
        allMeta['pixUnit'] = 'm'     
        
        # I think there is no overlap provided, so we calc from positions
        diffs = [abs(x - allMeta['TilePosX'][0]) for x in allMeta['TilePosX']]
        diffs = [x for x in diffs if x!=0]
        if diffs==[]:
            allMeta['MOlap'] = 0.1
        else:
            allMeta['MOlap'] = 1-(min(diffs)/allMeta['pixSizeX'])/allMeta['NX']
        # I don't understand why this has to be put here but the overlaps that 
        # the stitch code finds are consistently half of what this 
        # calculation gives
        allMeta['MOlap'] = allMeta['MOlap']/2
        
        # I think everything goes wrong if NMX and NMY aren't the same for all 
        # F so I don't know why we have these... in Andor they are set 
        # differently for some reason though I think the result is always 
        # the same
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]

        # order in opera is crazy. First image is at zero but then it jumps to 
        # top left and snakes down. And just skips the zero position one when 
        # it comes to it!
        # so we call ordering TilePos and this tells us to check positions 
        # not ordering
        allMeta['MontageOrder'] = 'TilePos'   

        allMeta['SessionN'] = -1 # only used for multisesh
        
    elif allMeta['madeBy']=='Andor':
        
        # get a metadata txt file
        mp = genF.stripTags(tp,'Andor')+'.txt'
        assert os.path.exists(mp),EM.md2
        with open(mp,'rt') as file:
            allMeta['metadata'] = file.read()
        meta = allMeta['metadata']

        allMeta['FileSessionN'] = None
            
        # take start moment from the txt metadata
        startT = re.search(startTimeReg,meta).group(1)
        startDate = re.search(startDateReg,meta).group(1)
        startMom = startDate + ' ' + startT
        TX = '%d/%m/%Y %H:%M:%S'
        startMom = datetime.strptime(startMom,TX)  
        
        # add the delay time if necessary
        if re.search(delayReg,meta):
            delayT = int(re.search(delayReg,meta).group(1))
            if re.search(delayReg,meta).group(2)=='sec':
                delayT = timedelta(seconds=delayT)
                startMom += delayT            
            elif re.search(delayReg,meta).group(2)=='min':
                delayT = timedelta(minutes=delayT)
                startMom += delayT
            elif re.search(delayReg,meta).group(2)=='hr':
                delayT = timedelta(hours=delayT)
                startMom += delayT
            elif re.search(delayReg,meta).group(2)=='ms':
                delayT = timedelta(milliseconds=delayT)
                startMom += delayT                
            else:
                raise Exception('Unknown format for start delay in session.')
        allMeta['startMom'] = startMom         
        
        # take interval between time points from txt meta
        seshTStep = re.search(DTReg,meta)
        if seshTStep:
            seshTStep = int(seshTStep.group(1))
            if re.search(DTReg,meta).group(2) == 'hr':
                allMeta['TStep'] = timedelta(hours=seshTStep)
            elif re.search(DTReg,meta).group(2) == 'min':
                allMeta['TStep'] = timedelta(minutes=seshTStep)
            elif re.search(DTReg,meta).group(2) == 'sec':
                allMeta['TStep'] = timedelta(seconds=seshTStep)
            elif re.search(DTReg,meta).group(2) == 'ms':
                allMeta['TStep'] = timedelta(milliseconds=seshTStep)                
            elif 'fastest' in re.search(DTReg,meta).group(2):
                allMeta['TStep'] = timedelta(seconds=0)
            else:
                raise Exception(EM.md3)
        else:
            allMeta['TStep'] = timedelta(hours=999)

        # haven't found this yet
        allMeta['ZStep'] = 1            
        
        allMeta['Chan'] = re.findall(chanReg,meta)
        
        # set the session imaging parameters and dimensions etc
        if re.search(NTReg,meta):
            allMeta['NT'] = int(re.search(NTReg,meta).group(1))
        else:
            allMeta['NT'] = 1
        if re.search(NFReg,meta):
            allMeta['NF'] = int(re.search(NFReg,meta).group(1))
        else:
            allMeta['NF'] = 1
        if re.search(NMReg2,meta):
            allMeta['NM'] = int(re.search(NMReg2,meta).group(1))
        else:
            allMeta['NM'] = 1
        if re.search(NZReg,meta):
            allMeta['NZ'] = int(re.search(NZReg,meta).group(1))
        else:
            allMeta['NZ'] = 1
        if re.search(NCReg,meta):
            allMeta['NC'] = int(re.search(NCReg,meta).group(1))
        else:
            allMeta['NC'] = 1
        if re.search(NYReg,meta):
            allMeta['NY'] = int(re.search(NYReg,meta).group(1))
        else:
            allMeta['NY'] = 1
        if re.search(NXReg,meta):
            allMeta['NX'] = int(re.search(NXReg,meta).group(1))
        else:
            allMeta['NX'] = 1 
            
        if re.search(imTypeReg,meta):
            allMeta['imType'] = int(re.search(imTypeReg,meta).group(1))
        else: 
            raise Exception(EM.it1)
        assert allMeta['imType'] <= 16,EM.it2
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        # montage overlap, NMY and NMX (= numbers of tiles) and pixel stuff 
        molap = re.search(OlapReg,meta)
        if molap:
            allMeta['MOlap'] = int(molap.group(2))/100
        else:
            allMeta['MOlap'] = 0.01
        nm1 = re.search(NMReg,meta)
        if nm1:
            allMeta['NMY'] = int(nm1.group(2))
            allMeta['NMX'] = int(nm1.group(1))
        else:
            allMeta['NMY'] = 1
            allMeta['NMX'] = 1
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]
 
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None

        allMeta['MontageOrder'] = 'LRUD'
        allMeta['pixSizeY'] = float(re.search(pixSizeRegY,meta).group(1))
        allMeta['pixSizeX'] = float(re.search(pixSizeRegX,meta).group(1))
        allMeta['pixUnit'] = re.search(pixUnitReg,meta).group(1)

        allMeta['SessionN'] = -1 # only used for multisesh

        
    elif allMeta['madeBy']=='MicroManager':
        
        # no .txt metadata file in micromanager
        allMeta['metadata'] = ''

        allMeta['FileSessionN'] = None
        
        # start moment from image file metadata
        if 'Time' in t.keys():
            startMom = t['Time']
            startMom = re.search(startMomRegMM,startMom).group(1)
            TX = '%Y-%m-%d %H:%M:%S'
            allMeta['startMom'] = datetime.strptime(startMom,TX)              
        elif 'ReceivedTime' in t.keys():
            startMom = t['ReceivedTime']
            startMom = re.search(startMomRegMM,startMom).group(1)
            TX = '%Y-%m-%d %H:%M:%S'
            allMeta['startMom'] = datetime.strptime(startMom,TX)              
        else:
            # unknown startMom shouldn't be set to now because we sort 
            # first by startMom so don't want to impose artificial 
            # ordering here. Always use this datetime            
            allMeta['startMom'] = datetime(1818,5,5,0,0)
            if not silenceWarnings:
                print('warning: no start time found, set to default')             

        
        # take start moment from image file metadata
        allMeta['TStep'] = timedelta(seconds=m['Summary']['Interval_ms']/1000)

        # haven't found this yet
        allMeta['ZStep'] = 1        
        
        # currently don't know how to get this
        allMeta['Chan'] = ['BF']

        allMeta['imType'] = int(t['BitDepth'])
        assert allMeta['imType'] <= 16,EM.it2     
        
        # note recent MM seems to have m['Summary']['Frames/Slices/Channels']
        if 'IndexMap' in m:
            if 'Frame' in m['IndexMap']:
                allMeta['NT'] = max(m['IndexMap']['Frame'])+1
            else: 
                allMeta['NT'] = 1
            if 'Slice' in m['IndexMap']:
                allMeta['NZ'] = max(m['IndexMap']['Slice'])+1
            else: 
                allMeta['NZ'] = 1     
            if 'Channel' in m['IndexMap']:
                allMeta['NC'] = max(m['IndexMap']['Channel'])+1
            else:
                allMeta['NC'] = 1
        else:
            allMeta['NT'] = 1
            allMeta['NZ'] = 1
            allMeta['NC'] = 1        
            
        # allMeta['NM']  is done below where it is easier
        # allMeta['NF'] = len(labelsSet) is done below where it is easier
        allMeta['NY'] = t['Height']
        allMeta['NX'] = t['Width']
        
        # these things will be useful: (details for every position)
        if 'InitialPositionList' in m['Summary']:
            summary =  m['Summary']['InitialPositionList']
        elif 'StagePositions' in m['Summary']:
            summary =  m['Summary']['StagePositions']
        else:
            raise Exception('could not find summary in MM metadata')
            
        #labels = [im['Label'].split('Pos_')[0] for im in summary] # old way!!
        labels = [im['Label'] for im in summary]
        labels = [re.sub(r'_\d\d\d_\d\d\d','',l) for l in labels]
        labelsSet = sorted(set(labels))

        allMeta['NF'] = len(labelsSet)

        if 'GridColumnIndex' in summary[0]:
            cols = [s['GridColumnIndex'] for s in summary]
            rows = [s['GridRowIndex'] for s in summary]
        elif 'GridRow' in summary[0]:
            cols = [s['GridCol'] for s in summary]
            rows = [s['GridRow'] for s in summary]        
        
        # montage overlap, NMY and NMX (= numbers of tiles) and pixel stuff 
        # currently don't know how to get most of them
        allMeta['MOlap'] = 0.05 # doesn't seem to be in metadata
        allMeta['pixSizeY'] = 1 # doesn't seem to be in metadata
        allMeta['pixSizeX'] = 1 # doesn't seem to be in metadata
        allMeta['pixUnit'] = None
        cols2 = [[c for c,L in zip(cols,labels) if l in L] for l in labelsSet]
        rows2 = [[r for r,L in zip(rows,labels) if l in L] for l in labelsSet]
        allMeta['NMY'] = max([max(c)+1 for c in cols2])
        allMeta['NMX'] = max([max(r)+1 for r in rows2])
        allMeta['NM'] = allMeta['NMY']*allMeta['NMX']
        allMeta['MontageOrder'] = 'UDRL'

        allMeta['FieldNMYs'] = [max(c)+1 for c in cols2]
        allMeta['FieldNMXs'] = [max(r)+1 for r in rows2]
        
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None        
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])

        allMeta['SessionN'] = -1 # only used for multisesh
        
    elif allMeta['madeBy']=='aicsimageio_compatible':

        if 'S' in aics.dims.__dict__.keys(): 
            assert aics.dims['S'][0]==3,'looks like RGB but not 3 S-chan??'
            allMeta['aics_RGB'] = True

        allMeta['FileSessionN'] = None
        
        # there's no big txt file, hopefully that's ok
        allMeta['metadata'] = ''
        
        # I think aics never has time stamp, so resorting to aics looses this
        # unknown startMom shouldn't be set to now because we sort 
        # first by startMom so don't want to impose artificial 
        # ordering here. Always use this datetime        
        allMeta['startMom'] = datetime(1818,5,5,0,0)
        if not silenceWarnings:
            print('warning: no start time found, set to default') 
        
        # I think aics never has DT, so resorting to aics looses this
        allMeta['TStep'] = timedelta(hours=999)

        if aics.physical_pixel_sizes.Z:
            allMeta['ZStep'] = aics.physical_pixel_sizes.Z
        allMeta['ZStep'] = 1                

        # in aics,dims there is only CZTXY so I assume we have to stick with 
        # F=1 per image
        if 'F' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['F'])
            countQ = []
            for k,v in grpd.items():
                countQ.append(1)
            allMeta['NF'] = sum(countQ)
        else:
            allMeta['NF'] = 1
        
        if 'T' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['T'])
            countQ = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                countQ.append(aics2.dims.T)
            allMeta['NT'] = sum(countQ)
        else:
            allMeta['NT'] = aics.dims.T
        
        if 'M' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['M'])
            countQ = []
            for k,v in grpd.items():
                countQ.append(1)
            allMeta['NM'] = sum(countQ)
        else:
            allMeta['NM'] = 1
            
        if 'Z' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['Z'])
            countQ = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                countQ.append(aics2.dims.Z)
            allMeta['NZ'] = sum(countQ)
        else:
            allMeta['NZ'] = aics.dims.Z
            
        if 'C' in customTags.keys():
            assert not allMeta['aics_RGB'],'can\'t have RGB and custom C-tag'
            grpd = genF.groupByTag(tps,customTags['C'])
            countQ = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                countQ.append(aics2.dims.C)
            allMeta['NC'] = sum(countQ)
        elif allMeta['aics_RGB']:
            allMeta['NC'] = 3
        else:
            allMeta['NC'] = aics.dims.C
            
        if 'Y' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['Y'])
            countQ = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                countQ.append(aics2.dims.Y)
            allMeta['NY'] = sum(countQ)
        else:
            allMeta['NY'] = aics.dims.Y
            
        if 'X' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['X'])
            countQ = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                countQ.append(aics2.dims.X)
            allMeta['NX'] = sum(countQ)
        else:
            allMeta['NX'] = aics.dims.X
            
            
        # note there is 'channel_names' in aicsimageio though so check that in future
        if allMeta['aics_RGB']:
            allMeta['Chan'] = ['Red','Green','Blue']
        elif 'C' in customTags.keys():
            grpd = genF.groupByTag(tps,customTags['C'])
            names = []
            for k,v in grpd.items():
                aics2 = AICSImage(v[0])
                names += aics2.channel_names 
            allMeta['Chan'] = names
        else:
            with tifffile.TiffFile(tp) as tif:
                I = tif.imagej_metadata
                # here we are checking if this is a composite FIJI image which we can get colours from
                if I and isinstance(I,dict) and 'mode' in I.keys() and I['mode']=='composite' and 'LUTs' in I.keys():
                    allMeta['Chan'] = [genF.LUT2ColourDict(tuple(np.any(lut,axis=1))) for lut in I['LUTs']]
                else:    
                    allMeta['Chan'] = aics.channel_names            
            
        allMeta['imType'] = str(aics.dtype)
        
        allAxes = ['NT','NF','NM','NZ','NC','NY','NX']
        allMeta['Shape'] = tuple([allMeta[k] for k in allAxes])
        
        # don't think we'll ever need this in incucyte
        allMeta['MOlap'] = False
        allMeta['MOlap'] = False
        allMeta['MOlap'] = False
        allMeta['NMY'] = 1
        allMeta['NMX'] = 1
        allMeta['FieldNMYs'] = [allMeta['NMY'] for f in range(allMeta['NF'])]
        allMeta['FieldNMXs'] = [allMeta['NMX'] for f in range(allMeta['NF'])]
        
        allMeta['TilePosX'] = None
        allMeta['TilePosY'] = None        
        
        allMeta['MontageOrder'] = False

        if aics.physical_pixel_sizes.Y:
            allMeta['pixSizeY'] = aics.physical_pixel_sizes.Y
        else:
            allMeta['pixSizeY'] = 1
        if aics.physical_pixel_sizes.X:
            allMeta['pixSizeX'] = aics.physical_pixel_sizes.X
        else:
            allMeta['pixSizeX'] = 1
        allMeta['pixUnit'] = 'micron'        

        allMeta['SessionN'] = -1 # only used for multisesh
        
        
    else:
        raise Exception(EM.md1)
    
    if isinstance(allMeta,dict):
        if not 'Name' in allMeta.keys():
            allMeta['Name'] = genF.stripTags(os.path.split(tp)[1],allMeta['madeBy'],customTags)

    return allMeta



def file2Dims(filePath,checkCorrupt=True):
    """
    This finds the dimensions of the file in 7D format.
    
    Parameters
    -----------
    filePath : str
        The path to the image file
    checkCorrupt : bool
        Whether to raise an assertion if the number of images in the file 
        doesn't correspond to the dimensions it thinks it has. I.e. if the 
        file is corrupt.
        
    Returns
    ---------
    dims : tuple
        (NT,NF,NM,NZ,NC,NY,NX)
    """
    try: 
        if filePath[-4:]=='.czi':
            
            aics = AICSImage(filePath)
            meta_xml = aics.metadata.find('Metadata').find('Information') 
            if meta_xml.find('Image').find('SizeT'):
                NT = int(meta_xml.find('Image').find('SizeT').text)
            else:
                NT = 1
            # for czi I haven't had multi F example yet
            NF = 1
            # for czi I haven't had multi M example yet
            NM = 1
            NZ = int(meta_xml.find('Image').find('SizeZ').text)
            try:
                NC = int(meta_xml.find('Image').find('SizeC').text)
            except:
                NC = len(meta_xml.find('Image').find('Dimensions').find('Channels').findall('Channel'))             
            NY = int(meta_xml.find('Image').find('SizeY').text)
            NX = int(meta_xml.find('Image').find('SizeX').text)
            dims = [NT,NF,NM,NZ,NC,NY,NX]
            
        elif filePath[-4:]=='.nd2':
            
            aics = AICSImage(filePath)

            NC = len(aics.metadata['metadata'].channels)
            NY = aics.metadata['attributes'].heightPx
            NX = aics.metadata['attributes'].widthPx
        
            time_type = "<class 'nd2.structures.TimeLoop'>"
            fields_type = "<class 'nd2.structures.XYPosLoop'>"
            zstack_type = "<class 'nd2.structures.ZStackLoop'>"
        
            types = [str(type(ob)) for ob in aics.metadata['experiment']]
        
            if time_type in types:
                time_index = types.index(time_type)
                NT = aics.metadata['experiment'][time_index].count
            else:
                NT = 1
            if fields_type in types:
                fields_index = types.index(fields_type)
                NF = aics.metadata['experiment'][fields_index].count
            else:
                NF = 1
            if zstack_type in types:
                zstack_index = types.index(zstack_type)
                NZ = aics.metadata['experiment'][zstack_index].count
            else:
                NZ = 1        
    
                # haven't done montages for nikon yet
            NM = 1
    
            dims = [NT,NF,NM,NZ,NC,NY,NX]            
            
        elif filePath[-4:]=='.npz':
            with np.load(filePath) as tif:
                mask = tif['mask']
                dims = mask.shape
        else:  
            with tifffile.TiffFile(filePath) as tif:
                m = tif.fluoview_metadata
                I = tif.imagej_metadata
                mm = tif.micromanager_metadata
                
                # this is for checking if it is made by incucyte
                incucyte = False
                opera = False
                for tag in tif.pages[0].tags:
                    if 'Incucyte' in str(tag):
                        incucyte=True
                        break
                    if 'PerkinElmer' in str(tag):
                        opera = True
                        break
                    
                if m and 'Dimensions' in m.keys():
                    dimDic = {l[0]:l[1] for l in m['Dimensions']}
                    shapeKeys = ['Time','XY','Montage','Z','Wavelength','y','x']
                    dims = []
                    for k in shapeKeys:
                        if k in dimDic.keys():
                            dims.append(dimDic[k])
                        else:
                            dims.append(1)
                    if 'Time1' in dimDic.keys():
                        dims[0] = dims[0]*dimDic['Time1']
                elif I and 'tdata_meta_data' in I.keys():
                    dims = genF.meta_str_2_dict(I['tdata_meta_data'])['Shape']
                elif I and 'tw_nt' in I.keys():
                    baseString = 'tw_n'
                    dimStrings = ['t','f','m','z','c','y','x']
                    dims = [I[baseString+LL] for LL in dimStrings]
                elif mm:
                    p = tif.pages[0].tags['MicroManagerMetadata'].value
                    
                    if 'InitialPositionList' in mm['Summary']:
                        summ = mm['Summary']['InitialPositionList']
                    elif 'StagePositions' in mm['Summary']:
                        summ = mm['Summary']['StagePositions']

                    if 'IndexMap' in mm and 'Frame' in mm['IndexMap']:
                        nt = len(set(mm['IndexMap']['Frame']))
                    else:
                        nt = 1
                    nf = 1
                    nm = 1
                    if 'IndexMap' in mm and 'Slice' in mm['IndexMap']:
                        nz = len(set(mm['IndexMap']['Slice']))
                    else:
                        nz = 1  
                    if 'IndexMap' in mm and 'Channel' in mm['IndexMap']:
                        nc = len(set(mm['IndexMap']['Channel']))
                    else:
                        nc = 1                          
                    ny = p['Height']
                    nx = p['Width']
                    dims = [nt,nf,nm,nz,nc,ny,nx]
                elif incucyte:
                    aTag = tif.pages[0].tags.values()
                    dims = [1,1,1,1,1,aTag[1].value,aTag[0].value]
                # assumes opera separates all axes, haven't tested for multi-t
                elif opera:
                    ny,nx = tif.series[0].shape
                    dims = [1,1,1,1,1,ny,nx]
                else:
                    try:
                        aics = AICSImage(filePath)
                        nt = aics.dims.T
                        nf = 1
                        nm = 1
                        nz = aics.dims.Z
                        nc = aics.dims.C
                        ny = aics.dims.Y
                        nx = aics.dims.X
                        dims = [nt,nf,nm,nz,nc,ny,nx]                    
                    except:
                        raise Exception(EM.md1.format(filePath))
                    
                if checkCorrupt:
                    L = len(tif.pages)
                    assert np.prod(dims[:5])==L,EM.cf1.format(filePath)
    except tifffile.TiffFileError as TFE:
        print('from multisesh.findMeta.file2Dims: Critical problem with tiff file: ',filePath)
        raise TFE
        
    return tuple(dims)
        

