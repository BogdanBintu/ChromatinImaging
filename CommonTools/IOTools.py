#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

#external packages
import numpy as np
import re
import glob

##Reader classes and functions
# Classes that handles reading STORM movie files.
# Currently this is limited to the dax format.
# It will be extended to tiff files in future releases.

#
# The superclass containing those functions that 
# are common to reading a (STORM/diffraction-limited) movie file.
# This was originally develloped by Hazen Babcok and extended by Bogdan Bintu

class Reader:
    
    # Close the file on cleanup.
    def __del__(self):
        if self.fileptr:
            self.fileptr.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if self.fileptr:
            self.fileptr.close()

    # Average multiple frames in a movie.
    def averageFrames(self, start = False, end = False, verbose = False):
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames 

        length = end - start
        average = np.zeros((self.image_width, self.image_height), np.float)
        for i in range(length):
            if verbose and ((i%10)==0):
                print " processing frame:", i, " of", self.number_frames
            average += self.loadAFrame(i + start)
            
        average = average/float(length)
        return average

    # returns the film name
    def filmFilename(self):
        return self.filename

    # returns the film size
    def filmSize(self):
        return [self.image_width, self.image_height, self.number_frames]

    # returns the picture x,y location, if available
    def filmLocation(self):
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    # returns the film focus lock target
    def lockTarget(self):
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0

    # returns the scale used to display the film when
    # the picture was taken.
    def filmScale(self):
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]


#
# Dax reader class. This is the Zhuang lab custom format.
#
class DaxReader(Reader):
    # dax specific initialization
    def __init__(self, filename, verbose = 0):
        # save the filenames
        self.filename = filename
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user 
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print "Could not determine image size, assuming 256x256."
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            self.fileptr = 0
            if verbose:
                print "dax data not found", filename
                
    # Create and return a memory map the dax file
    def loadMap(self):
        if os.path.exists(self.filename):
            if self.bigendian:
                self.image_map = np.memmap(self.filename, dtype='>u2', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
            else:
                self.image_map = np.memmap(self.filename, dtype='uint16', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
        return self.image_map
        
    # load a frame & return it as a np array
    def loadAFrame(self, frame_number):
        if self.fileptr:
            assert frame_number >= 0, "frame_number must be greater than or equal to 0"
            assert frame_number < self.number_frames, "frame number must be less than " + str(self.number_frames)
            self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
            image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
            image_data = np.transpose(np.reshape(image_data, [self.image_width, self.image_height]))
            if self.bigendian:
                image_data.byteswap(True)
            return image_data
    # load full movie and retun it as a np array        
    def loadAll(self):
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = -1)
        image_data = np.swapaxes(np.reshape(image_data, [self.number_frames,self.image_width, self.image_height]),1,2)
        if self.bigendian:
            image_data.byteswap(True)
        return image_data
  
# function for quickly reading fasta files
def fastaread(fl,force_upper=False):
    """
    Given a .fasta file <fl> this returns names,sequences
    """
    fid = open(fl,'r')
    names = []
    seqs = []
    lines = []
    while True:
        line = fid.readline()
        if not line:
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            break
        if line[0]=='>':
            name = line[1:-1]
            names.append(name)
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            lines = []
        else:
            lines.append(line[:-1])
    fid.close()
    return [names,seqs[1:]]



#Stand alone functions
def get_ims(dax,trans=[0,0,0],minmax_coords=[0,-1,0,-1,0,-1]):
    imRed = np.swapaxes(DaxReader(dax).loadMap(),1,2)
    im=imRed[12:-12]
    zmin,zmax,xmin,xmax,ymin,ymax = np.array(minmax_coords)
    im_cy3_base = im[2::3]
    im_cy5_base = im[1::3]
    im_dapi_base = im[0::3]
    shape_ = np.max([im_cy3_base.shape,im_cy5_base.shape,im_dapi_base.shape],axis=0)
    if zmax<0:zmax=shape_[0]
    if xmax<0:xmax=shape_[1]
    if ymax<0:ymax=shape_[2]
    trans_=np.array(np.round(trans),dtype=int)
    zmin-=trans_[0]
    zmax-=trans_[0]
    xmin-=trans_[1]
    xmax-=trans_[1]
    ymin-=trans_[2]
    ymax-=trans_[2]
    ims=[]
    for im_ in [im_cy5_base,im_cy3_base,im_dapi_base]:
        shape_=im_.shape
        im_base_0 = np.zeros([zmax-zmin,xmax-xmin,ymax-ymin])
        im_zmin = min(max(zmin,0),shape_[0])
        im_zmax = min(max(zmax,0),shape_[0])
        im_xmin = min(max(xmin,0),shape_[1])
        im_xmax = min(max(xmax,0),shape_[1])
        im_ymin = min(max(ymin,0),shape_[2])
        im_ymax = min(max(ymax,0),shape_[2])
        im_base_0[(im_zmin-zmin):(im_zmax-zmin),(im_xmin-xmin):(im_xmax-xmin),(im_ymin-ymin):(im_ymax-ymin)]=im_[im_zmin:im_zmax,im_xmin:im_xmax,im_ymin:im_ymax]
        ims.append(im_base_0)
    return ims
def hybe_number(hybe_tag):
    is_letter = [char.isalpha() for char in hybe_tag]
    pos = np.where(is_letter)[0]
    if len(pos)==1:
        pos=list(pos)+[len(hybe_tag)]
    return int(hybe_tag[pos[0]+1:pos[1]])
def get_valid_dax(spots_folder,ifov=0):
    files_folders = glob.glob(spots_folder+os.sep+'*')
    folders = [fl for fl in files_folders if os.path.isdir(fl)]
    valid_folders = [folder for folder in folders if os.path.basename(folder)[0]=='H']
    hybe_tags = [os.path.basename(folder) for folder in valid_folders]
    #order hybe tags
    
    hybe_tags = np.array(hybe_tags)[np.argsort(map(hybe_number,hybe_tags))]
    fov_tags=[]
    for hybe_tag in hybe_tags:
        fov_tags.extend(map(os.path.basename,glob.glob(spots_folder+os.sep+hybe_tag+os.sep+'*.dax')))
    fov_tags = np.unique(fov_tags)
    fov_tag = fov_tags[ifov]
    daxs = [spots_folder+os.sep+tag+os.sep+fov_tag for tag in hybe_tags]
    daxs = [dax for dax in daxs if os.path.exists(dax)]
    return daxs
