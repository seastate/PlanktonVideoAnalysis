import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from scipy import signal
from math import floor,sqrt
import json
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse

# Dictionaries of output headers and formats
part_fmts = {'fos-part':'f"{self.fseq.frame_pointer} {self.fseq.time} 1 {MAR[0][0]} {MAR[0][1]} {area} {i} {bbox[2]} {bbox[3]} {min(MAR[1])} {max(MAR[1])} {MAR[2]+90.} {MAR[2]} {MAR[1][1]/(MAR[1][0]+1.e-16)} {MAR[1][1]} {MAR[2]+90.} {MAR[1][0]} {MAR[2]} {MAR[2]} {MAR[1][1]/(MAR[1][0]+1.e-16)}"'
}
part_hdrs = {'fos-part':"% Frame #, Time, Camera #, X, Y, Area, Particle #, Bounding Box Width, Bounding Box Height, Min Dim, Max Dim, Min Dim Angle, Max Dim Angle, Max / Min, Length, Length Angle, Width, Width Angle, Length / Width\n"}

part_frm_mrk = {'fos-part':'f"{self.fseq.frame_pointer} {self.fseq.time} 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"'}

plt_pad=0.05

class TemporalFilter():
    """
    A class to implement accumulation of a background frame through temporal 
    filtering, and its use to isolate foreground (moving objects in video
    analysis). Uses include:
        (a) Subtraction of a slowly evolving background to emphasize a
            quickly evolving foreground; and,
        (b) Substitution of a temporally-filtered frame sequence to remove
            frame-rate noise and other short time scale interference.
    The temporal filter frame is updated by a weighted sum of the pre-existing
    temporal filter frame and a new frame from the image sequence. The relative
    weights, and therefore the timescale of change for the temporal filter frame,
    is determined by the parameter alpha, which must be in the interval [0,1].
    The temporal filter frame approaches the current image sequence frame at an
    exponential timescale of approximately 1/alpha.

    It is assumed that frame arrays have been converted to float,
    to avoid cumulative rounding errors.
    """
    def __init__(self,init_frame,pars={}):
        """
        Create a TemporalFilter instance. 

        Parameters are stored in the `pars' dictionary attribute, with some defaults 
        set at initialization, that are superceded by pars dictionary argument (if any).

        Class methods perform three basic functions:
            -- Update the temporal filter frame
            -- Apply the temporal filter to a new frame, by replacement or subtraction
            -- Display the current temporal filter frame
        """
        # Set default parameters
        self.pars = {'alpha':0.1,'offset':128,
                     'mode':'subtract','update':True,'display':False}
        if bool(pars):
            self.pars.update(pars)
        self.TFframe = init_frame #.astype('float')
        if bool(self.pars['display']):
            self.fig = plt.figure()
            if self.pars['mode'] == 'subtract':
                self.fig2 = plt.figure()
        self.frameno = None

    def update(self,frame):
        """
        Calculate a weighted sum of the existing temporal filter
        frame (TFframe) and the frame submitted as an argument.
        """
        self.TFframe = (1.-self.pars['alpha'])*self.TFframe + \
                        self.pars['alpha']*frame

    def display(self):
        """
        Display the current temporal filter frame (TFframe).
        """
        plt.figure(self.fig)
        self.fig.clf()
        if len(self.TFframe.shape) == 3: # A color image...
            plt.imshow(np.round(self.TFframe).astype('uint8'))
        else:
            plt.imshow(np.round(self.TFframe).astype('uint8'),cmap='gray', vmin=0, vmax=255)
        # If subtract mode, also plot result
        if self.pars['mode'] == 'replace':
            self.fig.canvas.manager.set_window_title(f'Temporal Filter result: Frame {self.frameno}')
        plt.pause(1e-2)
        if self.pars['mode'] == 'subtract':
            self.fig.canvas.manager.set_window_title(f'Temporal Filter background: Frame {self.frameno}')
            plt.figure(self.fig2)
            self.fig2.clf()
            if len(self.result.shape) == 3: # A color image...
                plt.imshow(np.round(self.result).astype('uint8'))
            else:
                plt.imshow(np.round(self.result).astype('uint8'),cmap='gray', vmin=0, vmax=255)
            self.fig2.canvas.manager.set_window_title(f'Temporal Filter result: Frame {self.frameno}')
            plt.pause(1e-2)

        
    def apply(self,frame,frameno=None):
        """
        Apply the current temporal filter frame (TFframe) to the frame submitted as an
        argument. If mode is "subtract", the returned image is the submitted frame minus
        the temporal filter frame. If mode is "replace", the returned image is the temporal
        filter frame. If the scalar "offset" is not None, it is added to each pixel value 
        to mitigate under- or over-running the [0,255] interval for uint8 variables 
        (typically useful for "subtract" mode).
        """
        self.frameno = frameno
        self.ret = False
        if self.pars['update'] == True:
            self.update(frame)
        if self.pars['mode'] == 'replace':
            self.result = self.TFframe
            #self.result = np.round(self.TFframe).astype('uint8')
        elif self.pars['mode'] == 'subtract':
            self.result = np.minimum(255.,np.maximum(0.,frame-self.TFframe))
        else:
            print(f"Unrecognized mode {self.pars['mode']}; aborting...")
            return self.ret,None
        if bool(self.pars['offset']):
            self.result += self.pars['offset']
        if bool(self.pars['display']):
            self.display()
        self.ret = True
        return self.ret,self.result


class SpatialFilter():
    """
    A class to implement accumulation of a background frame through spatial 
    filtering, and its use to isolate foreground (moving objects in video
    analysis). Uses include:
        (a) Subtraction of a diffuse or large-scale  background to emphasize 
            small-scale foreground features; and,
        (b) Substitution of a spatially-filtered frame sequence to remove
            pixel-scale noise and other short spatial scale interference.
    The spatial filter frame is determined by a convolution with a filter kernel,
    as for example a uniform disk. This convolution suppresses features small in 
    length scale compared to the disk radius. 

    Currently only disk kernels are implemented.

    It is assumed that frame arrays have been converted to float,
    to avoid cumulative rounding errors.
    """
    def __init__(self,pars={}):
        """
        Create a SpatialFilter instance. 

        Parameters are stored in the `pars' dictionary attribute, with some defaults 
        set at initialization, which are superceded by pars dictionary argument (if any).

        Class methods perform four basic functions:
            -- Define the convolution kernel.
            -- Update the temporal filter frame
            -- Apply the temporal filter to a new frame, by replacement or subtraction
            -- Display the current temporal filter frame
        """
        # default parameters
        self.pars={'n':11,'r':5,'ftype':'disk','offset':0,
                   'fillvalue':0,'plot_matrix':False,'verbose':False,
                   'get_sf':False,'get_filter':True,
                   'offset':None,'update':True,'mode':'subtract','display':False}
        if bool(pars):
            self.pars.update(pars)
        if self.pars['get_filter']:
            self.getFilter()
        if bool(self.pars['display']):
            self.fig = plt.figure()
            if self.pars['mode'] == 'subtract':
                self.fig2 = plt.figure()
        self.frameno = None

    def getSF(self,frame,return_sf=False):
        """
        Apply the current convolution kernel to the frame submitted as an argument.
        """
        self.ret = False
        #self.frame_float = frame.astype('float')
        #
        if len(frame.shape) == 3: # A color image...
            self.SFframe = np.zeros(frame.shape)
            for i in range(3):
                self.SFframe[:,:,i] = signal.convolve2d(frame[:,:,i],
                                                        self.conv_matrix,
                                                        mode='same',boundary='fill',
                                                        fillvalue=self.pars['fillvalue'])
        else:   # A grayscale image
            self.SFframe = signal.convolve2d(frame,self.conv_matrix,
                                             mode='same',boundary='fill',
                                             fillvalue=self.pars['fillvalue'])
        self.ret = True
        if return_sf:
            return self.ret,self.SFframe

    def getFilter(self,pars={}):
        """
        Generate a convolution kernel or read one from a file (not yet implemented).
        """
        self.conv_matrix = None
        if bool(pars):
            self.pars.update(pars)
        if self.pars['ftype']=='disk':
            self.getDiskFilter(verbose=self.pars['verbose'])
        else:
            print('Unrecognized filter type...aborting')
            return False
        if self.pars['plot_matrix']:
            plt.figure()
            plt.imshow(self.conv_matrix)
            plt.pause(1e-1)
            self.pars.update({'plot_matrix':False})

    def getDiskFilter(self,verbose=False):
        """
        Generate a disk convolution kernel, as specified by the parameters
        n (size of the square kernel, in pixels) and offset (the sum of the 
        kernel matrix entries). Offset > 0 is typically used only in subtraction
        mode, to mitigate under- or over-running the [0,255] interval for uint8
        variables
        """
        self.offset = self.pars['offset']
        self.conv_matrix = np.zeros([self.pars['n'],self.pars['n']])
        self.n_mid = floor(self.pars['n']/2)
        if self.pars['r'] is not None:
            r = self.pars['r']
        else:
            r = (self.pars['n']+1)/2
        for irow in range(self.pars['n']):
            for icol in range(self.pars['n']):
                dist = sqrt( (irow-self.n_mid)**2 + (icol-self.n_mid)**2 );
                if dist < r:
                    self.conv_matrix[irow,icol] = 1.
        self.conv_matrix /= np.sum(self.conv_matrix)
        if verbose:
            print(f'Disk filter: using n_mid = {self.n_mid}')
            print(self.conv_matrix)

    def display(self):
        """
        Display the current spatial filter frame (SFframe).
        """
        plt.figure(self.fig)
        self.fig.clf()
        if len(self.SFframe.shape) == 3: # A color image...
            plt.imshow(np.round(self.SFframe).astype('uint8'))
        else:
            plt.imshow(np.round(self.SFframe).astype('uint8'),cmap='gray', vmin=0, vmax=255)
        if self.pars['mode'] == 'replace':
            self.fig.canvas.manager.set_window_title(f'Spatial Filter result: Frame {self.frameno}')
        plt.pause(1e-2)
        # If subtract mode, also plot result
        if self.pars['mode'] == 'subtract':
            self.fig.canvas.manager.set_window_title(f'Spatial Filter background: Frame {self.frameno}')
            plt.figure(self.fig2)
            self.fig2.clf()
            if len(self.result.shape) == 3: # A color image...
                plt.imshow(np.round(self.result).astype('uint8'))
            else:
                plt.imshow(np.round(self.result).astype('uint8'),cmap='gray', vmin=0, vmax=255)
            self.fig2.canvas.manager.set_window_title(f'Spatial Filter result')
        plt.pause(1e-2)
        
    def apply(self,frame,frameno=None):
        """
        Apply the current spatial filter frame (SFframe) to the frame submitted as an
        argument. If mode is "subtract", the returned image is the submitted frame minus
        the spatial filter frame. If mode is "replace", the returned image is the spatial
        filter frame. If the scalar "offset" is not None, it is added to each pixel value 
        to mitigate under- or over-running the [0,255] interval for uint8 variables 
        (typically useful for "subtract" mode).
        """
        self.frameno = frameno
        self.ret = False
        if self.pars['update']:
            self.getSF(frame)
        #self.SFframe_int32 = np.round(self.SFframe).astype('int32')
        if self.pars['mode'] == 'replace':
            self.result = self.SFframe
        elif self.pars['mode'] == 'subtract':
            #self.frame_int32 = np.round(frame).astype('int32')
            self.result = np.minimum(255.,np.maximum(0.,frame-self.SFframe))
        else:
            print(f"Unrecognized mode {self.pars['mode']}; aborting...")
            return self.ret,None
        if bool(self.pars['offset']):
            self.result += self.pars['offset']
        if bool(self.pars['display']):
            self.display()
        self.ret = True
        return self.ret,self.result

    
class FrameSequence():
    """
    A class to facilitate work combining images sequences, frames in videos and
    lists of image files during image processing. A FrameSequence object provides
    a standardized way to present sequences of images in directories, file lists
    and videos for image processing, segmentation and generation of ROIs. Images
    are returned and/or stored as numpy arrays.

    The types of image sequences currently supported are:
        -- "imseq": a list of sequential image file paths
        -- "imfile": a file listing sequential image paths, in the directory source_dir
        -- "vidfile": the path to a video file
        -- "vidseq": a cv2 VideoCapture object
    The image sequence type can be specified by the fs_type item in the
    pars dictionary, or is inferred by the first non-None entry among
    the image_sequence, image_file_list, video_sequence and video_file
    items in the pars dictionary. 

    Other arguments include:
        -- init: If True, the image sequence is automatically initialized. If False, the 
                 image sequence must be explicitly initialized.
        -- return_frame: If True, initialization returns the resulting frame (and stores it 
                  in the frame attribute. If False, it is stored but not returned (e.g., 
                  when an object is being instantiated to be saved with a variable name).
        -- "interval": The interval at which to return frames from the sequence(default: 1,
                  i.e., return every frame).
        -- "frame_pointer": the initial frame to return and/or store.
        -- "fpd": frame per second for the image sequence
        -- "float_convert": If True, the frame is converted to a numpy float array. If False,
                  the native format (usually uint8) is preserved.
    """
    def __init__(self,pars={}):
        self.pars={'image_file_list':None,'image_sequence':None,
                 'video_file':None,'video_sequence':None,
                 'source_dir':None,'init':True,'gray_convert':False,
                 'frame_pointer':0,'interval':1,'fs_type':None,
                   'float_convert':False,'fps':30.,'display':False}
        """
        Create a FrameSequence instance. 

        Parameters are stored in the `pars' dictionary attribute, with some defaults 
        set at initialization, which are superceded by pars dictionary argument (if any).

        Class methods perform two basic functions:
            -- Initialzie by opening files and loading an initial image
            -- Apply the sequencing, by loading and returning and/or storing the next 
               requested frame
        """
        if bool(pars):
            self.pars.update(pars)
        self.image_file_list = self.pars['image_file_list']
        self.image_sequence = self.pars['image_sequence']
        self.video_file = self.pars['video_file']
        self.video_sequence = self.pars['video_sequence']
        self.source_dir = self.pars['source_dir']
        
        self.gray_convert = self.pars['gray_convert']
        self.float_convert = self.pars['float_convert']
        self.frame_pointer = self.pars['frame_pointer']
        self.interval = self.pars['interval']
        self.frame = None
        self.video = None
        self.ret = None
        self.fps = self.pars['fps']
        # Add accounting for time
        self.time = self.frame_pointer/self.fps
        
        # Set type of frame sequence. If not set explicitly, the type is determined
        # by the first of the following arguments that is not None:
        if self.pars['fs_type'] is not None:
            self.fs_type = self.pars['fs_type']
        elif self.pars['image_sequence'] is not None:
            self.fs_type = 'imseq'
        elif self.pars['image_file_list'] is not None:
            self.fs_type = 'imfil'
        elif self.pars['video_sequence'] is not None:
            self.fs_type = 'vidseq'
        elif self.pars['video_file'] is not None:
            self.fs_type = 'vidfil'
        else:
            print('Frame sequence type not set: manually set attribute ' + \
                  'fs_type to imseq, imfil, vidseq or vidfil')
        if self.pars['init']:
            self.initialize(return_frame=False)
        if bool(self.pars['display']):
            self.fig = plt.figure()

    def initialize(self,return_frame=True):
        """
        Initialize the image sequence by opening a file if necessary, and loading
        the first requested frame (the first, unless otherwise indicated by the
        frame_pointer parameter.
        """
        self.ret = False
        self.frame = None
        print('self.fs_type = ',self.fs_type,self.fs_type=='vidfil')
        try:
            if self.fs_type == 'imseq':
                self.frame = self.image_sequence[self.frame_pointer]
            elif self.fs_type == 'imfil':
                self.frame = cv2.imread(os.path.join(self.source_dir, self.image_file_list[self.frame_pointer]))
            elif self.fs_type == 'vidfil':
                print(f'Opening video file {self.video_file}')
                self.video_sequence = cv2.VideoCapture(self.video_file)
                for i in range(self.frame_pointer+1):
                    self.ret,self.frame = self.video_sequence.read()
            elif self.fs_type == 'vidseq':
                for i in range(self.frame_pointer+1):
                    self.ret,self.frame = self.video_sequence.read()
            else:
                print(f'Unrecognized frame sequence type (fs_type) in initialize: {self.fs_type}')
        except Exception as e:
            print('Reading first frame failed in initialize: ',e)
        if self.gray_convert:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.frame = np.array(self.frame) # convert image object to plain array
        if self.float_convert:
            self.frame = self.frame.astype('float')
        self.ret = True
        if return_frame:
            return self.ret,self.frame
    
    def apply(self,placeholder,frameno=None):
        """
        Load and return and/or store the next requested frame.
        """
        self.frameno = frameno
        self.ret = False
        self.frame = None
        try:
            self.frame_pointer += self.interval
            self.time = self.frame_pointer * self.fps
            if self.fs_type == 'imseq':
                self.frame = self.image_sequence[self.frame_pointer]
            elif self.fs_type == 'imfil':
                self.frame = cv2.imread(os.path.join(self.source_dir, self.image_file_list[self.frame_pointer]))
            elif self.fs_type in ['vidseq','vidfil']:
                for i in range(self.interval):
                    self.ret,self.frame = self.video_sequence.read()
                    if not self.ret:
                        self.frame = None
                        return self.ret,self.frame
            else:
                print('Unrecognized fs_type in FrameSequence.apply...')
        except Exception as e:
            print('Reading next frame failed in FrameSequence.apply: ',e)
        if self.gray_convert:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.frame = np.array(self.frame)
        if self.float_convert:
            self.frame = self.frame.astype('float')
        if self.pars['display']:
            self.display()
        #self.ret = True # comment out to preserve the ret from imread
        return self.ret,self.frame

    def display(self):
        """
        Display the current frame returned from the frame sequence.
        """
        plt.figure(self.fig)
        self.fig.clf()
        if len(self.frame.shape) == 3: # A color image...
            plt.imshow(np.round(self.frame).astype('uint8'))
        else:
            plt.imshow(np.round(self.frame).astype('uint8'),cmap='gray', vmin=0, vmax=255)
        #plt.imshow(self.frame)#,cmap='gray',vmin=0,vmax=255)
        self.fig.canvas.manager.set_window_title(f'Frame Sequence result: Frame {self.frameno} {self.frame_pointer}')
        plt.pause(1e-2)

class BinaryFrame():
    """
    A class to implement thresholding (conversion of color or grayscale images
    to black and white images. Pixel values in the interval [minThreshold,maxThreshold]
    are set to 1; pixels outside that interval are set to 0.

    Other items in the pars dictionary include:
        -- "display": If True, display grayscale and binary images in the specified figures
        -- "fill_holes": If True, flood-fill enclosed areas of 0's with 1's
    """
    def __init__(self,pars={}):
        """
        Create a BinaryFrame instance. 

        Parameters are stored in the `pars' dictionary attribute, with some defaults 
        set at initialization, that are superceded by pars dictionary argument (if any).

        Class methods perform two basic functions:
            -- Apply the binary filter to a new frame
            -- Display the current frame as grayscale and binary images
        """
        # Default parameters
        self.pars = {'minThreshold':20,'maxThreshold':255,'display':False,'fill_holes':True,
                     'bin_fig_num':101,'gray_fig_num':102,'display':False}
        if bool(pars):
            self.pars.update(pars)
        if bool(self.pars['display']):
            self.fig = plt.figure()
        self.frameno = None

    def apply(self,frame,return_bin=True,frameno=None):
        """
        Apply thresholding to the frame submitted as an argument.
        """
        self.frameno = frameno
        self.ret = False
        # Check if image is grayscale, or needs to be converted
        if len(frame.shape) == 3:
            self.gray_frame = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2GRAY)
        else:
            self.gray_frame = frame.astype('uint8')
        self.bin_frame=cv2.threshold(self.gray_frame,self.pars['minThreshold'],self.pars['maxThreshold'],
                                     cv2.THRESH_BINARY)[1]
        if self.pars['fill_holes']: # Fill holes within thresholded blobs
            # after example at https://www.programcreek.com/python/example/89425/cv2.floodFill
            frame_floodfill=self.bin_frame.copy()
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = self.bin_frame.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            # Floodfill from point (0, 0)
            cv2.floodFill(frame_floodfill, mask, (0,0), 255);
            # Invert floodfilled image
            frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)
            # Combine the two images to get the foreground.
            self.bin_frame = self.bin_frame.astype(np.uint8) | frame_floodfill_inv.astype(np.uint8)
        self.ret = True
        if self.pars['display']:
            self.display()
        if return_bin:
            return self.ret,self.bin_frame

    def display(self):
        """
        Display the results of thresholding the current frame.
        """
        plt.figure(self.fig)
        self.fig.clf()
        plt.imshow(self.bin_frame,cmap='gray',vmin=0,vmax=255)
        self.fig.canvas.manager.set_window_title(f'Binary Frame result: Frame {self.frameno}')
        plt.pause(1e-2)


class Segmenter():
    """
    A class to implement segmentation of images to identify objects and generate enclosing ROIs.
    Typically, segmentation is applied here to binary images to enable visualization of the 
    intermediate steps in an image processing sequence. Contours surrounding objects are
    obtained using cv2's findContours utility, from which basic statistics (area, bounding
    box, minimum enclosing rectangle) are saved for export, along with ROIs if requested.
    """
    def __init__(self,pars={}):
        self.pars = {'minThreshold':20, 'maxThreshold':255, 'minArea':10, 'maxArea':5000,'ROIpad':5,
                     'display_ROIs':False,'display_blobs':False,'display_MARs':False,
                     'fig_numROI':102,'fig_numBLOB':103,'fig_numCTR':104}
        if bool(pars):
            self.pars.update(pars)
        if bool(self.pars['display_ROIs']) or bool(self.pars['display_blobs']) or bool(self.pars['display_MARs']):
            self.fig = plt.figure()
        self.frameno = None

    def apply(self,bin_frame,frameno=None):
        self.bin_frame = bin_frame
        self.frameno = frameno
        self.ret = False
        ny,nx = bin_frame.shape  # these may be backwards!
        self.ROIlist=[] # clear ROI list at each call
        self.CTRlist=[] # clear ROI list at each call
        # 
        self.contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Parse ROIs from contours
        for i,ctr in enumerate(self.contours):
            #print(ctr[:,0,0])
            area=cv2.contourArea(ctr)
            # skip if area outside bounds
            if area < self.pars['minArea'] or area > self.pars['maxArea']:
                continue
            bbox=cv2.boundingRect(ctr)
            MAR = cv2.minAreaRect(ctr)
            self.ROIlist.append([i,area,bbox,MAR])
            self.CTRlist.append(ctr)
        if bool(self.pars['display_ROIs']) or bool(self.pars['display_blobs']) or bool(self.pars['display_MARs']):
            self.display()
        self.ret = True
        return self.ret,self.ROIlist

    def display(self):
        """
        Display the results of thresholding the current frame.
        """
        plt.figure(self.fig)
        self.fig.clf()
        plt.imshow(self.bin_frame,cmap='gray',vmin=0,vmax=255)
        self.fig.canvas.manager.set_window_title(f'Segmentation result: Frame {self.frameno}')
        for j,ROIinfo in enumerate(self.ROIlist):
            ctr = self.CTRlist[j]
            if bool(self.pars['display_blobs']):
                polygon = Polygon(np.squeeze(ctr, axis=1),True,linewidth=1,edgecolor='m',facecolor='none')
                # Add the patch to the Axes
                plt.gca().add_patch(polygon)
            if bool(self.pars['display_ROIs']):
                bbox=ROIinfo[2]
                print(f'bbox: {bbox}')
                rect = Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],
                             linewidth=1,edgecolor='c',facecolor='none')
                # Add the patch to the Axes
                print(f'rect: {rect}')
                plt.gca().add_patch(rect)
            if bool(self.pars['display_MARs']):
                print(f'raw MAR: {ROIinfo[3]}')
                MAR = ROIinfo[3]
                rotbox = np.int0(cv2.boxPoints(MAR))
                minArea_poly = Polygon(rotbox, linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                plt.gca().add_patch(minArea_poly)
        plt.tight_layout(pad=plt_pad)
        self.fig.canvas.draw()
        plt.pause(1e-2)

class VideoProcessor():
    """
    A class implementing background subtraction, segmentation and other processing of
    videos and image sequences to facilitate tracking and classification of moving
    objects.

    During processing, the active frame is propagated as a float array with the same
    dimensions as the original frame, to prevent accumulation of rounding errors. As
    a final step, it is converted to the original uint8 data type.
    """
    def __init__(self,pars=None,proc_seq=[],proc_objs=[]):
        """
        Create a VideoProcessor instance. 

        Parameters are stored in the `pars' dictionary attribute, with some defaults 
        set at initialization, that are superceded by pars dictionary argument (if any).

        Class methods perform four basic functions:
            -- Assemble a sequence consisting of image processing steps
            -- Apply the processing sequence to one or more image sequence
            -- Generate and load project files specifying processing sequences
            -- Produce output files of particle statistics and ROIs
        """
        self.pars = {'export_file':None,'fmt':'fos-part',
                     'export_ROIs':False,'output_dir':None,'output_prefix':None,'ROIpad':5,
                     'retain_result':1,'verbosity':1,'blur_min':1.,'range_min':0.}
        if bool(pars):
            self.pars.update(pars)
        if bool(self.pars['export_ROIs']):
            if bool(self.pars['output_dir']) == False:
                print('ROIwrite init failed: an ouput directory is required...')
                return False, None
            if bool(self.pars['output_prefix']) == False:
                print('ROIwrite init failed: an ouput prefix is required...')
                return False, None
        if bool(self.pars['export_file']):
            self.outfile = open(self.pars['export_file'],'w')
            self.outfile.write(part_hdrs[self.pars['fmt']])

        self.proc_seq = proc_seq
        self.proc_seq.append(['VideoProcessor',pars])
        self.proc_objs = proc_objs
        self.proc_objs.append('VP') # placeholder

        self.verbosity = self.pars['verbosity']
        self.frame_count = 0
        self.ROIlist = []
        self.ROIcounter = 0

    def load_seq(self,filename='VP.json'):
        """
        Load a project file in json format.
        """
        print(f'Loading json VideoProcess sequence file {filename}...')
        seq_file = open(filename,'r')
        proc_seq = json.load(seq_file)
        seq_file.close()
        print('...completed')
        print('Implementing loaded sequence:')
        for proc,pars in proc_seq:
            print(proc,pars)
            if proc == 'VideoProcessor':
                self.proc_obj = ['VP']
            elif proc == 'FrameSequence':
                fseq = FrameSequence(pars=pars)
                self.addFrameSequence(fseq,pars=pars)
            elif proc == 'TemporalFilter':
                self.addTemporalFilter(pars=pars)
            elif proc == 'SpatialFilter':
                self.addSpatialFilter(pars=pars)
            elif proc == 'BinaryFrame':
                self.addBinaryFrame(pars=pars)
            elif proc == 'Segmenter':
                self.addSegmenter(pars=pars)
        

    def save_seq(self,filename='VP.json'):
        """
        Save a project file in json format.
        """
        print(f'Saving json VideoProcess sequence file {filename}...')
        with open(filename,'w') as seq_file:
            self.jstr = json.dump(self.proc_seq,seq_file,indent=2)
        seq_file.close()
        print('...completed')
        
    def addFrameSequence(self,fseq,pars={}):
        """
        Add a FrameSequence to the processing sequence.
        """
        print('Adding FrameSequence')
        self.fseq = fseq
        self.proc_seq.append(['FrameSequence',fseq.pars])
        self.proc_objs.append(fseq)  # placeholder
        self.ret,self.frame = self.fseq.initialize()

    def addTemporalFilter(self,pars={}):
        """
        Add a temporal filter to the processing sequence.
        """
        print('Adding TemporalFilter')
        self.proc_seq.append(['TemporalFilter',pars])
        self.proc_objs.append(TemporalFilter(self.frame,pars=pars))

    def addSpatialFilter(self,pars={}):
        """
        Add a spatial convolution filter to the processing sequence.
        """
        print('Adding SpatialFilter')
        self.proc_seq.append(['SpatialFilter',pars])
        self.proc_objs.append(SpatialFilter(pars=pars))

    def addBinaryFrame(self,pars={}):
        """
        Add a thresholding filter to the processing sequence.
        """
        print('Adding BinaryFrame')
        self.proc_seq.append(['BinaryFrame',pars])
        self.proc_objs.append(BinaryFrame(pars=pars))
        
    def addSegmenter(self,pars={}):
        """
        Add a segmenting filter to the processing sequence.
        """
        print('Adding Segmenter')
        self.proc_seq.append(['Segmenter',pars])
        self.proc_objs.append(Segmenter(pars=pars))
        
    def processFrame(self):
        """
        Apply the current processing sequence to the current frame.
        """
        self.frame_count += self.fseq.interval
        self.result = None
        for i,proc_obj in enumerate(self.proc_objs):
            # The proc_objs entry for the VideoProcessor exists only to synchronize
            # with the proc_seq list, which has necessary parameters; so skip it
            print(i,proc_obj)
            if proc_obj == 'VP':
                continue
            print(f'Processing step {i}')
            ret,self.result = proc_obj.apply(self.result,frameno=self.frame_count)
            if not ret:
                return False, None
            # Save specified frame for extracting ROIs
            if self.pars['retain_result'] == i:
                print('retaining result, i = {i}, proc = {self.proc_seq[i][0]}')
                self.retain_result = self.result.copy()
            # After a call to the Segmenter, save the resulting ROIlist
            if self.proc_seq[i][0] == 'Segmenter':
                self.ROIlist = proc_obj.ROIlist
        return ret,self.result
    
    def outputFrame(self):
        """
        Output analysis results to the terminal and files as specified.
        """
        if self.verbosity>0:
            print(f'Frame number, time = {self.fseq.frame_pointer}, {self.fseq.time}')
        if self.verbosity > 2:
            print(self.result)
        # Output of ROIs (if any) to stats file and image directory
        for j,roi_stats in enumerate(self.ROIlist):
            #print('Got here: ',j,roi_stats)
            [ii,area,bbox,MAR] = roi_stats
            self.ROIcounter += 1
            i = self.ROIcounter
            if bool(self.pars['export_file']):
                #print('roi # ',j, i)
                if j == 0: # insert new frame marker
                    output_frame_mark = eval(part_frm_mrk[self.pars['fmt']])+'\n'
                    #print('output_frame_mark = ',output_frame_mark)
                    self.outfile.write(output_frame_mark)
                output_format = eval(part_fmts[self.pars['fmt']])+'\n'
                #print(output_format)
                self.outfile.write(output_format)
            if bool(self.pars['export_ROIs']):
                # Calculate ROI for export
                ns = self.retain_result.shape 
                ny,nx = ns[0:2]
                i_beg=np.max([bbox[0]-self.pars['ROIpad'],0])
                i_end=np.min([bbox[0]+bbox[2]+self.pars['ROIpad'],nx-1])
                j_beg=np.max([bbox[1]-self.pars['ROIpad'],0])
                j_end=np.min([bbox[1]+bbox[3]+self.pars['ROIpad'],ny-1])
                self.ROIimage = self.retain_result[j_beg:j_end,i_beg:i_end]
                range_stat = self.ROIimage.max()-self.ROIimage.min()
                if range_stat < self.pars['range_min']:
                    continue
                blur_stat = self.focus_index(self.ROIimage)
                if blur_stat < self.pars['blur_min']:
                    continue
                #print(i_beg,i_end,j_beg,j_end)
                output_filename = \
                f"{self.pars['output_prefix']}_f{self.fseq.frame_pointer}_n{i}_b{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{blur_stat}_{range_stat}.png"
                #f"{self.pars['output_prefix']}_f{self.fseq.frame_pointer}_n{i}_b{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{blur_stat}.png"
                #output_filename = f"{self.pars['output_prefix']}_f{self.fseq.frame_pointer}_n{i}_b{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{blur_stat}.tif"
                #output_filename = f"{self.pars['output_prefix']}_f{self.fseq.frame_pointer}_n{i}_b{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.tif"
                self.ROIfilename = os.path.join(self.pars['output_dir'],output_filename)
                ##output_suffix = f'_f{self.fseq.frame_pointer}_n{i}_b{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.tif'
                ##self.ROIfilename = os.path.join(self.pars['output_dir'],self.pars['output_prefix'],output_suffix)
                #self.ROIimage = self.retain_result[j_beg:j_end,i_beg:i_end]
                print('writing roi: ',self.ROIfilename)
                ret = cv2.imwrite(self.ROIfilename,np.round(self.ROIimage).astype('uint8'))
                #print('imwrite ret = ',ret)

    def focus_index(self,roi):
        """
        A simple statistic to assess focus.
        """
        float_roi = roi.astype('float')
        blurred_roi = cv2.GaussianBlur(roi,(3,3),cv2.BORDER_ISOLATED)
        blur_stat = np.mean(np.abs((float_roi-blurred_roi)/(blurred_roi+1.e-6)))
        return blur_stat
        
    def cleanup(self):
        """
        Close files and perform other cleanup as needed.
        """
        if bool(self.pars['export_file']):
            self.outfile.close()
                
    def processSequence(self,maxFrame=5000):
        """
        Execute the processing sequence to an entire set of images.
        """
        count = 0
        while count<maxFrame:
            ret,self.frame = self.processFrame()
            count += 1
            if not ret:
                break
            self.outputFrame()
        print(f'Exiting after processing {self.frame_count} frames.')
        self.cleanup()
                              
