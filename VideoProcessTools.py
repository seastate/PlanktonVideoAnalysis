import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from scipy import signal
from math import floor,sqrt

class TemporalFilter():
    """
    A class to implement accumulation of a background frame through temporal 
    filtering, and its use to isolate foreground (moving objects in video
    analysis).
    """
    def __init__(self,init_frame=None,alpha=0.1,offset=128):
        self.BGframe = init_frame.astype('float')
        self.alpha = alpha
        self.offset = offset

    def update(self,frame):
        self.BGframe = (1.-self.alpha)*self.BGframe + self.alpha*frame.astype('float')

    def subtract(self,frame,offset=None,update=True):
        if offset is not None:
            self.offset = offset
        self.BGframe_int32 = np.round(self.BGframe).astype('int32')
        self.frame_int32 = np.round(frame).astype('int32')
        self.FGframe = np.minimum(255,np.maximum(0,self.frame_int32-self.BGframe_int32+self.offset)).astype('uint8')
        #
        if update:
            self.update(frame)
        #
        return self.FGframe


class SpatialFilter():
    """
    A class to implement accumulation of a background frame through spatial 
    filtering, and its use to isolate foreground (moving objects in video
    analysis).
    """
    def __init__(self,frame=None,get_bg=True,get_filter=True,
                 pars={'n':11,'r':5,'ftype':'disk','offset':0,'fillvalue':0}):
        self.pars = pars
        if frame is not None:
            self.frame = frame
        if get_filter:
            self.getFilter()
            if get_bg and frame is not None:
                self.getBG(frame,return_bg=False)

    def getBG(self,frame,return_bg=True):
        self.ret = False
        self.frame_float = frame.astype('float')
        self.BGframe = np.zeros(frame.shape)
        #
        for i in range(3):
            self.BGframe[:,:,i] = signal.convolve2d(frame[:,:,i].astype('float'),
                                                    self.conv_matrix,
                                                    mode='same',boundary='fill',
                                                    fillvalue=self.pars['fillvalue'])
        self.ret = True
        if return_bg:
            return self.ret,self.BGframe

    def getFilter(self,pars=None):
        self.conv_matrix = None
        if pars is not None:
            self.pars.update(pars)
        if self.pars['ftype']=='disk':
            self.getDiskFilter()

    def getDiskFilter(self):
        self.offset = self.pars['offset']
        self.conv_matrix = np.zeros([self.pars['n'],self.pars['n']])
        self.n_mid = floor(self.pars['n']/2)
        print(f'Disk filter: using n_mid = {self.n_mid}')
        for irow in range(self.pars['n']):
            for icol in range(self.pars['n']):
                dist = sqrt( (irow-self.n_mid)**2 + (icol-self.n_mid)**2 );
                if dist < self.pars['r']-1:
                    self.conv_matrix[irow,icol] = 1.
        self.conv_matrix /= np.sum(self.conv_matrix)
        print(self.conv_matrix)

    def apply(self,frame,offset=None,update=True,mode='subtract'):
        self.ret = False
        if offset is not None:
            self.offset = offset
        if update:
            self.getBG(frame)
        self.BGframe_int32 = np.round(self.BGframe).astype('int32') + self.offset
        if mode == 'replace':
            self.ret = True
            return self.ret,self.BGframe_int32.astype('uint8')
        elif mode == 'subtract':
            self.frame_int32 = np.round(frame).astype('int32')
            self.FGframe = np.minimum(255,np.maximum(0,self.frame_int32-self.BGframe_int32+self.offset)).astype('uint8')
            self.ret = True
            return self.ret,self.FGframe
        else:
            return self.ret,None


class FrameSequence():
    """
    A class to facilitate work combining images sequences, frames in videos and
    lists of image files during image processing.
    """
    def __init__(self,image_file_list=None,image_sequence=None,
                 video_file=None,video_sequence=None,
                 source_dir=None,init=True,
                 frame_pointer=0,interval=1,fs_type=None):
        self.image_file_list = image_file_list
        self.image_sequence = image_sequence
        self.video_file = video_file
        self.video_sequence = video_sequence
        self.source_dir = source_dir
        
        self.frame_pointer = frame_pointer
        self.interval = interval
        self.frame = None
        self.video = None
        self.ret = None
        
        # Set type of frame sequence. If not set explicitly, the type is determined
        # by the first of the following arguments that is not None:
        if fs_type is not None:
            self.fs_type = fs_type
        elif image_sequence is not None:
            self.fs_type = 'imseq'
        elif image_file_list is not None:
            self.fs_type = 'imfil'
        elif video_sequence is not None:
            self.fs_type = 'vidseq'
        elif video_file is not None:
            self.fs_type = 'vidfil'
        else:
            print('Frame sequence type not set: manually set attribute ' + \
                  'fs_type to imseq, imfil, vidseq or vidfil')
        if init:
            self.initialize(return_frame=False)
        
    def initialize(self,return_frame=True):
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
                print(f'Unrecognized fs_type in initialize: {self.fs_type}')
        except Exception as e:
            print('Reading first frame failed in initialize: ',e)
        self.frame = np.array(self.frame)
        if return_frame:
            return self.ret,self.frame
    
    def nextFrame(self):
        self.ret = False
        self.frame = None
        try:
            self.frame_pointer += self.interval
            if self.fs_type == 'imseq':
                self.frame = self.image_sequence[self.frame_pointer]
            elif self.fs_type == 'imfil':
                self.frame = cv2.imread(os.path.join(self.source_dir, self.image_file_list[self.frame_pointer]))
            elif self.fs_type in ['vidseq','vidfil']:
                for i in range(self.interval):
                    self.ret,self.frame = self.video_sequence.read()
            else:
                print('Unrecognized fs_type in nextFrame...')
        except Exception as e:
            print('Reading next frame failed in nextFrame: ',e)
        self.frame = np.array(self.frame)
        return self.ret,self.frame

            
class VideoProcessor():
    """
    A class implementing background subtraction and other processing of
    videos and image sequences to facilitate tracking and classification
    of moving objects.
    """
    def __init__(self, video_name = None, image_dir = None,pars_list=[]):
        
        self.frame = None


    def FilterFrame(inframe=None):
        pass
                
