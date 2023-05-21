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
                 pars={'n':11,'r':5,'ftype':'disk','offset':0,
                       'fillvalue':0,'plot_matrix':False,'verbose':False}):
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
            self.getDiskFilter(verbose=self.pars['verbose'])
        if self.pars['plot_matrix']:
            plt.figure()
            plt.imshow(self.conv_matrix)
            plt.pause(1e-1)
            self.pars.update({'plot_matrix':False})

    def getDiskFilter(self,verbose=False):
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


class BinaryFrame():
    def __init__(self,frame=None,get_bin=True,display=False,pars=None):
        self.pars = {'minThreshold':20,'maxThreshold':255,'display':False,'fill_holes':True,'bin_fig_num':101,'gray_fig_num':102}
        if pars is not None:
            self.pars.update(pars)
        if frame is not None:
            self.frame = frame
            if get_bin:
                self.binary_frame(return_bin=False,display=display)
        #self.blobs_fig_num = fig_numBLOB
        #self.ROI_fig_num = fig_numROI
        # min_val=minThreshold, max_val=maxThreshold,min_area=minArea,
        # display_ROIs=False,display_blobs=False,display_blobsCV=False,use_binary=True,
        # fig_numROI=102,fig_numBLOB=103,fig_numCTR=104

    def binary_frame(self,frame,return_bin=True,display=False):
        self.ret = False
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bin_frame=cv2.threshold(self.gray_frame,self.pars['minThreshold'],self.pars['maxThreshold'], cv2.THRESH_BINARY)[1]
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
            #print('after:',self.binary_image)
        self.ret = True
        if display:
            self.show_binary_frame()
        if return_bin:
            self.ret,self.bin_frame
            
    def show_binary_frame(self):
        plt.figure(self.pars['gray_fig_num'])
        plt.imshow(self.gray_frame,cmap='gray',vmin=0,vmax=255)
        plt.figure(self.pars['bin_fig_num'])
        plt.imshow(self.bin_frame,cmap='gray',vmin=0,vmax=255)


"""    

class Segmenter():
    def __init__(self,frame=None,pars=None, min_val=minThreshold, max_val=maxThreshold,min_area=minArea,
                      display_ROIs=False,display_blobs=False,display_blobsCV=False,use_binary=True,
                             fig_numROI=102,fig_numBLOB=103,fig_numCTR=104):
        self.frame = frame
        self.pars = pars
        self.blobs_fig_num = fig_numBLOB
        self.ROI_fig_num = fig_numROI

    def getFrame(frame):
        self.frame = frame

    def binary_frame(self, min_val=minThreshold, max_val=maxThreshold,display=False,fill_holes=True,fig_num=101):
        self.binary_frame=cv2.threshold(self.frame,self.pars['min_val'],self.pars['max_val'], cv2.THRESH_BINARY)[1]
        if fill_holes: # Fill holes within thresholded blobs
            # after example at https://www.programcreek.com/python/example/89425/cv2.floodFill
            frame_floodfill=self.binary_image.copy()
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = self.binary_image.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            # Floodfill from point (0, 0)
            cv2.floodFill(frame_floodfill, mask, (0,0), 255);
            # Invert floodfilled image
            frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)
            # Combine the two images to get the foreground.
            self.binary_image = self.binary_image.astype(np.uint8) | frame_floodfill_inv.astype(np.uint8)
            #print('after:',self.binary_image)
            self.binary_fig_num=fig_num
        if display:
            self.show_binary_frame()
            
    def show_binary_frame(self):
        plt.figure(self.binary_fig_num,facecolor=tuple([i/255 for i in bg_color]))
        plt.imshow(self.binary_image, cmap='gray')
        plt.tight_layout(pad=plt_pad)
        #title_str='Figure '+str(self.binary_fig_num)+ \
        #    ',   Filename: '+self.frame_file
        #plt.gcf().canvas.manager.set_window_title(title_str)
    
                                    
    def segment(self,bin_frame=None,pars=None):
        if bin_frame is not None:
            self.binary_frame = bin_frame
        if pars is not None:
            self.pars.update(pars)
        # 
        self.ROIlist=[]
        self.blob_keypoints = []

        self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #if display_blobs:
        #    self.show_blobs_frame()
            
        # dimensions of image in pixels
        nx=len(self.frame_image)
        ny=len(self.frame_image[0])

        # Parse ROIs from contours
        for ctr in self.contours:
            #print(ctr[:,0,0])
            area=cv2.contourArea(ctr)
            bbox=cv2.boundingRect(ctr)
            try:
                ellbox = cv2.fitEllipse(ctr)
                ell = Ellipse((ellbox[0][0],ellbox[0][1]),ellbox[1][0],ellbox[1][1],angle=ellbox[2],
                              linewidth=1,edgecolor='y',facecolor='none')
                if display_ROIs:
                    plt.gca().add_patch(ell)
            except:
                ellbox=None
            i_beg=np.max([np.min(ctr[:,0,1])-ROIpad,0])
            i_end=np.min([np.max(ctr[:,0,1])+ROIpad,nx-1])
            j_beg=np.max([np.min(ctr[:,0,0])-ROIpad,0])
            j_end=np.min([np.max(ctr[:,0,0])+ROIpad,ny-1])
            
            # get blob subimage
            blob_img = Image.fromarray( self.color_frame_image[i_beg:i_end, j_beg:j_end])
            #blob_img = Image.fromarray( cv2.cvtColor(self.color_frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2RGB))
            #blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2RGB))
            #blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2GRAY))
            
            self.ROIlist.append(ROI(ROIimage=blob_img,edge=np.squeeze(ctr,axis=1),
                                    area=area,bbox=bbox,ellbox=ellbox,
                                    i_beg=i_beg,i_end=i_end,j_beg=j_beg,j_end=j_end,
                                    category=category))    
        if display_ROIs:
            self.show_ROIs_frame()
"""
    
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
                
