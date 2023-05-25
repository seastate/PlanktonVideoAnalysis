

# segmenter test

import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter, BinaryFrame, Segmenter


plt.ion()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

video_name = '1537773747.avi'
fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=0,gray_convert=False,interval=20)
bf = BinaryFrame(pars = {'minThreshold':10,'maxThreshold':255,'fill_holes':False},display=True)
seg = Segmenter()

ret,frame = fseq.initialize()


ret,binframe = bf.binary_frame(frame,display=True)
seg.getFrame(binframe)
ROIlist = seg.segment()


bin_frame=cv2.threshold(gray_frame,120,255, cv2.THRESH_BINARY)[1]
plt.figure(101)
plt.imshow(bin_frame,cmap='gray', vmin=0, vmax=255)

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plt.figure(102)
plt.imshow(gray_frame,cmap='gray', vmin=0, vmax=255)



# temporal filter example

import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter, BinaryFrame


plt.ion()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

video_name = '1537773747.avi'
fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=0,gray_convert=True,interval=20)
#video_name = '/home/dg/FHL2020/Pisaster_videos/8-8_surface_57d_20ppt/BTV_Movie_001.mov'
#fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=0,gray_convert=True)
#video_name = '/home/dg/Courses/LarvalEcology/TankDemos/GlopVideos/2019-07-24-145131.mp4'
#fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=2250)
video_name = '/home/dg/PublicSensors/Instruments/Deployments/Duamish/MOVI0966.AVI'
fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=0,gray_convert=False)

ret,frame = fseq.initialize()

tf = TemporalFilter(init_frame=frame,alpha=0.01)

bf = BinaryFrame(pars = {'minThreshold':140,'maxThreshold':255,'fill_holes':False})

count = 0

while ret and count<1500:
    plt.figure(fig1)
    print(f'showing frame {count}')
    plt.cla()
    if len(frame.shape)==3:
        plt.imshow(frame)
    else:
        plt.imshow(frame,cmap='gray', vmin=0, vmax=255)
    fig1.canvas.manager.set_window_title(f'Image: Frame {count}')
    if count>0:
        plt.figure(fig2)
        plt.cla()
        fig2.canvas.manager.set_window_title(f'Background: Frame {count}')
        #plt.imshow((np.round(sf.BGframe)).astype('uint8'))
        if len(tf.BGframe.shape)==3:
            plt.imshow((np.round(tf.BGframe)).astype('uint8'))
        else:
            plt.imshow((np.round(tf.BGframe)).astype('uint8'),cmap='gray', vmin=0, vmax=255)
        #
        plt.figure(fig3)
        plt.cla()
        fig3.canvas.manager.set_window_title(f'Foreground: Frame {count}')
        if len(FGframe.shape)==3:
            plt.imshow(FGframe)
        else:
            plt.imshow(FGframe,cmap='gray', vmin=0, vmax=255)
        bf.binary_frame(FGframe,display=True)
    
    plt.pause(1e-1)
    #cv2.imwrite("Frames/frame%d.jpg" % count, frame)     # save frame as JPEG file      
    ret,frame = fseq.nextFrame()
    FGframe = tf.subtract(frame)
    count += 1


gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plt.figure(102)
plt.imshow(gray_frame,cmap='gray', vmin=0, vmax=255)
bin_frame=cv2.threshold(gray_frame,120,255, cv2.THRESH_BINARY)[1]
plt.figure(101)
plt.imshow(bin_frame,cmap='gray', vmin=0, vmax=255)




while ret and count<1500:
    plt.pause(1e-1)
    #cv2.imwrite("Frames/frame%d.jpg" % count, frame)     # save frame as JPEG file      
    ret,frame = fseq.nextFrame()
    FGframe = tf.subtract(frame)
    bf.binary_frame(FGframe,display=True)
    count += 1


================================================================================
# spatial filter example
import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter


plt.ion()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

video_name = '/home/dg/FHL2020/Pisaster_videos/8-8_surface_57d_20ppt/BTV_Movie_001.mov' #'1537773747.avi'
fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=0)
#video_name = '/home/dg/Courses/LarvalEcology/TankDemos/GlopVideos/2019-07-24-145131.mp4'
#fseq = FrameSequence(video_file=video_name,init=False,frame_pointer=2250)
ret,frame = fseq.initialize()

tf = TemporalFilter(init_frame=frame,alpha=0.01)

#ret, frame = cap.read()
count = 0

sf_pars={'n':25,'r':None,'ftype':'disk','offset':128,'fillvalue':0,'plot_matrix':True,'verbose':False}

while ret and count<1500:
    plt.figure(fig1)
    print('showing frame')
    plt.cla()
    plt.imshow(frame)
    fig1.canvas.manager.set_window_title(f'Frame {count}')
    #
    if count>0:
        plt.figure(fig2)
        print('showing BGframe')
        plt.cla()
        #plt.imshow(sf.BGframe_int32)
        #plt.imshow(tf.BGframe_int32)
        fig2.canvas.manager.set_window_title(f'Frame {count}')
        plt.imshow((np.round(sf.BGframe)).astype('uint8'))
        #plt.imshow((np.round(tf.BGframe)).astype('uint8'))
        #
        plt.figure(fig3)
        print('showing FGframe')
        plt.cla()
        fig3.canvas.manager.set_window_title(f'Frame {count}')
        plt.imshow(FGframe)
    
    plt.pause(1e-1)
    #cv2.imwrite("Frames/frame%d.jpg" % count, frame)     # save frame as JPEG file      
    ret,frame = fseq.nextFrame()
    #FGframe = tf.subtract(frame)
    sf = SpatialFilter(frame=frame, pars=sf_pars)
    ret,FGframe = sf.apply(frame)
    print('Read a new frame: ', ret)
    count += 1










