

import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter, BinaryFrame, Segmenter, VideoProcessor

video_name = '1537773747.avi'
fseq = FrameSequence(video_file=video_name,init=False,
                     frame_pointer=0,gray_convert=False,interval=20,float_convert=True)

vp = VideoProcessor(verbosity=3)

vp.addFrameSequence(fseq)
vp.addTemporalFilter(pars={'display':True})
#vp.addSpatialFilter(pars={'display':False,'offset':128,'mode':'replace'})
vp.addBinaryFrame(pars={'display':True,'minThreshold':135})
#vp.addSegmenter(pars={'display':True})

vp.processSequence()


#============================================================================================

plt.ion()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

