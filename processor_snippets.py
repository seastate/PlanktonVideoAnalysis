

import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter, BinaryFrame, Segmenter, VideoProcessor


video_file = '1537773747.avi'
#video_name = '/home/dg/Courses/LarvalEcology/TankDemos/GlopVideos/2019-07-24-145131.mp4'
fseq = FrameSequence(pars={'video_file':video_file,'init':False,'gray_convert':False,
                 'frame_pointer':0,'interval':20,'fs_type':None,
                 'float_convert':True,'fps':20.})

vp = VideoProcessor(verbosity=3,pars={'retain_result':0,
                 'export_file':'test.fos-part','export_ROIs':True,'output_dir':'TMP', 'output_prefix':'test','display':True})

vp.addFrameSequence(fseq,pars={})
vp.addTemporalFilter(pars={'display':True})
#vp.addSpatialFilter(pars={'display':True,'offset':0,'mode':'replace'})
#vp.addSpatialFilter(pars={'display':False,'offset':128,'mode':'subtract'})
vp.addBinaryFrame(pars={'display':True,'minThreshold':135})
vp.addSegmenter(pars={'display':True})
#vp.addROIwriter(pars={'export_file':'test.fos-part','export_ROIs':True,'output_dir':'TMP', 'output_prefix':'test','display':True})

vp.processSequence()


#============================================================================================

import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

from VideoProcessTools import FrameSequence, TemporalFilter, SpatialFilter, BinaryFrame, Segmenter, VideoProcessor


video_name = '1537773747.avi'
#video_name = '/home/dg/Courses/LarvalEcology/TankDemos/GlopVideos/2019-07-24-145131.mp4'
fseq = FrameSequence(video_file=video_name,init=False,
                     frame_pointer=0,gray_convert=False,interval=20,fps=20,float_convert=True)

vp = VideoProcessor(verbosity=3,pars={'retain_result':None})

vp.addFrameSequence(fseq,pars={'export_file':'test.fos-part','export_ROIs':True,'output_dir':'TMP', 'output_prefix':'test','display':True})
vp.addTemporalFilter(pars={'display':True})
#vp.addSpatialFilter(pars={'display':True,'offset':0,'mode':'replace'})
#vp.addSpatialFilter(pars={'display':False,'offset':128,'mode':'subtract'})
vp.addBinaryFrame(pars={'display':True,'minThreshold':135})
vp.addSegmenter(pars={'display':True})
#vp.addROIwriter(pars={'export_file':'test.fos-part','export_ROIs':True,'output_dir':'TMP', 'output_prefix':'test','display':True})

vp.processSequence()


