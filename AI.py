"""
An example of scripted image processing and segmentation of particles in image sequences. This set of commands
specifies a sequence of image analysis steps and a video file to which to apply them.
"""
# Import classes from the VideoProcessTools module
from VideoProcessTools import FrameSequence, VideoProcessor

ROIdir = '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/blur_test'
#ROIdir = '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_ROIs'

# Specify a video file to process
video_files = ['/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_172503.avi',
               '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_173109.avi',
               '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_173757.avi',
               '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_174326.avi',
               '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_174851.avi']

#video_file = '/home/dg/Courses/LarvalEcology/ColumnDemo/FHL2023tank/DataAI/pisasterA_20230614_172503.avi'
video_file = video_files[0]

# Instatiate a frame sequence to handle frames from the video
fseq = FrameSequence(pars={'video_file':video_file,'init':False,'gray_convert':True,
                 'frame_pointer':0,'interval':20,'fs_type':None,
                 'float_convert':True,'fps':20.})
# Instantiate a video processor object
vp = VideoProcessor(pars={'retain_result':1,'verbosity':3,'blur_min':0.12,'range_min':200.,
                 'export_file':None,'export_ROIs':True,'output_dir':ROIdir, 'output_prefix':'pstrA','display':True})
#vp = VideoProcessor(pars={'retain_result':2,'verbosity':3,
#                 'export_file':'test.fos-part','export_ROIs':True,'output_dir':ROIdir, 'output_prefix':'test','display':True})
# Add the frame sequence and processing steps to the video processor analysis sequence
vp.addFrameSequence(fseq,pars={})
vp.addTemporalFilter(pars={'display':False})
#vp.addSpatialFilter(pars={'display':True,'offset':0,'mode':'replace'})
#vp.addSpatialFilter(pars={'display':False,'offset':128,'mode':'subtract'})
vp.addBinaryFrame(pars={'display':False,'minThreshold':138})
#vp.addBinaryFrame(pars={'display':True,'minThreshold':135})
vp.addSegmenter(pars={'display':False,'minArea':1500,'maxArea':15000})
# Execute the analysis to the specified video
vp.processSequence()

# Save the analysis sequence in json format
#vp.save_seq(filename='VP.json')
