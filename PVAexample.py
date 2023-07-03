"""
An example of scripted image processing and segmentation of particles in image sequences. This set of commands
specifies a sequence of image analysis steps and a video file to which to apply them.
"""
# Import classes from the VideoProcessTools module
from VideoProcessTools import FrameSequence, VideoProcessor
# Specify a video file to process
video_file = 'myVideo1.avi'
# Instatiate a frame sequence to handle frames from the video
fseq = FrameSequence(pars={'video_file':video_file,'init':False,'gray_convert':False,
                 'frame_pointer':0,'interval':1,'fs_type':None,
                 'float_convert':True,'fps':20.,'display':True})
# Instantiate a video processor object
vp = VideoProcessor(pars={'retain_result':2,'verbosity':3,
                 'export_file':'test.fos-part','export_ROIs':True,'output_dir':'TMP', 'output_prefix':'test','display':True})
# Add the frame sequence and processing steps to the video processor analysis sequence
vp.addFrameSequence(fseq,pars={})
vp.addTemporalFilter(pars={'display':True})
#vp.addSpatialFilter(pars={'display':True,'offset':0,'mode':'replace'})
#vp.addSpatialFilter(pars={'display':False,'offset':128,'mode':'subtract'})
vp.addBinaryFrame(pars={'display':True,'minThreshold':135})
vp.addSegmenter(pars={'display_ROIs':True,'display_blobs':False,'display_MARs':True})
# Execute the analysis to the specified video
vp.processSequence()
# Save the analysis sequence in json format
vp.save_seq(filename='VP.json')
