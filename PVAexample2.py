"""
An example of scripted image processing and segmentation of particles in image sequences. This set of commands
illustrates loading and executing a saved analysis sequence.
"""
# Import classes from the VideoProcessTools module
from VideoProcessTools import VideoProcessor
# Define a new video processor object
vp2 = VideoProcessor()
# Load the saved processing sequence
vp2.load_seq(filename='VP.json')
# Execute the processing sequence
vp2.processSequence()
