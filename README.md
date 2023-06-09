# PlanktonVideoAnalysis
Python tools for tracking and identifying plankton in videos and images

This is a self-contained module for basic image processing and segmentation of sequences of frames from videos and image sets. The primary intended use is for identification and tracking of moving particles. Outputs are a csv file of particle statistics (based on the opencv metrics area, bounding box and minimum enclosing rectangle) and, if specified, a library of ROI (Region of Interest) sub-images suitable for automated or human-driven classification.

The currently implemented capabilities are: spatial (convolution) and temporal (weighted running average) filtering to remove noise and isolate "foreground" from "background"; thresholding to produce binary images; segmentation of contiguous regions of pixels; output to statistics files and/or ROIs; and, saving and loading project files of analysis parameters in json format.

The PVAexample files illustrate defining, executing and saving an analysis, and subsequently loading and re-executing that analysis. In a real application, elements such as the video file name could be edited to systematically analyze sets of videos.

