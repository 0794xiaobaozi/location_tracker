"""

LIST OF FUNCTIONS

LoadAndCrop
cropframe
Reference
Locate
TrackLocation
LocationThresh_View
ROI_plot
ROI_Location
Batch_LoadFiles
Batch_Process
PlayVideo
PlayVideo_ext
showtrace
Heatmap
DistanceTool
ScaleDistance

"""





########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import time
import warnings
import functools as fct
from scipy import ndimage
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display

# Initialize Holoviews - only in Jupyter notebook environment
# In multiprocessing child processes, this will fail silently
try:
    hv.notebook_extension('bokeh')
except:
    # Fallback to general extension (works in non-notebook contexts)
    try:
        hv.extension('bokeh')
    except:
        pass  # In multiprocessing subprocesses, we don't need visualization

warnings.filterwarnings("ignore")


########################################################################################
# Mock classes for pickle-able data structures (needed for multiprocessing)
########################################################################################

class MockCrop:
    """Mock crop object that can be pickled for multiprocessing"""
    def __init__(self, data):
        self.data = data

class MockStream:
    """Mock stream object that can be pickled for multiprocessing"""
    def __init__(self, data):
        self.data = data





########################################################################################    

def LoadAndCrop(video_dict,cropmethod=None,fstfile=False,accept_p_frames=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Loads video and creates interactive cropping tool (video_dict['crop'] from first frame. In the 
    case of batch processing, the first frame of the first video is used. Additionally, 
    when batch processing, the same cropping parameters will be appplied to every video.  
    Care should therefore be taken that the region of interest is in the same position across 
    videos.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping 
                'Box' : Create box selection tool for cropping video
                
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame
            
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
    
    -------------------------------------------------------------------------------------
    Notes:
        - in the case of batch processing, video_dict['file'] is set to first 
          video in file 
        - prior cropping method HLine has been removed
    
    """
    
    #if batch processing, set file to first file to be processed
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print('total frames: {frames}'.format(frames=cap_max))
    print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    
    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame
    cap.release()

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(
        width=int(frame.shape[1]*video_dict['stretch']['width']),
        height=int(frame.shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        colorbar=True,
        toolbar='below',
        title="First Frame.  Crop if Desired"
    )
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        video_dict['crop'] = None
        return image, video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        video_dict['crop'] = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box), video_dict
    
    
def LoadAndCrop_cv2(video_dict, fstfile=False, accept_p_frames=False):
    """
    -------------------------------------------------------------------------------------
    
    OpenCV version of LoadAndCrop. Uses mouse drag to select crop region.
    More reliable interaction than Holoviews BoxEdit.
    
    -------------------------------------------------------------------------------------
    Instructions:
        - Click and drag to draw crop rectangle
        - Press ENTER to confirm
        - Press 'r' to reset and redraw
        - Press ESC to skip cropping
    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict] with 'crop' key containing crop coordinates
    """
    
    #if batch processing, set file to first file to be processed
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    
    #Upload file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print('total frames: {frames}'.format(frames=cap_max))
    print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    
    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read() 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame_gray = cv2.resize(
                    frame_gray,
                    (
                        int(frame_gray.shape[1]*video_dict['dsmpl']),
                        int(frame_gray.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame_gray
    cap.release()
    
    # OpenCV interactive cropping
    drawing = False
    start_point = None
    end_point = None
    crop_rect = None
    
    display_img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    original_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, display_img, crop_rect
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                display_img = original_img.copy()
                cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            crop_rect = (start_point, end_point)
            display_img = original_img.copy()
            cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 2)
    
    window_name = 'Crop Selection - Drag to select, ENTER to confirm, R to reset, ESC to skip'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize window for better visibility
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    print("\n" + "="*70)
    print("OpenCV Crop Tool")
    print("="*70)
    print("Instructions:")
    print("  - DRAG: Draw crop rectangle")
    print("  - ENTER: Confirm selection")
    print("  - 'r': Reset and redraw")
    print("  - ESC: Skip cropping (use full frame)")
    print("="*70)
    
    while True:
        temp_img = display_img.copy()
        cv2.putText(temp_img, "Drag to select crop region, ENTER to confirm, ESC to skip", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if crop_rect:
            x1, y1 = crop_rect[0]
            x2, y2 = crop_rect[1]
            cv2.putText(temp_img, f"Selected: ({min(x1,x2)}, {min(y1,y2)}) to ({max(x1,x2)}, {max(y1,y2)})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if crop_rect:
                x1, y1 = crop_rect[0]
                x2, y2 = crop_rect[1]
                # Use module-level MockCrop for pickle compatibility
                video_dict['crop'] = MockCrop({
                    'x0': [min(x1, x2)],
                    'x1': [max(x1, x2)],
                    'y0': [min(y1, y2)],
                    'y1': [max(y1, y2)]
                })
                print(f"[OK] Crop region set: ({min(x1,x2)}, {min(y1,y2)}) to ({max(x1,x2)}, {max(y1,y2)})")
                break
            else:
                print("[WARNING] No region selected. Draw a rectangle first.")
                
        elif key == ord('r') or key == ord('R'):
            display_img = original_img.copy()
            crop_rect = None
            print("[RESET] Cleared crop selection")
            
        elif key == 27:  # ESC
            video_dict['crop'] = None
            print("[SKIP] No cropping applied, using full frame")
            break
    
    cv2.destroyAllWindows()
    return video_dict


########################################################################################

def AnalysisROI_select_cv2(video_dict):
    """
    -------------------------------------------------------------------------------------
    
    OpenCV interactive tool to define Analysis ROI (Region of Interest for tracking).
    This defines the area where animal tracking will be performed, excluding areas 
    like water bottles, feeders, etc.
    
    This is SUPERIOR to mask because:
    - Centroid can never fall outside analysis ROI
    - More efficient (only process the ROI)
    - More intuitive concept
    
    -------------------------------------------------------------------------------------
    Instructions:
        - Click and drag to draw analysis region
        - Press ENTER to confirm
        - Press 'r' to reset and redraw
        - Press ESC to skip (analyze entire frame)
    
    -------------------------------------------------------------------------------------
    Returns:
        analysis_roi: tuple (x1, y1, x2, y2) or None
    """
    
    # Get first frame
    if 'f0' not in video_dict or video_dict['f0'] is None:
        if 'fpath' not in video_dict:
            video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start'])
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()
    else:
        frame_gray = video_dict['f0']
    
    # Apply existing crop if any
    if 'crop' in video_dict and video_dict['crop'] is not None:
        frame_gray = cropframe(frame_gray, video_dict['crop'])
    
    # OpenCV interactive selection
    drawing = False
    start_point = None
    end_point = None
    roi_rect = None
    
    display_img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    original_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, display_img, roi_rect
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                display_img = original_img.copy()
                # Draw green rectangle for analysis ROI
                cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 3)
                # Add text
                cv2.putText(display_img, "ANALYSIS ROI", (start_point[0], start_point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            roi_rect = (start_point, end_point)
            display_img = original_img.copy()
            cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 3)
            cv2.putText(display_img, "ANALYSIS ROI", (start_point[0], start_point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    window_name = 'Analysis ROI Selection - Define tracking area (exclude water bottle, etc.)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize window for better visibility
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    print("\n" + "="*70)
    print("OpenCV Analysis ROI Tool")
    print("="*70)
    print("Define the region where you want to track the animal.")
    print("This excludes areas like:")
    print("  - Water bottles")
    print("  - Food dispensers")
    print("  - Cage edges")
    print("  - Any other non-tracking areas")
    print("-" * 70)
    print("Instructions:")
    print("  - DRAG: Draw analysis region rectangle")
    print("  - ENTER: Confirm selection")
    print("  - 'r': Reset and redraw")
    print("  - ESC: Skip (analyze entire frame)")
    print("="*70)
    
    while True:
        temp_img = display_img.copy()
        cv2.putText(temp_img, "Drag to define ANALYSIS ROI (tracking area)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(temp_img, "ENTER to confirm, ESC to skip", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if roi_rect:
            x1, y1 = roi_rect[0]
            x2, y2 = roi_rect[1]
            cv2.putText(temp_img, f"ROI: ({min(x1,x2)}, {min(y1,y2)}) to ({max(x1,x2)}, {max(y1,y2)})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if roi_rect:
                x1, y1 = roi_rect[0]
                x2, y2 = roi_rect[1]
                analysis_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                print(f"[OK] Analysis ROI set: {analysis_roi}")
                print(f"     Only this region will be analyzed for tracking.")
                break
            else:
                print("[WARNING] No region selected. Draw a rectangle first.")
                
        elif key == ord('r') or key == ord('R'):
            display_img = original_img.copy()
            roi_rect = None
            print("[RESET] Cleared ROI selection")
            
        elif key == 27:  # ESC
            analysis_roi = None
            print("[SKIP] No analysis ROI defined, will analyze entire frame")
            break
    
    cv2.destroyAllWindows()
    return analysis_roi


########################################################################################

def AnalysisROI_polygon_select_cv2(video_dict):
    """
    -------------------------------------------------------------------------------------
    
    OpenCV interactive tool to define POLYGON Analysis ROI.
    For complex-shaped tracking areas (e.g., circular arenas, irregular cages).
    
    -------------------------------------------------------------------------------------
    Instructions:
        - Click to add polygon vertices
        - Press ENTER to complete polygon
        - Press 'r' to reset
        - Press 'p' to add current polygon and start a new one (multi-polygon)
        - Press ESC to skip
    
    -------------------------------------------------------------------------------------
    Returns:
        analysis_roi_polygon: dict with 'type', 'vertices', and 'mask' keys
    """
    
    # Get first frame
    if 'f0' not in video_dict or video_dict['f0'] is None:
        if 'fpath' not in video_dict:
            video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start'])
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()
    else:
        frame_gray = video_dict['f0']
    
    # Apply existing crop if any
    if 'crop' in video_dict and video_dict['crop'] is not None:
        frame_gray = cropframe(frame_gray, video_dict['crop'])
    
    # Polygon drawing
    current_polygon = []
    completed_polygons = []
    
    display_img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    original_img = display_img.copy()
    
    def redraw():
        nonlocal display_img
        display_img = original_img.copy()
        
        # Draw completed polygons
        for poly in completed_polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(display_img, [pts], True, (0, 200, 0), 2)
            cv2.fillPoly(display_img, [pts], (0, 100, 0, 50))
        
        # Draw current polygon
        if len(current_polygon) > 0:
            for i, pt in enumerate(current_polygon):
                cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw lines
            for i in range(len(current_polygon) - 1):
                cv2.line(display_img, current_polygon[i], current_polygon[i+1], (0, 255, 0), 2)
            
            # Draw closing line if more than 2 points
            if len(current_polygon) > 2:
                cv2.line(display_img, current_polygon[-1], current_polygon[0], (0, 255, 0), 1)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
            redraw()
    
    window_name = 'Polygon Analysis ROI - Click vertices, ENTER to finish'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize window
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    print("\n" + "="*70)
    print("OpenCV Polygon Analysis ROI Tool")
    print("="*70)
    print("Define a POLYGON region where you want to track the animal.")
    print("Useful for:")
    print("  - Circular or hexagonal arenas")
    print("  - Irregular cage shapes")
    print("  - Complex obstacle avoidance")
    print("-" * 70)
    print("Instructions:")
    print("  - CLICK: Add vertex to polygon")
    print("  - ENTER: Complete polygon (min 3 vertices)")
    print("  - 'p': Save current polygon and start new one")
    print("  - 'r': Reset current polygon")
    print("  - ESC: Skip (analyze entire frame)")
    print("="*70)
    
    while True:
        temp_img = display_img.copy()
        
        # Instructions overlay
        cv2.putText(temp_img, "Click to add polygon vertices", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        total_vertices = sum(len(p) for p in completed_polygons) + len(current_polygon)
        cv2.putText(temp_img, f"Polygons: {len(completed_polygons)}, Vertices: {total_vertices}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if len(current_polygon) >= 3:
            cv2.putText(temp_img, "Press ENTER to complete or 'P' for next polygon", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if len(current_polygon) >= 3:
                completed_polygons.append(current_polygon)
                current_polygon = []
                
                # Create combined mask
                mask = np.zeros(frame_gray.shape, dtype=np.uint8)
                for poly in completed_polygons:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                
                analysis_roi_polygon = {
                    'type': 'polygon',
                    'vertices': completed_polygons,
                    'mask': mask > 0  # Boolean mask (True = analyze region)
                }
                
                print(f"[OK] Polygon Analysis ROI created")
                print(f"     {len(completed_polygons)} polygon(s) with {sum(len(p) for p in completed_polygons)} total vertices")
                print("     Tracking will ONLY be performed inside polygons.")
                cv2.destroyAllWindows()
                return analysis_roi_polygon
            elif len(completed_polygons) > 0:
                # Already have completed polygons, finish
                mask = np.zeros(frame_gray.shape, dtype=np.uint8)
                for poly in completed_polygons:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                
                analysis_roi_polygon = {
                    'type': 'polygon',
                    'vertices': completed_polygons,
                    'mask': mask > 0
                }
                
                print(f"[OK] Polygon Analysis ROI created with {len(completed_polygons)} polygon(s)")
                cv2.destroyAllWindows()
                return analysis_roi_polygon
            else:
                print("[WARNING] Need at least 3 vertices to form a polygon.")
                
        elif key == ord('p') or key == ord('P'):  # Save current and start new
            if len(current_polygon) >= 3:
                completed_polygons.append(current_polygon)
                current_polygon = []
                redraw()
                print(f"[SAVED] Polygon {len(completed_polygons)} saved. Starting new polygon...")
            else:
                print("[WARNING] Need at least 3 vertices before starting new polygon.")
                
        elif key == ord('r') or key == ord('R'):
            current_polygon = []
            redraw()
            print("[RESET] Cleared current polygon")
            
        elif key == 27:  # ESC
            print("[SKIP] No polygon Analysis ROI defined")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None
    

########################################################################################

def cropframe(frame, crop=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Crops passed frame with `crop` specification
    
    -------------------------------------------------------------------------------------
    Args:
        frame:: [numpy.ndarray]
            2d numpy array 
        crop:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
    
    -------------------------------------------------------------------------------------
    Returns:
        frame:: [numpy.ndarray]
            2d numpy array
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
        return frame[fymin:fymax,fxmin:fxmax]
    except:
        return frame
 
    
    
    

########################################################################################

def Reference(video_dict,num_frames=100,
              altfile=False,fstfile=False,frames=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates reference frame by taking median of random subset of frames.  This has the 
    effect of removing animal from frame provided animal is not inactive for >=50% of
    the video segment.  
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        num_frames:: [uint]
            Number of frames to base reference frame on.
            
        altfile:: [bool]
            Specify whether alternative file than video to be processed will be
            used to generate reference frame. If `altfile=True`, it is expected
            that `video_dict` contains `altfile` key.
        
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        frames:: [np array]
            User defined selection of frames to use for generating reference
    
    -------------------------------------------------------------------------------------
    Returns:
        reference:: [numpy.array]
            Reference image. Median of random subset of frames.
        image:: [holoviews.image]
            Holoviews Image of reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
        - If `altfile` is specified, it will be used to generate reference.
    
    """
    
    #set file to use for reference
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    vname = video_dict.get("altfile","") if altfile else video_dict['file']    
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    frame = cropframe(
        frame, 
        video_dict.get('crop')
    )
    h,w = frame.shape[0], frame.shape[1]
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #Collect subset of frames
    if frames is None:
        #frames = np.random.randint(video_dict['start'],cap_max,num_frames)
        frames = np.linspace(start=video_dict['start'], stop=cap_max, num=num_frames)
    else:
        num_frames = len(frames) #make sure num_frames equals length of passed list
        
    collection = np.zeros((num_frames,h,w))  
    for (idx,framenum) in enumerate(frames):    
        grabbed = False
        while grabbed == False: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (video_dict['dsmpl'] < 1):
                    gray = cv2.resize(
                        gray,
                        (
                            int(gray.shape[1]*video_dict['dsmpl']),
                            int(gray.shape[0]*video_dict['dsmpl'])
                        ),
                        cv2.INTER_NEAREST)
                gray = cropframe(
                    gray, 
                    video_dict.get('crop')
                )
                collection[idx,:,:]=gray
                grabbed = True
            elif ret == False:
                framenum = np.random.randint(video_dict['start'],cap_max,1)[0]
                pass
    cap.release() 

    reference = np.median(collection,axis=0)
    image = hv.Image((np.arange(reference.shape[1]),
                      np.arange(reference.shape[0]), 
                      reference)).opts(width=int(reference.shape[1]*video_dict['stretch']['width']),
                                       height=int(reference.shape[0]*video_dict['stretch']['height']),
                                       invert_yaxis=True,
                                       cmap='gray',
                                       colorbar=True,
                                       toolbar='below',
                                       title="Reference Frame") 
    return reference, image    





########################################################################################

def Locate(cap,tracking_params,video_dict,prior=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Return location of animal in frame, in x/y coordinates. 
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.VideoCapture]
            OpenCV VideoCapture class instance for video.
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        prior:: [list]
            If window is being used, list of length 2 is passed, where first index is 
            prior y position, and second index is prior x position.
    
    -------------------------------------------------------------------------------------
    Returns:
        ret:: [bool]
            Specifies whether frame is returned in response to cv2.VideoCapture.read.
        
        dif:: [numpy.array]
            Pixel-wise difference from prior frame, after thresholding and
            applying window weight.
        
        com:: [tuple]
            Indices of center of mass as tuple in the form: (y,x).
        
        frame:: [numpy.array]
            Original video frame after cropping.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #attempt to load frame
    ret, frame = cap.read() 
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

    if ret == True:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(
            frame,
            video_dict.get('crop')
        )
        
        # ===== NEW: Analysis ROI Support (Rectangle or Polygon) ===== #
        use_analysis_roi = 'analysis_roi' in video_dict and video_dict['analysis_roi'] is not None
        
        # Check if polygon or rectangle
        is_polygon_roi = (use_analysis_roi and 
                         isinstance(video_dict['analysis_roi'], dict) and
                         video_dict['analysis_roi'].get('type') == 'polygon')
        
        roi_x_offset, roi_y_offset = 0, 0
        if use_analysis_roi and not is_polygon_roi:
            # Rectangle ROI (fast, direct slicing)
            x1, y1, x2, y2 = video_dict['analysis_roi']
            roi_x_offset, roi_y_offset = x1, y1
            frame_roi = frame[y1:y2, x1:x2]
            ref_roi = video_dict['reference'][y1:y2, x1:x2]
        else:
            # Full frame (for polygon or no ROI)
            frame_roi = frame
            ref_roi = video_dict['reference']
        
        #find difference from reference
        if tracking_params['method'] == 'abs':
            dif = np.absolute(frame_roi-ref_roi)
        elif tracking_params['method'] == 'light':
            dif = frame_roi-ref_roi
        elif tracking_params['method'] == 'dark':
            dif = ref_roi-frame_roi
        dif = dif.astype('int16')
        
        # Apply polygon ROI mask if used
        if is_polygon_roi:
            # Generate mask if not already exists (for direct parameter setting)
            if video_dict['analysis_roi'].get('mask') is None:
                vertices = video_dict['analysis_roi']['vertices']
                mask = np.zeros(frame_roi.shape, dtype=np.uint8)
                for poly in vertices:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                video_dict['analysis_roi']['mask'] = mask > 0  # Boolean mask
            
            polygon_mask = video_dict['analysis_roi']['mask']
            dif[~polygon_mask] = 0
        
        # Legacy mask support (only if not using analysis_roi)
        if not use_analysis_roi and 'mask' in video_dict.keys():
            if video_dict['mask']['mask'] is not None:
                    dif[video_dict['mask']['mask']] = 0
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif = dif + (dif.min() * -1) #scale so lowest value is 0
            dif_weights = np.ones(dif.shape)*weight
            # If using rectangle analysis ROI, dif is in ROI-local coordinates.
            # Convert prior (global cropped-frame coordinates) to local ROI coordinates.
            local_ymin = ymin - roi_y_offset
            local_ymax = ymax - roi_y_offset
            local_xmin = xmin - roi_x_offset
            local_xmax = xmax - roi_x_offset

            y_start = max(0, local_ymin)
            y_end = min(dif.shape[0], local_ymax)
            x_start = max(0, local_xmin)
            x_end = min(dif.shape[1], local_xmax)

            if y_end > y_start and x_end > x_start:
                dif_weights[slice(y_start, y_end), slice(x_start, x_end)] = 1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        # For polygon ROI: only calculate percentile on non-zero pixels (ROI area)
        if is_polygon_roi:
            # Only use non-zero values for percentile calculation
            nonzero_dif = dif[dif > 0]
            if len(nonzero_dif) > 0:
                threshold = np.percentile(nonzero_dif, tracking_params['loc_thresh'])
                dif[dif < threshold] = 0
            else:
                # No signal at all
                dif[:] = 0
        else:
            # Original behavior for non-polygon ROI
            dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        
        #remove influence of wire
        if tracking_params['rmv_wire'] == True:
            ksize = tracking_params['wire_krn']
            kernel = np.ones((ksize,ksize),np.uint8)
            dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
            krn_violation =  dif_wirermv.sum()==0
            dif = dif if krn_violation else dif_wirermv
            if krn_violation:
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
                    x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict['start'])))
            
        # Calculate center of mass
        if is_polygon_roi and dif.sum() > 0:
            # For polygon ROI, compute only within valid region to avoid edge effects
            y_coords, x_coords = np.mgrid[0:dif.shape[0], 0:dif.shape[1]]
            valid_mask = video_dict['analysis_roi']['mask'] & (dif > 0)
            
            if valid_mask.sum() > 0:
                total_weight = dif[valid_mask].sum()
                com_y = (dif[valid_mask] * y_coords[valid_mask]).sum() / total_weight
                com_x = (dif[valid_mask] * x_coords[valid_mask]).sum() / total_weight
                com = (com_y, com_x)
            else:
                com = (np.nan, np.nan)
        else:
            # Standard center of mass calculation
            com_local = ndimage.measurements.center_of_mass(dif)
            
            # ===== Convert local coordinates to global if using rectangle ROI ===== #
            if use_analysis_roi and not is_polygon_roi and not np.isnan(com_local[0]):
                x1, y1, x2, y2 = video_dict['analysis_roi']
                com = (com_local[0] + y1, com_local[1] + x1)
                
                # Create full-size difference image for visualization
                dif_full = np.zeros(frame.shape, dtype='int16')
                dif_full[y1:y2, x1:x2] = dif
                dif = dif_full
            else:
                com = com_local
        
        return ret, dif, com, frame
    
    else:
        return ret, None, None, frame

    
    
    
    
########################################################################################        

def TrackLocation(video_dict,tracking_params):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame in video define location of animal, in x/y coordinates, and distance
    travelled from previous frame.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array] 
                              
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]    
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
          
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) #set starting frame
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max  
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max - video_dict['start'])
    Y = np.zeros(cap_max - video_dict['start'])
    D = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    time.sleep(.2) #allow printing
    for f in tqdm(range(len(D))):
        
        if f>0: 
            # Check if prior position is valid (not NaN)
            if not np.isnan(Y[f-1]) and not np.isnan(X[f-1]):
                yprior = np.around(Y[f-1]).astype(int)
                xprior = np.around(X[f-1]).astype(int)
                ret,dif,com,frame = Locate(cap,tracking_params,video_dict,prior=[yprior,xprior])
            else:
                # Prior position is NaN, don't use prior
                ret,dif,com,frame = Locate(cap,tracking_params,video_dict)
        else:
            ret,dif,com,frame = Locate(cap,tracking_params,video_dict)
                                                
        if ret == True:          
            Y[f] = com[0]
            X[f] = com[1]
            if f>0:
                D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
        else:
            #if no frame is detected
            f = f-1
            X = X[:f] #Amend length of X vector
            Y = Y[:f] #Amend length of Y vector
            D = D[:f] #Amend length of D vector
            break   
            
    #release video
    cap.release()
    time.sleep(.2) #allow printing
    print('total frames processed: {f}\n'.format(f=len(D)))
    
    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'Location_Thresh': np.ones(len(D))*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(len(D))*tracking_params['window_weight'],
     'Window_Size': np.ones(len(D))*tracking_params['window_size'],
     'Start_Frame': np.ones(len(D))*video_dict['start'],
     'Frame': np.arange(len(D)),
     'X': X,
     'Y': Y,
     'Distance_px': D
    })
    
    #add region of interest info
    df = ROI_Location(video_dict, df) 
    if video_dict['region_names'] is not None:
        print('Defining transitions...')
        df['ROI_location'] = ROI_linearize(df[video_dict['region_names']])
        df['ROI_transition'] = ROI_transitions(df['ROI_location'])
    
    #update scale, if known
    df = ScaleDistance(video_dict, df=df, column='Distance_px')
       
    return df





########################################################################################

def LocationThresh_View(video_dict,tracking_params,examples=4):
    """ 
    -------------------------------------------------------------------------------------
    
    Display example tracking with selected parameters for a random subset of frames. 
    NOTE that because individual frames are analyzed independently, weighting 
    based upon prior location is not implemented.
    
    -------------------------------------------------------------------------------------
    Args:
  
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
            
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int] 
                           
        examples:: [uint]
            The number of frames for location tracking to be tested on.
        
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [holoviews.Layout]
            Returns Holoviews Layout with original images on left and heat plots with 
            animal's estimated position marked on right.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence display and not
          calculation
    
    """
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #examine random frames
    images = []
    for example in range (examples):
        
        #analyze frame
        ret = False
        while ret is False:     
            frm=np.random.randint(video_dict['start'],cap_max) #select random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,frm) #sets frame to be next to be grabbed
            ret,dif,com,frame = Locate(cap, tracking_params, video_dict) 

        #plot original frame
        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        image_orig.opts(
            width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
            height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='gray',toolbar='below',
            title="Frame: " + str(frm))
        orig_overlay = image_orig * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        #plot heatmap
        dif = dif*(255//dif.max())
        image_heat = hv.Image((
            np.arange(dif.shape[1]), 
            np.arange(dif.shape[0]), 
            dif))
        image_heat.opts(
            width=int(dif.shape[1]*video_dict['stretch']['width']),
            height=int(dif.shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='jet',toolbar='below',
            title="Frame: " + str(frm - video_dict['start']))
        heat_overlay = image_heat * hv.Points(([com[1]],[com[0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        images.extend([orig_overlay,heat_overlay])
    
    cap.release()
    layout = hv.Layout(images)
    return layout





########################################################################################    
    
def ROI_plot(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                                      
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define regions of interest.
        
        poly_stream:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            selection tool. `poly_stream.data` contains x and y coordinates of roi 
            vertices.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #get number of objects to be drawn
    nobjects = len(video_dict['region_names']) if video_dict['region_names'] else 0 

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]),
        np.arange(video_dict['reference'].shape[0]),
        video_dict['reference']))
    image.opts(
        width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
        height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,cmap='gray', colorbar=True,toolbar='below',
        title="No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(video_dict['region_names']))

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = video_dict['region_names'][:len(xs)]
        return hv.Labels((xs, ys, rois))
    
    if nobjects > 0:
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return (image * poly * dmap), poly_stream
    else:
        return (image),None
    

def ROI_plot_cv2(video_dict):
    """
    -------------------------------------------------------------------------------------
    
    Alternative ROI plotting function using OpenCV (simpler interaction).
    Uses single-click to add vertices, Enter to finish polygon, 'r' to reset.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Same as ROI_plot function
    
    Returns:
        roi_stream_data:: [dict]
            Dictionary with 'xs' and 'ys' keys containing polygon coordinates
    """
    if video_dict['region_names'] is None:
        print("[INFO] No region_names specified, skipping ROI definition")
        return None
    
    nobjects = len(video_dict['region_names'])
    reference = video_dict['reference']
    
    # Storage for all ROI polygons
    all_rois = []
    current_roi_points = []
    current_roi_idx = 0
    
    # Create a copy of reference image for drawing (convert to uint8 if needed)
    display_img = reference.copy()
    if display_img.dtype != np.uint8:
        display_img = ((display_img - display_img.min()) / (display_img.max() - display_img.min()) * 255).astype(np.uint8)
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_roi_points, current_roi_idx, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current ROI
            current_roi_points.append([x, y])
            
            # Draw point
            cv2.circle(display_img, (x, y), 3, (0, 255, 0), -1)
            
            # Draw line if more than one point
            if len(current_roi_points) > 1:
                pt1 = tuple(current_roi_points[-2])
                pt2 = tuple(current_roi_points[-1])
                cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
            
            # Update display
            temp_img = display_img.copy()
            if current_roi_idx < nobjects:
                cv2.putText(temp_img, f"ROI {current_roi_idx + 1}/{nobjects}: {video_dict['region_names'][current_roi_idx]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(temp_img, f"Points: {len(current_roi_points)} | Press ENTER to finish, 'r' to reset", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Draw ROI - Click to add points, ENTER to finish, ESC to skip', temp_img)
    
    # Create window and set callback
    window_name = 'Draw ROI - Click to add points, ENTER to finish, ESC to skip'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize window for better visibility
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        cv2.resizeWindow(window_name, new_w, new_h)
    
    print("\n" + "="*70)
    print("OpenCV ROI Drawing Tool")
    print("="*70)
    print("Instructions:")
    print("  - LEFT CLICK: Add a vertex to current ROI")
    print("  - ENTER: Finish current ROI and move to next")
    print("  - 'r' or 'R': Reset current ROI (clear points)")
    print("  - ESC: Skip remaining ROIs and finish")
    print("="*70)
    
    while current_roi_idx < nobjects:
        # Reset for new ROI
        current_roi_points = []
        display_img = reference.copy()
        if display_img.dtype != np.uint8:
            display_img = ((display_img - display_img.min()) / (display_img.max() - display_img.min()) * 255).astype(np.uint8)
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        # Draw all completed ROIs
        for i, roi in enumerate(all_rois):
            if len(roi) > 2:
                pts = np.array(roi, np.int32)
                # Draw filled polygon with transparency
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
                # Draw border
                cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)
                cv2.putText(display_img, video_dict['region_names'][i], 
                           tuple(roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show current status
        temp_img = display_img.copy()
        cv2.putText(temp_img, f"ROI {current_roi_idx + 1}/{nobjects}: {video_dict['region_names'][current_roi_idx]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(temp_img, "Click to add points, ENTER to finish, 'r' to reset, ESC to skip", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(window_name, temp_img)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter key
                if len(current_roi_points) >= 3:  # Need at least 3 points for polygon
                    all_rois.append(current_roi_points.copy())
                    print(f"[OK] ROI {current_roi_idx + 1} '{video_dict['region_names'][current_roi_idx]}' completed with {len(current_roi_points)} points")
                    current_roi_idx += 1
                    break
                else:
                    print(f"[WARNING] Need at least 3 points to create a polygon. Current points: {len(current_roi_points)}")
            
            elif key == ord('r') or key == ord('R'):  # Reset
                current_roi_points = []
                display_img = reference.copy()
                if display_img.dtype != np.uint8:
                    display_img = ((display_img - display_img.min()) / (display_img.max() - display_img.min()) * 255).astype(np.uint8)
                if len(display_img.shape) == 2:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                # Redraw completed ROIs
                for i, roi in enumerate(all_rois):
                    if len(roi) > 2:
                        pts = np.array(roi, np.int32)
                        overlay = display_img.copy()
                        cv2.fillPoly(overlay, [pts], (255, 0, 0))
                        cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
                        cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)
                temp_img = display_img.copy()
                cv2.putText(temp_img, f"ROI {current_roi_idx + 1}/{nobjects}: {video_dict['region_names'][current_roi_idx]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(temp_img, "Reset! Click to add points, ENTER to finish", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow(window_name, temp_img)
                print(f"[RESET] Cleared points for ROI {current_roi_idx + 1}")
            
            elif key == 27:  # ESC key
                print(f"[SKIP] Skipping remaining ROIs")
                cv2.destroyAllWindows()
                if len(all_rois) == 0:
                    return None
                # Fill remaining with empty lists
                while len(all_rois) < nobjects:
                    all_rois.append([])
                break
    
    cv2.destroyAllWindows()
    
    # Convert to format compatible with PolyDraw stream
    xs = []
    ys = []
    for roi in all_rois:
        if len(roi) > 0:
            xs.append([p[0] for p in roi])
            ys.append([p[1] for p in roi])
        else:
            xs.append([])
            ys.append([])
    
    # Use module-level MockStream for pickle compatibility
    roi_stream_data = {'xs': xs, 'ys': ys}
    roi_stream = MockStream(roi_stream_data)
    
    print("\n[COMPLETE] All ROIs defined!")
    print(f"  Total ROIs: {len([r for r in all_rois if len(r) > 0])}/{nobjects}")
    
    return roi_stream


    
    
########################################################################################    

def ROI_Location(video_dict, location):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame, determine which regions of interest the animal is in.  For each
    region of interest, boolean array is added to `location` dataframe passed, with 
    column name being the region name.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Must contain column names 'X' and 'Y'.

    -------------------------------------------------------------------------------------
    Returns:
        location:: [pandas.dataframe]
            For each region of interest, boolean array is added to `location` dataframe 
            passed, with column name being the region name. Additionally, under column
            `ROI_coordinates`, coordinates of vertices of each region of interest are
            printed. This takes the form of a dictionary of x and y coordinates, e.g.:
                'xs' : [[region 1 x coords], [region 2 x coords]],
                'ys' : [[region 1 y coords], [region 2 y coords]]
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    if video_dict['region_names'] == None:
        return location

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(video_dict['roi_stream'].data['xs'])):
        x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates
        y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(video_dict['reference'].shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
        ROI_masks[video_dict['region_names'][poly]] = mask==255 #save to ROI masks as boolean 

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location['Frame']:
        y,x = location['Y'][f], location['X'][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]
    
    #Add data to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]
    
    #Add ROI coordinates
    location['ROI_coordinates']=str(video_dict['roi_stream'].data)
    
    return location





########################################################################################        

def ROI_linearize(rois, null_name = 'non_roi'):
    
    """ 
    -------------------------------------------------------------------------------------
    
    Creates array defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Args:
        rois:: [pd.DataFrame]
            Pandas dataframe where each column corresponds to an ROI, with boolean values
            defining if animal is in said roi.
        null_name:: [string]
            Name used when animals is not in any defined roi.
    
    -------------------------------------------------------------------------------------
    Returns:
        rois['ROI_location']:: [pd.Series]
            pd.Series defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    region_names = rois.columns.values
    rois['ROI_location'] = null_name
    for region in region_names:
        rois['ROI_location'][rois[region]] = rois['ROI_location'][rois[region]].apply(
            lambda x: '_'.join([x, region]) if x!=null_name else region
        )
    return rois['ROI_location']






########################################################################################        

def ROI_transitions(regions, include_first=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates boolean array defining where transitions between each ROI occur.
    
    -------------------------------------------------------------------------------------
    Args:
        regions:: [Pandas Series]
            Pandas Series defining ROI as string for each frame
        include_first:: [string]
            Whether to count first frame as transition
    
    -------------------------------------------------------------------------------------
    Returns:
        transitions:: [Boolean array]
            pd.Series defining where transitions between ROIs occur.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    regions_offset = np.append(regions[0], regions[0:-1])
    transitions = regions!=regions_offset
    if include_first:
        transitions[0] = True
    return transitions





########################################################################################        
    
def Summarize_Location(location, video_dict, bin_dict=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates summary of distance travelled and proportional time spent in each region
    of interest according to user defined time bins.  If bins are not provided 
    (`bin_dict=None`), average of entire video segment will be provided.
    
    -------------------------------------------------------------------------------------
    Args:
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame.
      
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                              
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example: bin_dict = {1:(0,100), 2:(100,200)}                             

    
    -------------------------------------------------------------------------------------
    Returns:
        bins:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #define bins
    avg_dict = {'all': (location['Frame'].min(), location['Frame'].max())}   
    bin_dict = bin_dict if bin_dict is not None else avg_dict
    
    #get summary info
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))    
    bins['Distance_px'] = bins['range(f)'].apply(
        lambda r: location[location['Frame'].between(*r)]['Distance_px'].sum())
    if video_dict['region_names'] is not None:
        bins_reg = bins['range(f)'].apply(
            lambda r: location[location['Frame'].between(*r)][video_dict['region_names']].mean())
        bins = bins.join(bins_reg)
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y'] + video_dict['region_names']
    else:
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y']
    bins = pd.merge(
        location.drop(drp_cols, axis='columns'),
        bins,
        left_index=True,
        right_index=True)
    
    #scale distance
    bins = ScaleDistance(video_dict,df=bins,column='Distance_px') 
    
    return bins





######################################################################################## 

def Batch_LoadFiles(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Populates list of files in directory (`dpath`) that are of the specified file type
    (`ftype`).  List is held in `video_dict['FileNames']`.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #Get list of video files of designated type
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
        return video_dict
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

        
        
        
        
######################################################################################## 

def Batch_Process(video_dict,tracking_params,bin_dict,accept_p_frames=False):   
    """ 
    -------------------------------------------------------------------------------------
    
    Run LocationTracking on folder of videos of specified filetype. 
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
         accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
            
        layout:: [hv.Layout]
            Holoviews layout wherein for each session the reference frame is returned
            with the regions of interest highlightted and the animals location across
            the session overlaid atop the reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    images = []
    for file in video_dict['FileNames']:
        
        print ('Processing File: {f}'.format(f=file))  
        video_dict['file'] = file 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
        
        #Print video information. Note that max frame is updated later if fewer frames detected
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        print('total frames: {frames}'.format(frames=cap_max))
        print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
        print('dimensions (h x w): {h},{w}'.format(
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        
        #check for video p-frames
        if accept_p_frames is False:
            check_p_frames(cap)
        
        video_dict['reference'], image = Reference(video_dict,num_frames=50) 
        location = TrackLocation(video_dict,tracking_params)
        location.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv', index=False)
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)
               
        try: 
            summary_all = pd.concat([summary_all,file_summary],sort=False)
        except NameError: 
            summary_all = file_summary
        
        trace = showtrace(video_dict,location)
        heatmap = Heatmap(video_dict, location, sigma=None)
        images = images + [(trace.opts(title=file)), (heatmap.opts(title=file))]

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout, index=False)
    
    layout = hv.Layout(images)
    return summary_all, layout


def _process_single_video(args):
    """
    Helper function to process a single video (for multiprocessing)
    
    Args:
        args: tuple containing (file, video_dict, tracking_params, bin_dict, accept_p_frames)
    
    Returns:
        dict with keys: 'file', 'location', 'summary', 'success', 'error'
    """
    file, video_dict_template, tracking_params, bin_dict, accept_p_frames = args
    
    # Create a copy of video_dict for this process
    video_dict = video_dict_template.copy()
    video_dict['file'] = file
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
    
    try:
        print(f'[Process {os.getpid()}] Processing: {file}')
        
        # Load video and get info
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        print(f'[{file}] frames: {cap_max}, fps: {fps}, dimensions: {h}x{w}')
        
        # Check for p-frames
        if accept_p_frames is False:
            check_p_frames(cap)
        
        # Generate reference frame
        video_dict['reference'], _ = Reference(video_dict, num_frames=50)
        
        # Track location
        location = TrackLocation(video_dict, tracking_params)
        
        # Save location data
        csv_path = os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv'
        location.to_csv(csv_path, index=False)
        
        # Generate summary
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)
        
        print(f'[{file}] ✓ Completed successfully')
        
        return {
            'file': file,
            'location': location,
            'summary': file_summary,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f'[{file}] ✗ Error: {str(e)}')
        return {
            'file': file,
            'location': None,
            'summary': None,
            'success': False,
            'error': str(e)
        }


def _prepare_video_dict_for_pickle(video_dict):
    """
    Prepare video_dict for multiprocessing by converting unpicklable objects.
    
    This handles cases where Mock objects were created with old local classes.
    """
    video_dict_clean = video_dict.copy()
    
    # Fix crop object if it exists and uses old local class
    if 'crop' in video_dict_clean and video_dict_clean['crop'] is not None:
        crop = video_dict_clean['crop']
        if hasattr(crop, 'data'):
            # Recreate with module-level MockCrop
            video_dict_clean['crop'] = MockCrop(crop.data)
    
    # Fix roi_stream if it exists
    if 'roi_stream' in video_dict_clean and video_dict_clean['roi_stream'] is not None:
        roi_stream = video_dict_clean['roi_stream']
        if hasattr(roi_stream, 'data'):
            # Recreate with module-level MockStream
            video_dict_clean['roi_stream'] = MockStream(roi_stream.data)
    
    # Fix mask stream if it exists
    if 'mask' in video_dict_clean and isinstance(video_dict_clean['mask'], dict):
        if 'stream' in video_dict_clean['mask'] and video_dict_clean['mask']['stream'] is not None:
            mask_stream = video_dict_clean['mask']['stream']
            if hasattr(mask_stream, 'data'):
                # Recreate with module-level MockStream
                video_dict_clean['mask']['stream'] = MockStream(mask_stream.data)
    
    return video_dict_clean


def Batch_Process_Parallel(video_dict, tracking_params, bin_dict, 
                           n_processes=None, accept_p_frames=False):
    """
    -------------------------------------------------------------------------------------
    
    Run LocationTracking on folder of videos using PARALLEL PROCESSING.
    This is much faster than Batch_Process for multiple videos.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Same as Batch_Process
        
        tracking_params:: [dict]
            Same as Batch_Process
        
        bin_dict:: [dict or None]
            Same as Batch_Process
        
        n_processes:: [int or None]
            Number of parallel processes to use. If None, uses CPU count - 1.
            Recommended: leave as None for automatic optimization.
        
        accept_p_frames:: [bool]
            Same as Batch_Process
    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Combined summary of all videos
            
        layout:: [hv.Layout]
            Holoviews layout with traces and heatmaps
    
    -------------------------------------------------------------------------------------
    Notes:
        - Much faster than Batch_Process for multiple videos
        - Uses multiprocessing to process videos in parallel
        - Memory usage will be higher (multiple videos loaded at once)
        - Progress bar shows real-time updates in Jupyter
    
    """
    import multiprocessing as mp
    from functools import partial
    import time
    
    # Try to import tqdm for progress bar
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("💡 Tip: Install tqdm for better progress display: pip install tqdm")
    
    # Clean video_dict for pickle compatibility
    video_dict = _prepare_video_dict_for_pickle(video_dict)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    n_videos = len(video_dict['FileNames'])
    n_processes = min(n_processes, n_videos)  # Don't use more processes than videos
    
    print('='*70)
    print('🚀 PARALLEL BATCH PROCESSING')
    print('='*70)
    print(f'📊 Total videos: {n_videos}')
    print(f'⚡ Parallel processes: {n_processes}')
    print(f'🎯 Expected speedup: ~{min(n_processes, n_videos)}x')
    print('='*70)
    print()
    
    # Prepare arguments for each video
    args_list = [
        (file, video_dict, tracking_params, bin_dict, accept_p_frames)
        for file in video_dict['FileNames']
    ]
    
    start_time = time.time()
    
    # Process videos in parallel with progress bar
    with mp.Pool(processes=n_processes) as pool:
        if use_tqdm:
            # Use imap_unordered for immediate progress updates
            # ncols removed - auto-adjusts to terminal/notebook width for better display
            results = list(tqdm(
                pool.imap_unordered(_process_single_video, args_list),
                total=n_videos,
                desc="Processing videos",
                unit="video"
            ))
        else:
            # Fallback: use callback for progress updates
            results = []
            completed = [0]
            
            def update_progress(result):
                completed[0] += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed[0]
                remaining = (n_videos - completed[0]) * avg_time
                
                status = '✓' if result['success'] else '✗'
                print(f"[{completed[0]}/{n_videos}] {status} {result['file']} | "
                      f"Elapsed: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s")
                
                if not result['success']:
                    print(f"    ❌ Error: {result['error']}")
            
            for args in args_list:
                result = pool.apply_async(_process_single_video, (args,), callback=update_progress)
                results.append(result)
            
            # Wait for all to complete
            results = [r.get() for r in results]
    
    print()
    print('='*70)
    print('PROCESSING COMPLETE - Generating visualizations...')
    print('='*70)
    
    # Collect results
    summary_list = []
    images = []
    success_count = 0
    error_count = 0
    
    for result in results:
        if result['success']:
            success_count += 1
            
            # Add summary
            summary_list.append(result['summary'])
            
            # Generate trace and heatmap for this video
            # Need to reload video_dict for this file
            video_dict_single = video_dict.copy()
            video_dict_single['file'] = result['file']
            video_dict_single['fpath'] = os.path.join(
                os.path.normpath(video_dict['dpath']), result['file'])
            video_dict_single['reference'], _ = Reference(video_dict_single, num_frames=50)
            
            trace = showtrace(video_dict_single, result['location'])
            heatmap = Heatmap(video_dict_single, result['location'], sigma=None)
            images.extend([
                trace.opts(title=result['file']), 
                heatmap.opts(title=result['file'])
            ])
        else:
            error_count += 1
            print(f"[ERROR] Failed to process: {result['file']}")
            print(f"  Error: {result['error']}")
    
    # Combine all summaries
    if summary_list:
        summary_all = pd.concat(summary_list, sort=False, ignore_index=True)
    else:
        summary_all = pd.DataFrame()
    
    # Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout, index=False)
    
    # Create layout
    layout = hv.Layout(images)
    
    print()
    print('='*70)
    print('BATCH PROCESSING SUMMARY')
    print('='*70)
    print(f'✓ Successful: {success_count}/{n_videos}')
    print(f'✗ Failed: {error_count}/{n_videos}')
    print(f'Summary saved to: {sum_pathout}')
    print('='*70)
    
    return summary_all, layout




########################################################################################        

def PlayVideo(video_dict,display_dict,location):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location. Video is played
    in notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'resize' : Default is None, in which original size is retained.
                           Alternatively, set to tuple as follows: (width,height).
                           Because this is in pixel units, must be integer values.
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
            
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """


    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, video_dict['crop'])
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, video_dict['crop'])
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            display_image(frame,display_dict['fps'],display_dict['resize'])
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open
    print('Done playing segment')
    if display_dict['save_video']==True:
        writer.release()

def display_image(frame,fps,resize):
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer,format="JPEG")    
    display(Image(data=buffer.getvalue()))
    time.sleep(1/fps)
    clear_output(wait=True)

    
    

    
########################################################################################

def PlayVideo_ext(video_dict,display_dict,location,crop=None):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
  
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, crop)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 
    rate = int(1000/display_dict['fps']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, crop)
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow("preview",frame)
            cv2.waitKey(rate)
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open        
    cv2.destroyAllWindows()
    _=cv2.waitKey(1) 
    if display_dict['save_video']==True:
        writer.release()

    
    
    
    
########################################################################################

def showtrace(video_dict, location, color="red",alpha=.8,size=3):
    """ 
    -------------------------------------------------------------------------------------
    
    Create image where animal location across session is displayed atop reference frame

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 

            
        color:: [str]
            Color of trace.  See Holoviews documentation for color options
                               
        alpha:: [float]
            Alpha of trace.  See Holoviews documentation for details
        
        size:: [float]
            Size of trace.  See Holoviews documentation for details.     
    
    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Location of animal superimposed upon reference. If poly_stream is passed
            than regions of interest will also be outlined.
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    video_dict['roi_stream'] = video_dict['roi_stream'] if 'roi_stream' in video_dict else None
    if video_dict['roi_stream'] != None:
        lst = []
        for poly in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates
            y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates
            lst.append( [ (x[vert],y[vert]) for vert in range(len(x)) ] )
        poly = hv.Polygons(lst).opts(fill_alpha=0.1,line_dash='dashed')
        
    image = hv.Image((np.arange(video_dict['reference'].shape[1]),
                      np.arange(video_dict['reference'].shape[0]),
                      video_dict['reference'])
                    ).opts(width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
                           height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
                           invert_yaxis=True,cmap='gray',toolbar='below',
                           title="Motion Trace")
    
    points = hv.Scatter(np.array([location['X'],location['Y']]).T).opts(color='red',alpha=alpha,size=size)
    
    return (image*poly*points) if video_dict['roi_stream']!=None else (image*points)





########################################################################################    

def Heatmap (video_dict, location, sigma=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Create heatmap of relative time in each location. Max value is set to maxiumum
    in any one location.

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
                
        sigma:: [numeric]
            Optional number specifying sigma of guassian filter
  
    
    -------------------------------------------------------------------------------------
    Returns:
        map_i:: [holoviews.Image]
            Heatmap image
    
    -------------------------------------------------------------------------------------
    Notes:
        stretch only affects display

    """    
    heatmap = np.zeros(video_dict['reference'].shape)
    for frame in range(len(location)):
        Y,X = int(location.Y[frame]), int(location.X[frame])
        heatmap[Y,X]+=1
    
    sigma = np.mean(heatmap.shape)*.05 if sigma == None else sigma
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap = (heatmap / heatmap.max())*255
    
    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts(width=int(heatmap.shape[1]*video_dict['stretch']['width']),
           height=int(heatmap.shape[0]*video_dict['stretch']['height']),
           invert_yaxis=True, cmap='jet', alpha=1,
           colorbar=False, toolbar='below', title="Heatmap")
    
    return map_i





########################################################################################    

def DistanceTool(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for measuring length between two points, in pixel units, in 
    order to ease process of converting pixel distance measurements to some other scale.
    Use point drawing tool to calculate distance beteen any two popints.
    
    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
    
    -------------------------------------------------------------------------------------
    Returns:
        image * points * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define 2 points, the distance 
            between which will be measured and displayed.
        
        distance:: [dict]
            Dictionary with the following keys:
                'd' : Euclidean distance between two reference points, in pixel units, 
                      rounded to thousandth. Returns None if no less than 2 points have 
                      been selected.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]), 
        np.arange(video_dict['reference'].shape[0]), 
        video_dict['reference']))
    image.opts(width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
               height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Select Points")

    #Create Point instance on which to draw and connect via stream to pointDraw drawing tool 
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def markers(data, distance):
        try:
            x_ls, y_ls = data['x'], data['y']
        except TypeError:
            x_ls, y_ls = [], []
        
        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        if len(x_ls) > 1:
            x_dist = (x_ls[0] - x_ls[1])
            y_dist = (y_ls[0] - y_ls[1])
            distance['px_distance'] = np.around( (x_dist**2 + y_dist**2)**(1/2), 3)
            text = "{dist} px".format(dist=distance['px_distance'])
        return hv.Labels((x_ctr, y_ctr, text if len(x_ls) > 1 else "")).opts(
            text_color='blue',text_font_size='14pt')
    
    distance = dict(px_distance=None)
    markers_ptl = fct.partial(markers, distance=distance)
    dmap = hv.DynamicMap(markers_ptl, streams=[pointDraw_stream])
    return (image * points * dmap), distance


def DistanceTool_cv2(video_dict):
    """
    -------------------------------------------------------------------------------------
    
    OpenCV version of DistanceTool. Click to place two points and measure distance.
    
    -------------------------------------------------------------------------------------
    Instructions:
        - Click to place first point
        - Click to place second point
        - Distance is displayed automatically
        - Press 'r' to reset points
        - Press ENTER to confirm
        - Press ESC to skip
    
    -------------------------------------------------------------------------------------
    Returns:
        distance:: [dict] with 'px_distance' key containing pixel distance
    """
    
    reference = video_dict['reference'].copy()
    # Convert to uint8 if needed
    if reference.dtype != np.uint8:
        reference = ((reference - reference.min()) / (reference.max() - reference.min()) * 255).astype(np.uint8)
    points = []
    
    display_img = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR) if len(reference.shape) == 2 else reference.copy()
    original_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                # Redraw
                display_img = original_img.copy()
                for i, pt in enumerate(points):
                    cv2.circle(display_img, pt, 6, (0, 0, 255), -1)
                    cv2.putText(display_img, f"P{i+1}", (pt[0]+10, pt[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if len(points) == 2:
                    cv2.line(display_img, points[0], points[1], (0, 255, 0), 2)
                    dist = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
                    mid_x = (points[0][0] + points[1][0]) // 2
                    mid_y = (points[0][1] + points[1][1]) // 2
                    cv2.putText(display_img, f"{dist:.2f} px", (mid_x+10, mid_y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    window_name = 'Distance Tool - Click 2 points, ENTER to confirm, R to reset'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    print("\n" + "="*70)
    print("OpenCV Distance Tool")
    print("="*70)
    print("Instructions:")
    print("  - CLICK: Place points (need 2)")
    print("  - 'r': Reset points")
    print("  - ENTER: Confirm distance")
    print("  - ESC: Skip")
    print("="*70)
    
    distance = dict(px_distance=None)
    
    while True:
        temp_img = display_img.copy()
        cv2.putText(temp_img, f"Click to place points ({len(points)}/2)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(points) == 2:
            dist = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
            cv2.putText(temp_img, f"Distance: {dist:.2f} pixels - Press ENTER to confirm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if len(points) == 2:
                dist = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
                distance['px_distance'] = np.around(dist, 3)
                print(f"[OK] Distance: {distance['px_distance']} pixels")
                break
            else:
                print("[WARNING] Need 2 points to measure distance")
                
        elif key == ord('r') or key == ord('R'):
            points = []
            display_img = original_img.copy()
            print("[RESET] Cleared points")
            
        elif key == 27:  # ESC
            print("[SKIP] Distance measurement skipped")
            break
    
    cv2.destroyAllWindows()
    return distance


########################################################################################

def setScale(distance, scale, scale_dict):

    """ 
    -------------------------------------------------------------------------------------
    
    Updates dictionary with scale information, given the true distance between points 
    (e.g. 100), and the scale unit (e.g. 'cm')
    
    -------------------------------------------------------------------------------------
    Args:
    
        distance :: [numeric]
            The real-world distance between the selected points
        
        scale :: [string]
            The scale used for defining the real world distance.  Can be any string
            (e.g. 'cm', 'in', 'inch', 'stone')

        scale_dict :: [dict]
            Dictionary with the following keys:
                'px_distance' : distance between reference points, in pixels [numeric]
                'true_distance' : distance between reference points, in desired scale 
                                   (e.g. cm) [numeric]
                'true_scale' : string containing name of scale (e.g. 'cm') [str]
                'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]

    -------------------------------------------------------------------------------------
    Returns:
        scale_dict :: [dict]
                Dictionary with the following keys:
                    'px_distance' : distance between reference points, in pixels [numeric]
                    'true_distance' : distance between reference points, in desired scale 
                                       (e.g. cm) [numeric]
                    'true_scale' : string containing name of scale (e.g. 'cm') [str]
                    'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
    -------------------------------------------------------------------------------------
    Notes:

    """

    scale_dict['true_distance'] = distance
    scale_dict['true_scale'] = scale
    return scale_dict
    


########################################################################################    

def ScaleDistance(video_dict, df=None, column=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Adds column to dataframe by multiplying existing column by scaling factor to change
    scale. Used in order to convert distance from pixel scale to desired real world 
    distance scale.
    
    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        df:: [pandas.dataframe]
            Pandas dataframe with column to be scaled.
        
        column:: [str]
            Name of column in df to be scaled
        
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with column of scaled distance values.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    if 'scale' not in video_dict.keys():
        return df

    if video_dict['scale']['px_distance']!= None:
        video_dict['scale']['factor'] = video_dict['scale']['true_distance']/video_dict['scale']['px_distance']
        new_column = "_".join(['Distance', video_dict['scale']['true_scale']])
        df[new_column] = df[column]*video_dict['scale']['factor']
        order = [col for col in df if col not in [column,new_column]]
        order = order + [column,new_column]
        df = df[order]
    else:
        print('Distance between reference points undefined. Cannot scale column: {c}.\
        Returning original dataframe'.format(c=column))
    return df



########################################################################################    
    
def Mask_select(video_dict, fstfile=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            First frame of video that can be drawn upon to define regions of interest.
            
        mask:: [dict]
            Dictionary with the following keys:
                'mask' : boolean numpy array identifying regions to exlude
                         from analysis.  If no such regions, equal to
                         None. [bool numpy array)   
                'mask_stream' : Holoviews stream object enabling dynamic selection 
                        in response to selection tool. `mask_stream.data` contains 
                        x and y coordinates of region vertices. [holoviews polystream]
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #Load first file if batch processing
    if fstfile:
        video_dict['file'] = video_dict['FileNames'][0] 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        if os.path.isfile(video_dict['fpath']):
            print('file: {file}'.format(file=video_dict['fpath']))
            cap = cv2.VideoCapture(video_dict['fpath'])
        else:
            raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
                file=video_dict['fpath']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        video_dict['f0'] = frame
    
    #Make first image the base image on which to draw
    f0 = cropframe(
        video_dict['f0'],
        video_dict.get('crop')
    )
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(width=int(f0.shape[1]*video_dict['stretch']['width']),
               height=int(f0.shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Draw Regions to be Exluded")

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    mask = dict(mask=None)
    poly = hv.Polygons([])
    mask['stream'] = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    #poly_stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def make_mask(data, mask):
        try:
            x_ls, y_ls = data['xs'], data['ys'] 
        except TypeError:
            x_ls, y_ls = [], []
        
        if len(x_ls)>0:
            mask['mask'] = np.zeros(f0.shape) 
            for submask in range(len(x_ls)):
                x = np.array(mask['stream'].data['xs'][submask]) #x coordinates
                y = np.array(mask['stream'].data['ys'][submask]) #y coordinates
                xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
                cv2.fillPoly(mask['mask'], pts =[xy], color=1) #fill polygon  
            mask['mask'] = mask['mask'].astype('bool')
        return hv.Labels((0,0,""))
    
    
    make_mask_ptl = fct.partial(make_mask, mask=mask)        
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask['stream']])
    return image*poly*dmap, mask


def Mask_select_cv2(video_dict, fstfile=False):
    """
    -------------------------------------------------------------------------------------
    
    OpenCV version of Mask_select. Uses click to add vertices for mask regions.
    
    -------------------------------------------------------------------------------------
    Instructions:
        - Click to add vertices
        - Press ENTER to finish current mask region
        - Press 'n' to start a new mask region
        - Press 'r' to reset all masks
        - Press ESC to finish
    
    -------------------------------------------------------------------------------------
    Returns:
        mask:: [dict] with 'mask' key containing boolean numpy array
    """
    
    #Load first file if batch processing
    if fstfile:
        video_dict['file'] = video_dict['FileNames'][0] 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        if os.path.isfile(video_dict['fpath']):
            print('file: {file}'.format(file=video_dict['fpath']))
            cap = cv2.VideoCapture(video_dict['fpath'])
        else:
            raise FileNotFoundError('{file} not found.'.format(file=video_dict['fpath']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(frame,
                (int(frame.shape[1]*video_dict['dsmpl']),
                 int(frame.shape[0]*video_dict['dsmpl'])),
                cv2.INTER_NEAREST)
        video_dict['f0'] = frame
        cap.release()
    
    # Get cropped frame
    f0 = cropframe(video_dict['f0'], video_dict.get('crop'))
    
    # Convert to uint8 if needed
    if f0.dtype != np.uint8:
        f0 = ((f0 - f0.min()) / (f0.max() - f0.min()) * 255).astype(np.uint8)
    
    # Storage for mask polygons
    all_masks = []
    current_points = []
    
    display_img = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR)
    original_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            # Redraw
            display_img = original_img.copy()
            # Draw completed masks
            for mask_pts in all_masks:
                if len(mask_pts) > 2:
                    pts = np.array(mask_pts, np.int32)
                    overlay = display_img.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
                    cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
            # Draw current points
            for i, pt in enumerate(current_points):
                cv2.circle(display_img, tuple(pt), 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display_img, tuple(current_points[i-1]), tuple(pt), (0, 255, 0), 2)
    
    window_name = 'Mask Selection - Click to add points, ENTER to finish region, N for new, ESC to done'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = display_img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    print("\n" + "="*70)
    print("OpenCV Mask Tool - Define regions to EXCLUDE from analysis")
    print("="*70)
    print("Instructions:")
    print("  - CLICK: Add vertex to mask polygon")
    print("  - ENTER: Finish current mask polygon")
    print("  - 'n': Start new mask polygon")
    print("  - 'r': Reset all masks")
    print("  - ESC: Done, apply masks")
    print("="*70)
    
    while True:
        temp_img = display_img.copy()
        cv2.putText(temp_img, "Click to add points, ENTER to finish polygon, ESC when done", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(temp_img, f"Masks: {len(all_masks)} | Current points: {len(current_points)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter - finish current polygon
            if len(current_points) >= 3:
                all_masks.append(current_points.copy())
                print(f"[OK] Mask {len(all_masks)} completed with {len(current_points)} points")
                current_points = []
                # Redraw
                display_img = original_img.copy()
                for mask_pts in all_masks:
                    pts = np.array(mask_pts, np.int32)
                    overlay = display_img.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
                    cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
            else:
                print("[WARNING] Need at least 3 points for a polygon")
                
        elif key == ord('n') or key == ord('N'):  # New polygon
            if len(current_points) >= 3:
                all_masks.append(current_points.copy())
                print(f"[OK] Mask {len(all_masks)} completed, starting new mask")
            current_points = []
            
        elif key == ord('r') or key == ord('R'):  # Reset
            all_masks = []
            current_points = []
            display_img = original_img.copy()
            print("[RESET] Cleared all masks")
            
        elif key == 27:  # ESC - done
            if len(current_points) >= 3:
                all_masks.append(current_points.copy())
            break
    
    cv2.destroyAllWindows()
    
    # Create mask array
    mask = dict(mask=None, stream=None)
    if len(all_masks) > 0:
        mask_array = np.zeros(f0.shape, dtype=bool)
        for mask_pts in all_masks:
            if len(mask_pts) >= 3:
                pts = np.array(mask_pts, np.int32)
                temp_mask = np.zeros(f0.shape, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], 1)
                mask_array = mask_array | (temp_mask == 1)
        mask['mask'] = mask_array
        
        # Use module-level MockStream for pickle compatibility
        xs = [[p[0] for p in m] for m in all_masks]
        ys = [[p[1] for p in m] for m in all_masks]
        mask['stream'] = MockStream({'xs': xs, 'ys': ys})
        
        print(f"\n[COMPLETE] Created {len(all_masks)} mask region(s)")
    else:
        print("\n[SKIP] No mask regions defined")
    
    return mask


def check_p_frames(cap, p_prop_allowed=.01, frames_checked=300):
    """ 
    -------------------------------------------------------------------------------------
    
    Checks whether video contains substantial portion of p/blank frames
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.videocapture]
            OpenCV video capture object.
        p_prop_allowed:: [numeric]
            Proportion of putative p-frames permitted.  Alternatively, proportion of 
            frames permitted to return False when grabbed.
        frames_checked:: [numeric]
            Number of frames to scan for p/blank frames.  If video is shorter
            than number of frames specified, will use number of frames in video.
    
    -------------------------------------------------------------------------------------
    Returns:
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked*p_prop_allowed)
    
    p_frms = 0
    for i in range(frames_checked):
        ret, frame = cap.read()
        p_frms = p_frms+1 if ret==False else p_frms
    if p_frms>p_allowed:
        raise RuntimeError(
            'Video compression method not supported. ' + \
            'Approximately {p}% frames are p frames or blank. '.format(
                p=(p_frms/frames_checked)*100) + \
            'Consider video conversion.')


########################################################################################        
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')

    