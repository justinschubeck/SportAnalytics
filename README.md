# Sport Analytics (Basketball)

## University of Florida : [Electrical and Computer Engineering](https://www.ece.ufl.edu/)<br />EEL4924C - Design 2 Spring 2022
Faculty: [Dr. Bobda](https://bobda.ece.ufl.edu/) <br />
Mentor: [Erman Nghonda](https://smartsystems.ece.ufl.edu/people/enghonda/) <br />
Students: [Justin Schubeck](https://www.linkedin.com/in/justinschubeck/), [Alex Liu](https://www.linkedin.com/in/alex-liu-m1/) <br />

## Project Overview
The purpose of this project is to capture, track, and analyze basketball free throw
trajectories. Each shot was captured via iPhone video camera from more than one angle to properly analyze trajectories, distances, velocities, and 3D planes. This collected data could be used in conjunction with computer vision or machine learning to extract critical information about the free throw tactics and routine of a player. 

## run.py
This file will run the entire pipeline. It will call the functions to: calibrate the cameras on the chessboards, calculate the necessary camera matrices, detect the ball centers, triangulate points, and plot analysis for each video specified. All inputs must be changed in the run.py file itself. Inputs consist of: <br />
* ```width``` - integer width of frame resizing for processing
* ```height``` - integer height of frame resizing for processing
* ```calibration_folder``` - string name of the folder containing the chessboard images
* ```video_location``` - string name of folder with videos to be processed
* ```file_names``` - list of string prefixes of video names (must start with number and end with angle 'L', 'C', or 'R')
* ```file_type``` - string file extension of videos
* ```video_numbers``` - list of integer prefixed number of video(s) to process in list file_names

**To run this file, the user will:**
1. Update any inputs in the file first. 
2. Type ```python run.py``` in the terminal.
3. Double click the ball center to initialize first camera. Then press 'q' on keyboard.
4. Double click the ball center to initialize second camera. Then press 'q' on keyboard.
5. Double click the ball center to initialize third camera. Then press 'q' on keyboard.
6. View plots, then press 'Enter' to finish analysis on current video.
7. Repeats step 3-6 if more than one video number is specified. 

## data.py
This file contains 3D and 2D coordinates of key features in the video. Video must be recorded on tripod for the calibration and coordinate system definition to work. Twelve keypoints were marked, which were manually measured in real world dimensions, as well as in pixel values of a frame for each camera angle. The conversion of 100 3D units to 70 inches was chosen arbitrarily.

![3D Defined Coordinate System](/documentation_images/3D_Points.jpg)

![2D Keypoints Left](/documentation_images/L_2D_Points.png)

![2D Keypoints Center](/documentation_images/C_2D_Points.png)

![2D Keypoints Right](/documentation_images/R_2D_Points.png)

## calibration.py
This file is called before any detection or triangulation in order to calibrate the cameras that were used for recording. It contains two functions:

<br />**calibrate_chessboard()**<br />
Inputs:
* ```folder``` - string path to folder with chessboard calibration images for camera

Outputs:
* ```i_mtx``` - matrix of intrinsic values of camera being calibrated
* ```dist``` - array of distortion coefficients of camera being calibrated

<br />**calibrate_camera()**<br />
Inputs:
* ```Calib_3D``` - array of 3D defined coordinate system keypoints
* ```Calib_2D``` - array of 2D defined keypoints for camera being calibrated
* ```intrinsic``` - matrix of intrinsic values of camera being calibrated
* ```dist``` - array of distortion coefficients of camera being calibrated

Outputs:
* ```rvec``` - array of rotation vector values for camera being calibrated
* ```tvec``` - array of translateion vector values for camera being calibrated 
* ```P``` - matrix of projections values for camera being calibrated

![Example Calibrate Chessboard](/documentation_images/Chessboard_Calibration_Example.jpeg)

![Test Projection Left](/documentation_images/Test_Projection_Left.jpeg)
![Test Projection Center](/documentation_images/Test_Projection_Center.jpeg)
![Test Projection Right](/documentation_images/Test_Projection_Right.jpeg)

## detection.py
This file contains one function that performs segmentation of frames, boosting three detection algorithms, and storing of 2D data.

<br />**detect_ball()**<br />
Inputs:
* ```video_location``` - string name of folder with videos to be processed
* ```file_names``` - list of string prefixes of video names (must start with number and end with angle 'L', 'C', or 'R')
* ```file_type``` - string file extension of videos
* ```video_number``` - integer prefixed number of video to process in list file_names
* ```camera``` - string angle of camera to detect ball ('L', 'C', or 'R')
* ```resize_w``` - integer width of frame resizing for processing
* ```resize_h``` - integer height of frame resizing for processing
* ```visualize``` - boolean that shows detection for each frame

Outputs:
* ```pos_list``` - list of x,y pixel values for ball center detected per frame
* ```r_list``` - list of radii values for ball detected per frame

![Example Detection](/documentation_images/Detection_Example.png)

## plotting.py
This file contains performs the analysis of ball detection and triangulation. Figures are displayed in the form of a 3D interactive plot with detected 3D points and trajectory, along with total, x direction, y direction, and z direction velocities over time.

<br />**get_velocities()**<br />
Inputs:
* ```vals``` - array of 3D triangulated points with number of cameras detected at this point label

Outputs:
* ```times``` - list of time stamps for times where instantaneous velocity can be calculated
* ```velocities``` - velocities in miles per hour for times where instantaneous velocity can be calculated
* ```vel_x``` - velocities in x direction in 3D units per second for times where instantaneous velocity can be calculated
* ```vel_y``` - velocities in y direction in 3D units per second for times where instantaneous velocity can be calculated
* ```vel_z``` - velocities in z direction in 3D units per second for times where instantaneous velocity can be calculated
* ```indices``` - indices in the vals input array for which instantaneous velocites were found

<br />**analysis()**<br />
Inputs:
* ```points_3D``` - array of 3D triangulated points with number of cameras detected at this point label

![Example 3D Plot](/documentation_images/Example_3D_Plot.png)
![Example Velocities Plot](/documentation_images/Example_Velocities_Plot.png)
![Example Height Plot](/documentation_images/Example_Height_Plot.png)