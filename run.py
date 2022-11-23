from calibration import *
from data import *
from detection import *
import numpy as np
from plotting import analysis


# ------------------------- Inputs ------------------------- # 
# Frame Resizing (If this is changed, some fof the detection parameters must also be changed.)
width = 960
height = 540

# Camera Calibration (Holds chessboard images, with a 'L', 'C', and 'R' folder.)
calibration_folder = "calibration_images"

# Ball Detection
video_location = "videos\\"
file_names = ["1_FT_","2_FT_","3_FT_","4_MISS_","5_MISS_","6_MISS_"]
file_type = ".mp4"
video_numbers = [1,2,3,4,5,6] # Options are 1 through 6 for the videos listed in 'file_names'.

# ----------------------- END: Inputs ---------------------- # 


# -------------------- Calibrate Cameras ------------------- # 
# Calculate the intrinsic matrix and distortion coefficients of each recording camera. 
intrinsic_L, distortion_L = calibrate_chessboard(calibration_folder + "/L")
intrinsic_C, distortion_C = calibrate_chessboard(calibration_folder + "/C")
intrinsic_R, distortion_R = calibrate_chessboard(calibration_folder + "/R")
# print(intrinsic_L, distortion_L)
# print(intrinsic_C, distortion_C)
# print(intrinsic_R, distortion_R)

# Calculate the projection matrix for each recording camera.
rotation_vector_L, translation_L, projection_L = calibrate_camera(coord_system_3D, coord_system_2D_L, intrinsic_L, distortion_L)
rotation_vector_C, translation_C, projection_C = calibrate_camera(coord_system_3D, coord_system_2D_C, intrinsic_C, distortion_C)
rotation_vector_R, translation_R, projection_R = calibrate_camera(coord_system_3D, coord_system_2D_R, intrinsic_R, distortion_R)
# print(rotation_vector_L, translation_L, projection_L)
# print(rotation_vector_C, translation_C, projection_C)
# print(rotation_vector_R, translation_R, projection_R)

# ------------------ END: Calibrate Cameras ---------------- # 

# Iterate over all videos to run the process on. 
for video_number in video_numbers:
    print("Processing beginnning for video " + str(video_number) +'!')

    # --------------------- Ball Detection --------------------- # 
    # !! EACH ANGLE: Double click ball center, then press 'q' !! #
    ball_centers_L, radius_values_L = detect_ball(video_location, file_names, file_type, video_number, "L", width, height, visualize=False)
    ball_centers_C, radius_values_C = detect_ball(video_location, file_names, file_type, video_number, "C", width, height, visualize=False)
    ball_centers_R, radius_values_R = detect_ball(video_location, file_names, file_type, video_number, "R", width, height, visualize=False)
    # ------------------ END: Ball Detection ------------------- # 


    # --------------------- Triangulation ---------------------- # ]
    ball_centers_3D = []
    for i in range(len(ball_centers_L)):
        # Get all the 2D points for ball centers. 
        tri_L = np.array([ball_centers_L[i][0], ball_centers_L[i][1]], dtype=np.float32)
        tri_C = np.array([ball_centers_C[i][0], ball_centers_C[i][1]], dtype=np.float32)
        tri_R = np.array([ball_centers_R[i][0], ball_centers_R[i][1]], dtype=np.float32)


        if ball_centers_L[i][0] is None and ball_centers_C[i][0] is None and ball_centers_R[i][0] is None:
            # Ball not detected. 
            ball_centers_3D.append([None, None, None, 0])
        elif (ball_centers_L[i][0] is None and ball_centers_C[i][0] is None) or (ball_centers_C[i][0] is None and ball_centers_R[i][0] is None) or (ball_centers_L[i][0] is None and ball_centers_R[i][0] is None):
            # If only one camera detected the ball, we cannot triangulate.
            ball_centers_3D.append([None, None, None, 1])
        elif ball_centers_L[i][0] is None:
            # If left camera did not detect ball, use center and right camera to triangulate. 
            data = cv2.triangulatePoints(projection_C, projection_R, tri_C, tri_R)
            data /= data[3]
            data = data.T[0, :]
            ball_centers_3D.append([data[0], data[1], data[2], 2])
        elif ball_centers_C[i][0] is None:
            # If center camera did not detect ball, use left and right camera to triangulate. 
            data = cv2.triangulatePoints(projection_L, projection_R, tri_L, tri_R)
            data /= data[3]
            data = data.T[0, :]
            ball_centers_3D.append([data[0], data[1], data[2], 2])
        elif ball_centers_R[i][0] is None:
            # If right camera did not detect ball, use left and center camera to triangulate. 
            data = cv2.triangulatePoints(projection_L, projection_C, tri_L, tri_C)
            data /= data[3]
            data = data.T[0, :]
            ball_centers_3D.append([data[0], data[1], data[2], 2])
        else:
            # If all three cameras detected ball, use the two angles that missed detection points the least. 
            none_count_L, none_count_C, none_count_R  = 0, 0, 0
            for i in range(len(ball_centers_L)):
                if ball_centers_L[i][0] is None:
                    none_count_L += 1
                if ball_centers_C[i][0] is None:
                    none_count_C += 1
                if ball_centers_R[i][0] is None:
                    none_count_R += 1
            
            if (none_count_L > none_count_C) and (none_count_L > none_count_R):
                data = cv2.triangulatePoints(projection_C, projection_R, tri_C, tri_R)
                data /= data[3]
                data = data.T[0, :]
                ball_centers_3D.append([data[0], data[1], data[2], 3])
            elif (none_count_C > none_count_L) and (none_count_C > none_count_R):
                data = cv2.triangulatePoints(projection_L, projection_R, tri_L, tri_R)
                data /= data[3]
                data = data.T[0, :]
                ball_centers_3D.append([data[0], data[1], data[2], 3])
            else:
                data = cv2.triangulatePoints(projection_L, projection_C, tri_L, tri_C)
                data /= data[3]
                data = data.T[0, :]
                ball_centers_3D.append([data[0], data[1], data[2], 3])

    ball_centers_3D = np.array(ball_centers_3D) # Array of ball center points in 3D plane, with extra column for how many cameras captured ball.
    analysis(ball_centers_3D) # Plot 3D interactive plot of trajectory. 
    
    print("Press Enter to end analysis for video " + str(video_number) +':')
    input()     # Keep interactive window open until user presses enter. 
    print("Done with video " + str(video_number) +'.')
    # ------------------ END: Triangulation -------------------- # 