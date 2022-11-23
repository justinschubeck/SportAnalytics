import numpy as np
import cv2
import glob

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def calibrate_chessboard(folder):

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./' + str(folder) + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If desired number of corner are detected,
        # we refine the pixel coordinates and display 
        # them on the images of checker board
        if ret == True:
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # Performing camera calibration by 
    # passing the value of known 3D points (objpoints)
    # and corresponding pixel coordinates of the 
    # detected corners (imgpoints)
    ret, i_mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return i_mtx, dist

def calibrate_camera(Calib_3D, Calib_2D, intrinsic, dist):
    '''
    Function to generate necessary camera data between one camera angle and the 3D coordinate system.
        input: 3D points, corresponding 2D camera image points, intrinsic matrix, distortion coefficients
        output: rotation vector, translation vector, projection matrix
    '''
    # generate rotation vector and translation vector from 3D and 2D calibration points
    _, rvec, tvec = cv2.solvePnP(Calib_3D, Calib_2D, intrinsic, dist, flags=0)

    # convert rotation vector to rotation matrix
    rmtx, _ = cv2.Rodrigues(rvec)

    # concactenate rotation matrix and translation vector
    rt = np.concatenate((rmtx, tvec),axis=1)

    # perform matrix multiplication to obtain projection matrix
    P = intrinsic@rt

    return rvec, tvec, P