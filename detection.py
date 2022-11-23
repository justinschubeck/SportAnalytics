import cv2
import os
import numpy as np

def detect_ball(video_location, file_names, file_type, video_number, camera, resize_w, resize_h, visualize=False):
    print("Starting " + camera + " angle! Double click on ball center, then press 'q'.")
    
    global pos_list
    pos_list = []                                       # Make variable global that stores ball center points.

    def get_ball(event,x,y,flags,param):                # Function for when mouse click initializes ball position.
        if event == cv2.EVENT_LBUTTONDBLCLK:            # If we double click left mouse button.
            pos_list.append((x,y))                      # Add mouse cursor pixel values to list of ball centers.
            print("Initial Ball Position: ",pos_list)   # Display ball centers.

    video_path = video_location + file_names[video_number-1] + camera + file_type   # Create video path string.
    if os.path.exists(video_path):                      # If we can find video. 
        vidcap = cv2.VideoCapture(video_path)           # Setup frame capture.                     

        # --------- Create Background Image ---------- #
        object1MOG = cv2.createBackgroundSubtractorMOG2()               # Create backgound object.
        while True:                                                     # Iterate until we are out of frames.
            has_frames, frame = vidcap.read()
            if not has_frames:                                          # If no frames, exit while loop.
                break
            frame = cv2.resize(frame, (resize_w, resize_h))             # Resize each frame.
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                # Convert frame from Red,Green,Blue to Blue,Green,Red
            object1MOG.apply(img)                                       # Apply the background object to remember frame.
        background = object1MOG.getBackgroundImage()                    # After all frames memorized, create background image. 
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)       # Convert background to grayscale.
        background = cv2.GaussianBlur(background,(5,5),0)               # Apply Gaussian blue to grayscale background.     

        # ----------- Setup Blob Detection ----------- #
        params = cv2.SimpleBlobDetector_Params()                        # Create Blob object.
        if camera == "L":                                               # Left Camera Parameters
            params.minThreshold = 30
            params.filterByArea = True
            params.minArea = 125
            params.maxArea = 750
            params.filterByCircularity = True
            params.minCircularity = 0.4
            params.filterByConvexity = True
            params.minConvexity = 0.4
            params.filterByColor = True
            params.blobColor = 255
        elif camera == "C":                                             # Center Camera Parameters
            params.minThreshold = 30
            params.filterByArea = True
            params.minArea = 250
            params.filterByCircularity = True
            params.minCircularity = 0.4
            params.filterByConvexity = True
            params.minConvexity = 0.4
            params.filterByColor = True
            params.blobColor = 255
        else:                                                           # Right Camera Parameters
            params.minThreshold = 30                                    # Set pixels below this value to zero.
            params.filterByArea = True
            params.minArea = 125
            params.filterByCircularity = True
            params.minCircularity = 0.4
            params.filterByConvexity = True
            params.minConvexity = 0.4
            params.filterByColor = True
            params.blobColor = 255                                    
        detector = cv2.SimpleBlobDetector_create(params)                # Create a background detector for our frames.

        # ------------ Setup Color Mask -------------- #
        if camera == "L":       # Left Camera Parameters
            lower_HSV = np.array([0, 100, 155])
            upper_HSV = np.array([100, 240, 255])
        elif camera == "C":     # Center Camera Parameters
            lower_HSV = np.array([0, 65, 30])
            upper_HSV = np.array([179, 255, 90])
        else:                   # Right Camera Parameters
            lower_HSV = np.array([0, 100, 35])
            upper_HSV = np.array([179, 140, 140])

        # ------------- Common Variables ------------- #
        r_list = []         # Stores radius of ball detected in each frame. No value stored on first initialization.
        x,y = 0,0           # Create variables for segmenting subimage. 
        w,h = 100, 100      # Set width of subimage.
        framecounter = 0    # Count which frame we are processing; indexes pos_list.
        sub_x, sub_y = 0,0  # Center positions of ball.
        
        vidcap = cv2.VideoCapture(video_path)                           # Setup frame capture.
        while True:                                                     # Iterate until we are out of frames.
            has_frames, frame = vidcap.read()
            if not has_frames:                                          # If no frames, exit while loop.
                break
            frame = cv2.resize(frame, (resize_w, resize_h))             # Resize each frame.
            frame_h, frame_w, _ = frame.shape                           # Store frame width and height into variables.

            if framecounter == 0:                                       # Only do this when we are processing first frame.
                cv2.namedWindow('image')                                # Create a popup window for user.
                cv2.setMouseCallback('image',get_ball)                  # Create 'interrupt for mouse.
                while(1):                                               # Wait for user to double click mouse on center of ball.
                    cv2.imshow('image',frame)                           # Show main image of first frame.
                    if cv2.waitKey(0) & 0xFF == ord('q'):               # After double clicking center of ball ONCE, press 'q' on keyboard.
                        break                                           
                cv2.destroyAllWindows()                                 # Stop showing first frame.
                sub_x = pos_list[0][0]                                  # Initialize ball center x position.
                sub_y = pos_list[0][1]                                  # Initialize ball center y position.
                framecounter+=1                                         # Move onto next processing stage.
            else:
                if pos_list[framecounter-1][0] is None:
                    # use last segment that is not None
                    non_None = [i for i in range(len(pos_list)) if pos_list[i][0] != None]
                    index = non_None[-1]
                    sub_x = pos_list[index][0]
                    sub_y = pos_list[index][1]
                else:
                    sub_x = pos_list[framecounter-1][0]
                    sub_y = pos_list[framecounter-1][1]

            # Create bounds for subimage. (0,0 is top left corner of an image)
            if sub_x < w/2:                 
                x = 0                           # Out of bounds on left condition. 
            elif sub_x >= frame_w - (w/2):
                x = int(frame_w - w)            # Out of bounds right condition.
            else:  
                x = int(sub_x - w/2)            # Middle of image condition.

            if sub_y < h/2:
                y = 0                           # Out of bounds top condition.
            elif sub_y >= frame_h - (h/2):
                y = int(frame_h - h)            # Out of bounds bottom condition. 
            else:
                y = int(sub_y - h/2)            # Middle of image condition.
            # print("Segmenting image to be pixels: ("+str(x)+","+str(y)+") to ("+str(x+w)+","+str(y+h)+").")
            image = frame[y:y + h, x:x + w]     # Create subimage. (used by Hough Cirles)
            image1 = image.copy()               # Copy subimage. (used by Hough Cirles)
            image2 = image.copy()               # Copy subimage. (used by Background Subtraction with Blob Detection)
            image3 = image.copy()               # Copy subimage. (used by Color Mask with Blob Detection)
            if visualize:
                cv2.imshow("Sub-Image", image)      # Show subimage

            # ---------- Hough Circles ---------- #
            image_blur = cv2.GaussianBlur(image1,(3, 3),0)              # Apply 3x3 kernel Gaussian Blur to subimage.
            image_gray = cv2.cvtColor(image_blur,cv2.COLOR_BGR2GRAY)    # Convert subimage to grayscale.
            circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,
                                    1,15,param1=200,param2=20,
                                    minRadius=8,maxRadius=20)           # Apply Hough Circles detection method.
            hough_x, hough_y, hough_r = 0,0,0                           # Initialize center points/radius found for Hough Circles.
            if circles is not None:                                     # If the method detected something.
                circles = np.uint16(np.around(circles))                 # Round values. 
                keys = []                                               # List to collect center points of circles.
                rads = []                                               # List to collect radii of circles.
                for i in circles[0,:]:                                  # Iterate over all circles found.
                    cv2.circle(image1,(i[0],i[1]),i[2],(0,0,255),1)     # Draw outer circle.
                    keys.append((i[0]+x, i[1]+y))                       # Collect center points of all circles found.
                    rads.append(int(i[2]))                              # Collect radii of all circles found.
                # use last segment that is not None
                non_None = [i for i in range(len(pos_list)) if pos_list[i][0] != None]
                index = non_None[-1]
                oldcenter = np.array((pos_list[index][0],
                                    pos_list[index][1]))                # Collect the previous center value of ball.
                distances = np.linalg.norm(keys-oldcenter, axis=1)      # Find the distances from all keypoints to previous center.
                min_index = np.argmin(distances)                        # Find which point was closest to last ball center
                hough_x = keys[min_index][0]                            # Store center x point for Hough.
                hough_y = keys[min_index][1]                            # Store center y point for Hough.
                hough_r = rads[min_index]                               # Store radius for Hough.
            if visualize:
                cv2.imshow('Hough on Sub-Image',image1)                     # Show subimage with circles. 
            # ----------------------------------- #

            # ------ Background Subtraction ----- #
            background_sub = background[y:y + h, x:x + w]                   # Create subimage of background.
            diff = cv2.absdiff(background_sub, image_gray)                  # Do background subtraction with gray subimage and subbackground.
            _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)       # Create binary image. 
            diff = cv2.erode(diff, np.ones((5,5),np.uint8), iterations=1)   # Erode small specs of white. 
            diff = cv2.dilate(diff, np.ones((5,5),np.uint8), iterations=1)  # Dilate small holes of black.
            if visualize:
                cv2.imshow("BackSub on Sub-Image", diff)                        # Show background subtraction mask. 

                # --- Blob Detection for BackSub ---- # 
            keypoints_a = detector.detect(diff)                             # Apply detector to get keypoints of blobs.
            img_a = cv2.drawKeypoints(image2, keypoints_a, 
                                np.array([]), (0,0,255), 
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw the keypoints on the subimage. 
            back_x, back_y, back_r = 0,0,0                                  # Initialize center points/radius found for BackSub with Blob.
            if len(keypoints_a) != 0:                                       # If we have a keypoint.
                if len(keypoints_a) == 1:                                   # If we only have one keypoint.
                    back_x = x+keypoints_a[0].pt[0]                         # Calculate center x point of circle on full frame.
                    back_y = y+keypoints_a[0].pt[1]                         # Calculate center y point of circle on full frame.
                    back_r = int(keypoints_a[0].size/2)                     # Calculate radius of circle.
                else:                                                       # If we have more than one keypoint.
                    keys = []                                               # Stores keypoint center values. 
                    for i in range(len(keypoints_a)):                       # Iterate over all keypoints.
                        keys.append((keypoints_a[i].pt[0]+x,                
                                    keypoints_a[i].pt[1]+y))                # Store center values of keypoints.
                    keys = np.array(keys)                                   # Turn into numpy array.
                    # use last segment that is not None
                    non_None = [i for i in range(len(pos_list)) if pos_list[i][0] != None]
                    index = non_None[-1]
                    oldcenter = np.array((pos_list[index][0],
                                        pos_list[index][1]))                # Collect the previous center value of ball.
                    distances = np.linalg.norm(keys-oldcenter, axis=1)      # Find the distances from all keypoints to previous center.
                    min_index = np.argmin(distances)                        # Find which point was closest to last ball center
                    back_x = keys[min_index][0]                             # Store center x point for BackSub with Blob.
                    back_y = keys[min_index][1]                             # Store center y point for BackSub with Blob.
                    back_r = int(keypoints_a[min_index].size/2)             # Store radius for BackSub with Blob
            if visualize:
                cv2.imshow("BackSub Blob Output", img_a)                        # Show BackSub image with blobs outlined. 
                # ----------------------------------- # 
            # ----------------------------------- # 


            # -------------- Color -------------- #
            hsv2 = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)              # Convert subimage to HSV.
            mask2 = cv2.inRange(hsv2, lower_HSV, upper_HSV)                 # Create binary image with color range. 
            mask2 = cv2.erode(mask2, np.ones((3,3),np.uint8), iterations=1) # Erode small specs of white. 
            mask2 = cv2.dilate(mask2, np.ones((5,5),np.uint8), iterations=2)# Dilate small holes of black.
            if visualize:
                cv2.imshow("Color Mask", mask2)                                 # Show Color mask. 

                # ---- Blob Detection for Color ----- # 
            keypoints_b = detector.detect(mask2)                            # Apply detector to get keypoints of blobs.
            img_b = cv2.drawKeypoints(image3, keypoints_b, 
                        np.array([]), (0,0,255), 
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)         # Draw the keypoints on the subimage. 
            color_x, color_y, color_r = 0,0,0                               # Initialize center points/radius found for Color with Blob.
            if len(keypoints_b) != 0:                                       # If we have a keypoint.
                if len(keypoints_b) == 1:                                   # If we only have one keypoint.
                    color_x = x+keypoints_b[0].pt[0]                        # Calculate center x point of circle on full frame.
                    color_y = y+keypoints_b[0].pt[1]                        # Calculate center y point of circle on full frame.
                    color_r = int(keypoints_b[0].size/2)                    # Calculate radius of circle.
                else:                                                       # If we have more than one keypoint.
                    keys = []                                               # Stores keypoint center values. 
                    for i in range(len(keypoints_b)):                       # Iterate over all keypoints.
                        keys.append((keypoints_b[i].pt[0]+x,                
                                    keypoints_b[i].pt[1]+y))                # Store center values of keypoints.
                    keys = np.array(keys)                                   # Turn into numpy array.
                    # use last segment that is not None
                    non_None = [i for i in range(len(pos_list)) if pos_list[i][0] != None]
                    index = non_None[-1]
                    oldcenter = np.array((pos_list[index][0],
                                        pos_list[index][1]))                # Collect the previous center value of ball.
                    distances = np.linalg.norm(keys-oldcenter, axis=1)      # Find the distances from all keypoints to previous center.
                    min_index = np.argmin(distances)                        # Find which point was closest to last ball center
                    color_x = keys[min_index][0]                            # Store center x point for Color with Blob.
                    color_y = keys[min_index][1]                            # Store center y point for Color with Blob.
                    color_r = int(keypoints_b[min_index].size/2)            # Store radius for Color with Blob
            if visualize:
                cv2.imshow("Color Blob Output", img_b)                          # Show Color image with blobs outlined. 
                # ----------------------------------- #
            # ----------------------------------- #

            # ----------------- Update Stored Values ------------------ #
            if camera == "R":
                if circles is not None:                             # If Hough found circle.
                    pos_list.append((hough_x,hough_y))              # Append center from Hough to list. 
                    r_list.append((hough_r))                        # Append radius from Hough to list.
                    print("Adding point ("+str(hough_x)+','+str(hough_y)+') to pos_list from Hough.')
                elif len(keypoints_b) != 0:                         # If Color with Blob found circle when Hough did not.
                    pos_list.append((int(color_x),int(color_y)))    # Append center from Color with Blob to list.
                    r_list.append((color_r))                        # Append radius from Color with Blob to list.
                    print("Adding point ("+str(int(color_x))+','+str(int(color_y))+') to pos_list from Color.')
                elif len(keypoints_a) != 0:                         # If BackSub with Color found circle when Hough and Color did not.
                    pos_list.append((int(back_x),int(back_y)))      # Append center from BackSub with Blob to list.
                    r_list.append((back_r))                         # Append radius from BackSub with Blob to list.
                    print("Adding point ("+str(int(back_x))+','+str(int(back_y))+') to pos_list from Background.')
                else:                                               # No method found ball. Estimate where next subimage should be.
                    # predx, predy = 0, 0                             # Initialize center of next SubImage.
                    # if len(pos_list) > 1:                           # Linearly predict ball movement to next SubImage with last two centers.
                    #     predx = pos_list[-1][0] + (pos_list[-1][0] - pos_list[-2][0])
                    #     predy = pos_list[-1][1] + (pos_list[-1][1] - pos_list[-2][1])
                    #     if predx < 0:
                    #         predx = 0                               # Out of bounds on left condition. 
                    #     elif predx >= frame_w:
                    #         predx = frame_w-1                       # Out of bounds on right condition. 
                    #     if predy < 0:
                    #         predy = 0                               # Out of bounds on top condition. 
                    #     elif predy >= frame_h:
                    #         predy = frame_h-1                       # Out of bounds on bottom condition. 
                    # else:                                           # If we only have one previous center, store that point.
                    #     predx = pos_list[-1][0]
                    #     predy = pos_list[-1][1]
                    # pos_list.append((predx,predy))                  # Add predicted center to list of ball centers.
                    # r_list.append(int(sum(r_list)/len(r_list)))     # Append predicted radius as average.
                    # print("Adding point ("+str(int(predx))+','+str(int(predy))+') to pos_list from Prediction.')

                    # NOT PREDICTING ANYMORE ^^^ : Using None
                    pos_list.append((None,None))                  # Append None to list of ball centers.
                    r_list.append(None)                           # Append None to list of ball radii.
                    print("Adding point (None, None) to pos_list.")

            else:
                if len(keypoints_b) != 0:                           # If Color with Blob found circle when Hough did not.
                    pos_list.append((int(color_x),int(color_y)))    # Append center from Color with Blob to list.
                    r_list.append((color_r))                        # Append radius from Color with Blob to list.
                    print("Adding point ("+str(int(color_x))+','+str(int(color_y))+') to pos_list from Color.')
                elif circles is not None:                           # If Hough found circle.
                    pos_list.append((hough_x,hough_y))              # Append center from Hough to list. 
                    r_list.append((hough_r))                        # Append radius from Hough to list.
                    print("Adding point ("+str(hough_x)+','+str(hough_y)+') to pos_list from Hough.')
                elif len(keypoints_a) != 0:                         # If BackSub with Color found circle when Hough and Color did not.
                    pos_list.append((int(back_x),int(back_y)))      # Append center from BackSub with Blob to list.
                    r_list.append((back_r))                         # Append radius from BackSub with Blob to list.
                    print("Adding point ("+str(int(back_x))+','+str(int(back_y))+') to pos_list from Background.')
                else:                                               # No method found ball. Estimate where next subimage should be.
                    # predx, predy = 0, 0                             # Initialize center of next SubImage.
                    # if len(pos_list) > 1:                           # Linearly predict ball movement to next SubImage with last two centers.
                    #     predx = pos_list[-1][0] + (pos_list[-1][0] - pos_list[-2][0])
                    #     predy = pos_list[-1][1] + (pos_list[-1][1] - pos_list[-2][1])
                    #     if predx < 0:
                    #         predx = 0                               # Out of bounds on left condition. 
                    #     elif predx >= frame_w:
                    #         predx = frame_w-1                       # Out of bounds on right condition. 
                    #     if predy < 0:
                    #         predy = 0                               # Out of bounds on top condition. 
                    #     elif predy >= frame_h:
                    #         predy = frame_h-1                       # Out of bounds on bottom condition. 
                    # else:                                           # If we only have one previous center, store that point.
                    #     predx = pos_list[-1][0]
                    #     predy = pos_list[-1][1]
                    # pos_list.append((predx,predy))                  # Add predicted center to list of ball centers.
                    # r_list.append(int(sum(r_list)/len(r_list)))     # Append predicted radius as average.
                    # print("Adding point ("+str(int(predx))+','+str(int(predy))+') to pos_list from Prediction.')

                    # NOT PREDICTING ANYMORE ^^^ : Using None
                    pos_list.append((None,None))                  # Append None to list of ball centers.
                    r_list.append(None)                           # Append None to list of ball radii.
                    print("Adding point (None, None) to pos_list.")
            # --------------------------------------------------------- #

            # --------- Show Full Image With Data Predictions --------- #
            # Display detected or predicted ball outline and center. 
            if pos_list[framecounter] is not None:
                cv2.circle(frame,(pos_list[framecounter][0],pos_list[framecounter][1]),r_list[framecounter-1],(0,255,0),1)
                cv2.circle(frame,(pos_list[framecounter][0],pos_list[framecounter][1]),2,(0,0,255),2)
                cv2.putText(frame, "Detected Ball Position: ("+str(pos_list[framecounter][0])+","+str(pos_list[framecounter][1])+")", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if visualize:
                cv2.imshow('detected circle on main image',frame)
            # --------------------------------------------------------- #
            
            if visualize:
                cv2.waitKey(0)              # Wait to update until user presses enter.
                cv2.destroyAllWindows()     # Close all open windows.
            framecounter+=1                 # Update frame index. 

        print("Done with " + camera + " angle!")
        return pos_list, r_list

    else:
        print('Invalid video path: ' + video_path)
        print("Done with " + camera + " angle!")
        return "Error", "Error"   