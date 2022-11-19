import numpy as np
import cv2 as cv

side_len = 80
panel_size = (4,24)
# min_side = 70
# max_side = 80
# min_area = min_side**2
# max_area = max_side**2
# min_perimeter = 4*min_side
# max_perimeter = 4*max_side


def bit_detection(image):

    # ------------------ Preprocessing --------------------- #

    # Denoise image
    # Remove noises by Gaussian filter
    mask = cv.GaussianBlur(image, (5,5), 0)

    # Binary image
    lower_bound = np.array([0,100,100])
    upper_bound = np.array([179,255,255])
    # Convert to HSV format and color threshold
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(hsv, lower, upper)
    # result = cv.bitwise_and(image, image, mask=mask)

    mask = cv.inRange(hsv, lower_bound, upper_bound)
    # cv.imshow("mask0", mask)


    # Enhance the mask by using Morphological transfomation
    # This is an operation based on the shape of an image
    kernel = np.ones((6, 6), np.uint8)      # Kernel

    mask = cv.erode(mask, kernel, iterations=5)     # Erosion
    mask = cv.dilate(mask, kernel, iterations=3)    # Dilation
    # mask = np.invert(mask)                          # Inverse Binary

    # cv.imshow("mask", mask)


    # ------------------ Contours --------------------- #

    # Find all contours in the image based on the mask
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours

    # TO-DO:
    # This sort is not necessery, but we need to convert contours from tuple to list
    # There should be a better way than sorted function
    contours = sorted(contours, key=lambda x:cv.boundingRect(x)[1])         # Convert tuple to list

    # # Remove all contours that detect by mistake
    # for i, c in enumerate(contours):
    #     # Filter by Counter Area
    #     areaContour=cv.contourArea(c)   # Calculate area of contour
    #     if areaContour < min_area or areaContour > max_area:
    #         contours.pop(i)
    #         continue

    #     # Filter by Counter Perimeter
    #     perimeter = cv.arcLength(c, True)
    #     if perimeter < min_perimeter or perimeter > max_perimeter:
    #         contours.pop(i)
    #         continue

    #     # Filter by Contour Shape
    #     approx = cv.approxPolyDP(c, 0.03*perimeter, True)
    #     if not len(approx) == 4:
    #         contours.pop(i)
    #         continue
 

    # Classify contours based on the row that they belong
    # row = 0
    # sorted_contours = [[], [], [], [], [], [], [], []]
    # # sorted_contours = [[], [], [], [], []]
    # cx0, cy0 = cv.minAreaRect(contours[0])[0]
    # for c in contours:
    #     cx, cy = cv.minAreaRect(c)[0]
    #     if abs(cy0-cy) > 20.0:
    #         cx0, cy0 = cx, cy
    #         row += 1
    #     sorted_contours[row].append(c)
    row = 0
    sorted_contours = [[], [], [], [], [], [], [], []]
    # sorted_contours = [[], [], [], [], []]
    cx0, cy0 = cv.minAreaRect(contours[0])[0]
    for c in contours:
        cx, cy = cv.minAreaRect(c)[0]
        if 10 < cy < 70:
            row = 0
        elif 90 < cy < 150:
            row = 1
        elif 170 < cy < 230:
            row = 2
        elif 250 < cy < 310:
            row = 3
        # if abs(cy0-cy) > 20.0:
        #     cx0, cy0 = cx, cy
        #     row += 1
        sorted_contours[row].append(c)

    # Sort contours in a row from left to right
    i = 0
    for row in sorted_contours:
        row = sorted(row, key=lambda x:cv.boundingRect(x)[0])
        for c in row:
            contours[i] = c 
            i += 1  

    # # TO-DO:
    # # This sort is not necessery, but we need to convert contours from tuple to list
    # # There should be a better way than sorted function
    # contours = sorted(contours, key=lambda x:cv.boundingRect(x)[1])         # Convert tuple to list      

    # Draw contours
    cv.drawContours(image, contours, contourIdx=-1, color=(125, 125, 0), thickness=2)

    # Label contours
    for i, c in enumerate(contours):
        # Claculate the coordinates of the label
        M = cv.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # Draw the label
        cv.putText(image, text=str(i+1), org=(cx,cy),
                fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                thickness=2, lineType=cv.LINE_AA)


    # # ------------------ Positioning --------------------- #

    # Positiong
    panel = [[0 for i in range(panel_size[1])] for j in range(panel_size[0])]
    row, col = (0, 0)       # Panel's row and column indicator
    cx0, cy0 = (side_len/2, side_len/2)
    for i, c in enumerate(contours):
        cx, cy = cv.minAreaRect(c)[0]                   # Coordination of current bit's center
        
        if 10 < cy < 70:
            row = 0
        elif 90 < cy < 150:
            row = 1
        elif 170 < cy < 230:
            row = 2
        elif 250 < cy < 310:
            row = 3

        if i >= len(contours)-1:
            col = round(abs(cx0 - cx1)/side_len)
            panel[row][col] = 1
            break
        
        cx1, cy1 = cv.minAreaRect(contours[i+1])[0]     # Coordination of next bit's center

        col = round(abs(cx0 - cx)/side_len)
        panel[row][col] = 1

        # if abs(cy - cy1) > 60:       # End of row
        #     row += 1

    # Print Result
    for row in panel:
        print(*row, sep=' ', end='\n')


    # Seprate the panel to code and check
    code = []
    check = []
    for i, row in enumerate(panel):
        for bit in row:
            if i == 0 or i == 1:
                code.append(bit)
            if i == 2 or i == 3:
                check.append(bit)
            
    # Show result
    #cv.imshow("preprocessed", image)
    #cv.waitKey(0)

    return code, check, image