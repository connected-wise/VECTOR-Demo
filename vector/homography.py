import cv2 as cv2
import numpy as np

def findcorners(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    # Binary image
    lower_bound = np.array([0,100,100])
    upper_bound = np.array([179,255,255])
    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    reds = 255 - cv2.inRange(hsv, lower_bound, upper_bound)
    blur = cv2.GaussianBlur(reds, (5,5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    


    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return opening
    
     
def homography(im_src, 
               corners, 
               save=False, 
               output="data/images/panel.jpg",
               output_size=(1920,320)
               ):

    # Read source image.
    #im_src = cv.imread(source)

    scale_percent = 60 # percent of original size
    width = int(im_src.shape[1] * scale_percent / 100)
    height = int(im_src.shape[0] * scale_percent / 100)
    dim = (width, height)
    # dim = im_src.shape

    
    # resize image
    im_src = cv2.resize(im_src, dim, interpolation = cv.INTER_AREA)
    

    # Four corners of the book in source image
    # pts_src = np.array([[8, 14], [763, 17], [24, 379], [742, 392]])
    pts_src = np.array(corners)

    width, height = output_size[0], output_size[1]

    # Read destination image.
    im_dst = cv2.resize(im_src, (width, height),
               interpolation = cv2.INTER_NEAREST)

    # Four corners of the book in destination image.
    pts_dst = np.array([[0, 0],[width, 0],[0, height],[width, height]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    # Display images
    #cv.imshow("Source Image", im_src)
    #cv.imshow("Warped Source Image", im_out)
    if save:
        cv2.imwrite(output, im_out)
        print("Result successfully saved in ", output)

    #cv.waitKey(0)
    return im_out