import numpy as np
import cv2 as cv

# input --> gray --> binary --> edge --> hough


# Read the image in grayscale
image = cv.imread("data/images/panel-raw.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# blur = cv.medianBlur(gray, 5)

# cv.imshow('original image', image)
# cv.imshow('gray scale image', gray)


# Binary image
binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU )[1]
# cv.imshow('binary image', binary)


# Edge detection
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
edge = cv.Canny(binary, t_lower, t_upper)
cv.imshow('edge detection', edge)


# Hough Transform
hough = image.copy()

circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, 1, 10, param1=100, param2=50, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(hough, (i[0],i[1]), i[2], (0,255,0), 2)
    # # draw the center of the circle
    # cv.circle(hough, (i[0],i[1]), 2, (0,0,255), 3)

# cv.imshow('hough transform', np.hstack([image, hough]))
cv.imshow('hough transform', hough)

# cv.imwrite('edge.jpg', edge)
# cv.imwrite('hough.jpg', hough)

hough1 = image.copy()
circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, 1, 10, param1=100, param2=50, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv.circle(hough1, (i[0],i[1]), i[2], (0,255,0), 2)
cv.imshow('hough transform 1', hough1)


hough2 = image.copy()
circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, 1, 10, param1=100, param2=50, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv.circle(hough2, (i[0],i[1]), i[2], (0,255,0), 2)
cv.imshow('hough transform 2', hough2)



# Wait for ESC and close windows
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()