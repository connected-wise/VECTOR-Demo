from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.gridspec as gridspec


# keeping the frame + sign box tos how at the result
det = None
frame_count = 0
# Function to order points for perspective transformation


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Function to perform perspective transform
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def panel_enhancer(image_to_enhance):

    # Convert the image to grayscale for processing
    gray_image_to_enhance = cv2.cvtColor(image_to_enhance, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve the contrast of the image
    equalized_image = cv2.equalizeHist(gray_image_to_enhance)

    # Now, we'll attempt to remove the background. Since the LEDs emit light and have a distinct color,
    # we can use color segmentation to isolate them from the background.
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image_to_enhance, cv2.COLOR_BGR2HSV)

    # Define the range of the red color in HSV
    # These values can be adjusted to capture the color of the LEDs more accurately
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Define range for upper range of red
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the masks for lower and upper red ranges
    mask = mask1 + mask2

    # Bitwise-AND mask and original image to isolate the LEDs
    res = cv2.bitwise_and(image_to_enhance, image_to_enhance, mask=mask)

    # Initialize minimum area rectangle
    min_area_rect = None
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangle for the largest contour which should be encompassing all LEDs
    if contours:
        # Merge all contours into one for the purpose of finding the minimum area bounding box
        all_contours = np.vstack(contours[i] for i in range(len(contours)))
        min_area_rect = cv2.minAreaRect(all_contours)

        # Get the box points and draw them
        box = cv2.boxPoints(min_area_rect)
        box = np.int0(box)

        # be sure if the box is colse to the aspect ratio of the panel

        dist_1 = np.linalg.norm(box[0] - box[1])
        dist_2 = np.linalg.norm(box[1] - box[2])

        # Width and height of the rectangle
        width = max(dist_1, dist_2)
        height = min(dist_1, dist_2)

        # Calculate aspect ratio (width / height)
        aspect_ratio = width / height if height > 0 else 0
        # print(aspect_ratio)
        if aspect_ratio > target_ratio[0] and aspect_ratio < target_ratio[1]:
            cv2.drawContours(res, [box], 0, (0, 255, 0), 2)
        else: 
            return
    else:
        return

    # Return the angle of the rotated rectangle for further analysis
    min_area_rect[2] if min_area_rect else None
    # Use the four point transform to make the rectangle area a straight rectangle and crop the outside

    warped_image = four_point_transform(res, box)

    # Draw a 4x20 grid on the warped image
    grid_image = cv2.resize(warped_image.copy(), (0, 0), fx=10, fy=10)
    height, width = grid_image.shape[:2]

    # Number of cells in each direction
    num_cells_vertical = 4
    num_cells_horizontal = 24

    # The step size for each cell
    step_size_x = width // num_cells_horizontal
    step_size_y = height // num_cells_vertical

    # Draw the corrected grid
    for x in range(num_cells_horizontal):
        cv2.line(grid_image, (int(x * step_size_x), 0), (int(x * step_size_x), height), (255, 0, 0), 5)

    for y in range(num_cells_vertical):
        cv2.line(grid_image, (0, int(y * step_size_y)), (width, int(y * step_size_y)), (255, 0, 0), 5)

    # Plotting the result in two columns
    processed_imgs = [image_to_enhance, res, grid_image]
    processed_img_titles = ['Cropped panel image',
                            'Enhanced panel with rotated bounding box', 'Aligned panel with grid']

    # plt.figure(figsize=(16, 8))

    # Making a grid to fill for the plot
    gs = gridspec.GridSpec(3, 2)

    # Left side: frmae + box
    ax1 = plt.subplot(gs[0:3, 0])
    # ax1.imshow(cv2.cvtColor(det, cv2.COLOR_BGR2RGB)) 
    ax1.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)) 
    ax1.set_title(f'Camera Object Detection frame {frame_count}')
    ax1.axis('off')

    # Right side: 3 vertically stacked images
    for i in range(3):
        ax = plt.subplot(gs[i, 1])

        ax.imshow(cv2.cvtColor(processed_imgs[i], cv2.COLOR_BGR2RGB))
        ax.set_title(processed_img_titles[i])
        ax.axis('off')

    plt.show()
    plt.draw()
    plt.pause(0.1)  # Small delay

    # # Optional: Clear the plot
    plt.clf()

#Main Function:
model_path = 'VECTOR-detect.pt'
video_path = 0  # for webcam
video_path = '/home/cw/vec-backup/VECTOR-Demo/vector/panel-test.mp4'  # for video file
# detecting class 7 with > 0.6 confidence
sign_det_conf = 0.6
panel_cls = 7
target_ratio = [4,6] # target aspect ratio of the panel

plt.ion()
plt.figure(figsize=(16, 8))

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count+=1
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=sign_det_conf)
        annotated_frame = results[0].plot()

        # Iterate through detections
        for r in results:
            # Extract bounding box coordinates,
            boxes = r.boxes.xyxy.tolist()
            classes = r.boxes.cls.tolist()
            for box, cls in zip(boxes, classes):
                if cls == panel_cls:
                    x1, y1, x2, y2 = map(int, box)
                    # det = frame.copy()
                    # cv2.rectangle(det, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    sign = frame[y1:y2, x1:x2]
                    panel_enhancer(sign)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
