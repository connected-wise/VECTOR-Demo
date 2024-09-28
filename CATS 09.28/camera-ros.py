from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO, checks
import cv2
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans

checks()

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
def four_point_transform(image, pts, margin_percentage=-0.005):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width and height of the rectangle
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Calculate margin to be inset from the edges of the rectangle
    margin_w = int(maxWidth * margin_percentage)
    margin_h = int(maxHeight * margin_percentage)

    # Adjust the points of the rectangle to inset by the margin
    dst = np.array([
        [margin_w, margin_h],
        [maxWidth - margin_w - 1, margin_h],
        [maxWidth - margin_w - 1, maxHeight - margin_h - 1],
        [margin_w, maxHeight - margin_h - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped

# Function to perform a kmeans segmentation
def kmeans_segmentation(image, k=3):
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    _pixel_values = image.reshape((-1, 3))
    _pixel_values = np.float32(_pixel_values)

    # Implement k-means clustering
    _kmeans = KMeans(n_clusters=k, random_state=0)
    _kmeans.fit(_pixel_values)
    _kmeans_labels = _kmeans.labels_
    _kmeans_centers = np.uint8(_kmeans.cluster_centers_)

    # Convert back to original image dimensions
    _segmented_image = _kmeans_labels.reshape(image.shape[:2])
    return _segmented_image, _kmeans_labels, _kmeans_centers

def panel_enhancer(image_car,image_to_enhance):
    # Now, we'll attempt to remove the background. Since the LEDs emit light and have a distinct color,
    # we can use color segmentation to isolate them from the background.
    # Convert the image from BGR to HSV color space
    image_rgb = cv2.cvtColor(image_to_enhance, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    segmented_image, kmeans_labels, kmeans_centers = kmeans_segmentation(image_hsv)

    # Find the brightest cluster (which should correspond to the LED lights)
    brightest_cluster_index = np.argmax(kmeans_centers[:, 2])

    # Create a mask where only the pixels of the brightest cluster are white
    mask = np.where(kmeans_labels.flatten() == brightest_cluster_index, 255, 0).astype('uint8')
    mask = mask.reshape(image_hsv.shape[:2])

    # Define kernel for morphological operations
    kernel = np.ones((3,3), np.uint8)

    # Apply morphological operations to refine the mask
    # Closing (dilation followed by erosion) to close small holes inside the foreground
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Opening (erosion followed by dilation) to remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Initialize minimum area rectangle
    min_area_rect = None
    contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangle for the largest contour which should be encompassing all LEDs
    if contours:
        # Merge all contours into one for the purpose of finding the minimum area bounding box
        all_contours = np.vstack(contours)
        min_area_rect = cv2.minAreaRect(all_contours)

        # Get the box points and draw them
        box = cv2.boxPoints(min_area_rect)
        box = np.intp(box)
        result = cv2.drawContours(image_to_enhance.copy(), [box], 0, (255, 255, 0), 2)

    # Use the four point transform to make the rectangle area a straight rectangle and crop the outside
    warped_image = four_point_transform(image_to_enhance, box)

    # Draw a 4x24 grid on the warped image
    # grid_image = cv2.resize(warped_image.copy(), (0,0), fx=4, fy=4)
    grid_image = warped_image.copy()
    height, width = grid_image.shape[:2]

    # Number of cells in each direction
    num_cells_vertical = 4
    num_cells_horizontal = 24

    # The step size for each cell
    step_size_x = width / num_cells_horizontal
    step_size_y = height / num_cells_vertical

    # Draw the corrected grid
    for x in range(num_cells_horizontal+1):
        cv2.line(grid_image, (int(x * step_size_x), 0), (int(x * step_size_x), height), (255, 0, 0), 1)

    for y in range(num_cells_vertical+1):
        cv2.line(grid_image, (0, int(y * step_size_y)), (width, int(y * step_size_y)), (255, 0, 0), 1)

    # Initialize a binary matrix representing the LED states
    binary_matrix = np.zeros((num_cells_vertical, num_cells_horizontal), dtype=int)

    # Threshold to determine if an LED is on or off
    # We use Otsu's threshold for a bimodal distribution as a heuristic

    warped_image_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(warped_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Process each LED block to determine its state
    for i in range(num_cells_vertical):
        for j in range(num_cells_horizontal):
            # Define the coordinates of the block
            x_start = int(j * step_size_x)
            y_start = int(i * step_size_y)
            x_end = int(x_start + step_size_x)
            y_end = int(y_start + step_size_y)

            # Extract the block from the binary image
            led_block = binary_image[y_start:y_end, x_start:x_end]

            # Determine the state of the LED by the intensity of the block
            if np.mean(led_block) > 125:  # If the block's mean intensity is high, the LED is 'on'
                binary_matrix[i, j] = 1

    # Making a grid to fill for the plot
    gs = gridspec.GridSpec(6, 2)

    # Left side: frmae + box
    ax1 = plt.subplot(gs[0:6, 0])
    ax1.imshow(cv2.cvtColor(image_car, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Camera Object Detection frame {frame_count}')
    ax1.axis('off')

    # Right side: 6 vertically stacked images
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(image_rgb)
    ax2.set_title('Cropped Panel Image')
    ax2.axis('off')

    ax3 = plt.subplot(gs[1, 1])
    ax3.imshow(segmented_image)
    ax3.set_title('Clustered Image')
    ax3.axis('off')

    ax4 = plt.subplot(gs[2, 1])
    ax4.imshow(mask,cmap='gray')
    ax4.set_title('Mask of Brightest Cluster')
    ax4.axis('off')

    ax5 = plt.subplot(gs[3, 1])
    ax5.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax5.set_title('Rotated Bounding Rectangle')
    ax5.axis('off')

    ax6 = plt.subplot(gs[4, 1])
    ax6.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    ax6.set_title('Warped Image with 4x24 Grid')
    ax6.axis('off')

    ax7 = plt.subplot(gs[5, 1])
    ax7.imshow(binary_matrix, cmap='gray', interpolation='nearest')
    ax7.set_title('Binary Matrix Visualization')
    ax7.axis('off')

    plt.show()
    plt.draw()
    plt.pause(0.01)  # Small delay

    # # Optional: Clear the plot
    plt.clf()

    return binary_matrix


#Main Function
plt.ion()
plt.figure(figsize=(16, 8))

model_path = 'VECTOR-detect.pt'
model = YOLO(model_path)
# detecting class 7 with > 0.35 confidence
sign_det_conf = 0.35
panel_cls = 7
target_ratio = [4,6] # target aspect ratio of the panel

#ROS CV bridge:
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('image_processing_node')
bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg,'bgr8')
    except CvBridgeError as e:
        print(e)

    frame_count+=1
    # Run YOLOv8 inference on the frame
    results = model.predict(cv_image, classes=panel_cls, conf=sign_det_conf)
    annotated_frame = results[0].plot()

    #TODO Iterate through detections
    for r in results:
        # Extract bounding box coordinates,
        boxes = r.boxes.xyxy.tolist()
        classes = r.boxes.cls.tolist()
        for box, cls in zip(boxes, classes):
            if cls == panel_cls:
                x1, y1, x2, y2 = map(int, box)
                sign = cv_image[y1:y2, x1:x2]
                matrix = panel_enhancer(annotated_frame,sign)
                print(matrix)

    # Break the loop if 'q' is pressed
    cv2.waitKey(1) & 0xFF == ord("q")

camera_topic = "/usb_cam/image_raw"
image_sub = rospy.Subscriber(camera_topic, Image, image_callback)