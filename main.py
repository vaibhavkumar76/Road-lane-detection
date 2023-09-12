import cv2
import numpy as np

# Function to apply Canny edge detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return canny

# Function to mask the region of interest
def region_of_interest(image):
    height, width = image.shape
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to display the detected lane lines
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

# Main function for lane detection
def detect_lanes(image):
    canny_image = canny(image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(image, lines)
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

# Read the video clip
cap = cv2.VideoCapture('C:\\Users\\vaibh\\Downloads\\testing_video5.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_detected_frame = detect_lanes(frame)
    cv2.imshow('Lane Detection', lane_detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()