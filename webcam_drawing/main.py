import cv2
import numpy as np

W, H = 640, 480  # Frame width & height
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

# HSV ranges for tracking colors
color_ranges = [
    [5, 107, 0, 19, 255, 255],    # Orange
    [133, 56, 0, 159, 156, 255],  # Purple
    [57, 76, 0, 100, 255, 255],   # Green
    [90, 48, 0, 118, 255, 255]    # Blue
]

# Corresponding BGR values for drawing
draw_colors = [
    [51, 153, 255],   # Orange
    [255, 0, 255],    # Purple
    [0, 255, 0],      # Green
    [255, 0, 0]       # Blue
]

points = []  # [x, y, color_id]

def find_color(img, ranges, bgr_vals):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_pts = []
    
    for idx, rng in enumerate(ranges):
        lower, upper = np.array(rng[:3]), np.array(rng[3:])
        mask = cv2.inRange(hsv, lower, upper)
        cx, cy = get_contour_center(mask)
        
        # Draw detection circle
        if cx != 0 and cy != 0:
            cv2.circle(result, (cx, cy), 15, bgr_vals[idx], cv2.FILLED)
            new_pts.append([cx, cy, idx])
    
    return new_pts

def get_contour_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cx, cy = 0, 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + w // 2, y + h // 2  # Center of bounding box
            break  # Take the first valid contour
    
    return cx, cy

def draw_canvas(pts, bgr_vals):
    for x, y, color_id in pts:
        cv2.circle(result, (x, y), 10, bgr_vals[color_id], cv2.FILLED)

print("Camera initialized. Press 'q' to quit, 'c' to clear canvas.")

while True:
    ret, frame = cap.read()
    
    # Check if frame was read successfully
    if not ret:
        print("Error: Failed to read frame from camera")
        break
    
    result = frame.copy()
    new_pts = find_color(frame, color_ranges, draw_colors)
    
    if new_pts:
        points.extend(new_pts)
    
    if points:
        draw_canvas(points, draw_colors)
    
    cv2.imshow("Result", result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points.clear()  # Clear the canvas
        print("Canvas cleared")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")