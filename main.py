import cv2
import numpy as np
import math
import time

class ColorTracker:
    def __init__(self, lower_color, upper_color):
        self.lower_color = np.array(lower_color, dtype=np.uint8)
        self.upper_color = np.array(upper_color, dtype=np.uint8)
        self.tracked_object = None
        self.tracking_threshold = 0.5

    def track_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            detection = np.array([[x / frame.shape[1], y / frame.shape[0], w / frame.shape[1], h / frame.shape[0]]])
            return detection

        return []

def analyze_camera_output(duration):
    # Set camera resolution to a square format
    width = 1280
    height = 720

    # Open camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Failed to open camera.")
        return None

    # Color tracker for black targets
    color_tracker = ColorTracker(lower_color=(0, 0, 0), upper_color=(30, 30, 30))

    # Calculate square size and grid dimensions
    square_size = 720 // 10
    grid_start_x = (width - 720) // 2
    grid_end_x = grid_start_x + 720
    grid_start_y = (height - 720) // 2
    grid_end_y = grid_start_y + 720

    # Variables for tracking black targets
    detections = []

    # Start time
    start_time = time.time()

    while time.time() - start_time < duration:
        # Read frame from camera
        ret, frame = camera.read()

        # Create a blank frame for the grid
        grid_frame = np.zeros((height, width, 3), np.uint8)

        # Extract the middle section of the frame
        middle_frame = frame[grid_start_y:grid_end_y, grid_start_x:grid_end_x]

        # Track color in the middle frame
        detection = color_tracker.track_color(middle_frame)

        # Process detection
        if len(detection) > 0:
            x, y, _, _ = detection[0]
            x = int(x * middle_frame.shape[1])
            y = int(y * middle_frame.shape[0])
            cv2.circle(middle_frame, (x, y), 5, (0, 255, 0), -1)

            # Add detection to the list
            detections.append((x, y))

        # Draw grid lines on the grid frame
        for i in range(1, 10):
            cv2.line(grid_frame, (grid_start_x + i * square_size, grid_start_y),
                     (grid_start_x + i * square_size, grid_end_y), (0, 0, 255), 1)
            cv2.line(grid_frame, (grid_start_x, grid_start_y + i * square_size),
                     (grid_end_x, grid_start_y + i * square_size), (0, 0, 255), 1)

        # Merge the middle frame and grid frame
        frame[grid_start_y:grid_end_y, grid_start_x:grid_end_x] = middle_frame
        frame = cv2.addWeighted(frame, 1, grid_frame, 0.5, 0)

        # Show frame with detections and grid
        cv2.imshow("Color Tracking", frame)

        # Check for exit key
        if cv2.waitKey(1) == ord('q'):
            break

    # Release camera
    camera.release()

    # Calculate the average location of black targets
    if len(detections) > 0:
        avg_x = sum(x for x, _ in detections) / len(detections)
        avg_y = sum(y for _, y in detections) / len(detections)
        avg_location = (avg_x, avg_y)

        # Convert average location to square number
        x_square = math.floor(avg_x / square_size)
        y_square = math.floor(avg_y / square_size)
        square_number = y_square * 10 + x_square

        # Draw a rectangle around the correct square
        cv2.rectangle(frame, (grid_start_x + x_square * square_size, grid_start_y + y_square * square_size),
                      (grid_start_x + (x_square + 1) * square_size, grid_start_y + (y_square + 1) * square_size),
                      (0, 255, 0), 2)

        return square_number, frame
    else:
        return None, frame


def main():
    # Analyze camera output for 5 seconds
    duration = 5.0
    square_number, frame = analyze_camera_output(duration)

    if square_number is not None:
        print("Average Location Square Number:", square_number)

        # Display the frame with grid and rectangle
        cv2.imshow("Grid Visualization", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No black targets detected.")

if __name__ == "__main__":
    main()
