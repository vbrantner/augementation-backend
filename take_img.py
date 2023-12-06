import cv2
import time

# Initialize the camera
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, it is True. Otherwise, it is False
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Get the current timestamp and use it for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"frame_{timestamp}.jpg"

        # Save the frame
        cv2.imwrite(filename, frame)
        print("took image")

        # Wait for 3 seconds
        time.sleep(3)

except KeyboardInterrupt:
    # Release the camera when interrupted (e.g., pressing Ctrl+C)
    cap.release()
    cv2.destroyAllWindows()
