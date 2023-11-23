import cv2
import cv2.aruco as aruco
import numpy as np

# Define the camera matrix and distortion coefficients.
# (These values are camera specific and are usually obtained via camera calibration.
# For demonstration, I'm using dummy values.)
camera_matrix = np.array(
    [
        [1.48020801e03, 0.00000000e00, 9.58896881e02],
        [0.00000000e00, 1.48006176e03, 4.86044787e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)

distortion_coefficients = np.array(
    [6.36269606e-02, -9.40625150e-01, 2.41614783e-03, 2.78816524e-03, 3.56098106e00],
    dtype=np.float32,
)


# Create an ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)


parameters = aruco.DetectorParameters()


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# Open a video capture object for the webcam
cap = cv2.VideoCapture(0)
# set camera to resolution to 800x600
cap.set(3, 800)
cap.set(4, 600)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    # If markers are detected
    if len(corners) > 0:
        # Draw the markers on the frame
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker and return the values rvec and tvec
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, distortion_coefficients
        )
        for rvec, tvec in zip(rvecs, tvecs):
            aruco.drawAxis(
                frame, camera_matrix, distortion_coefficients, rvec, tvec, 0.03
            )

        rotation_matrix = cv2.Rodrigues(rvecs[0])[0]
        euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

        print(euler_angles)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break out of the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
