import numpy
import cv2
import pickle
import glob

# Create arrays to store object points and image points from all images processed
objpoints = []  # 3D points in real world space where chess squares are
imgpoints = []  # 2D points in image plane, determined by CV2

# Chessboard variables
CHESSBOARD_CORNERS_ROWCOUNT = 9
CHESSBOARD_CORNERS_COLCOUNT = 6

# Theoretical object points for the chessboard we're calibrating against
objp = numpy.zeros((CHESSBOARD_CORNERS_ROWCOUNT*CHESSBOARD_CORNERS_COLCOUNT, 3), numpy.float32)
objp[:, :2] = numpy.mgrid[0:CHESSBOARD_CORNERS_ROWCOUNT, 0:CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)

# Set of images or a video taken with the camera for calibration
images = glob.glob('./*.jpg')
imageSize = None  # Determined at runtime

# Loop through images
for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard in the image
    board, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), None)

    if board == True:
        objpoints.append(objp)
        corners_acc = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_acc)

        if not imageSize:
            imageSize = gray.shape[::-1]
    else:
        print(f"Not able to detect a chessboard in image: {iname}")

# Check if images were found
if len(images) < 1:
    print("Calibration unsuccessful. No images of chessboards found.")
    exit()

# Check if we detected any chessboards
if not imageSize:
    print("Calibration unsuccessful. Couldn't detect chessboards in any images.")
    exit()

# Perform the camera calibration
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

# Output matrix and distortion coefficient
print(cameraMatrix)
print(distCoeffs)

# Save calibration data
with open('calibration.pckl', 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)

print('Calibration successful. Calibration file used: calibration.pckl')

