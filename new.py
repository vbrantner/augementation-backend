import asyncio
import base64
import json
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import websockets
from scipy.spatial.transform import Rotation

# Load calibration data
with open("calibration.pckl", "rb") as f:
    camera_matrix, distortion_coefficients, rvecs, tvecs = pickle.load(f)

# Aruco marker setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Video capture setup
video_device = "/dev/video2"
cap = cv2.VideoCapture(video_device)
cap.set(3, 800)  # Width
cap.set(4, 600)  # Height

# Constants
max_distance = 0.1
marker_size = 0.01

# Choose the index of the corner to use (0, 1, 2, or 3)
chosen_corner_index = 0


def map_to_3d_space(center_x, center_y, image_width, image_height, tvec):
    # Convert pixel coordinates to normalized device coordinates for Three.js
    x_normalized = (center_x / image_width) * 2 - 1
    y_normalized = 1 - (center_y / image_height) * 2  # invert y for Three.js
    z_depth = tvec[0][0][2]
    z_remap = -((z_depth / max_distance) * 11)
    return x_normalized, y_normalized, z_remap


async def send_data(websocket, path):
    try:
        print("Client connected")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )

            data = {}

            if len(corners) > 0:
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, distortion_coefficients
                )

                # Choose the index of the corner to use (0, 1, 2, or 3)
                chosen_corner_index = 0

                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    # Get the coordinates of the chosen corner
                    corner_coord = corners[i][0][chosen_corner_index]

                    # Compute the 3D position of the corner in the marker's coordinate system
                    corner_3d = np.array(
                        [
                            [
                                corner_coord[0] - marker_size / 2,
                                corner_coord[1] - marker_size / 2,
                                0,
                            ]
                        ]
                    )

                    # Adjust the translation vector
                    tvec_adjusted = tvec + corner_3d

                    # Draw the axes using the adjusted translation vector
                    frame = cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        distortion_coefficients,
                        rvec,
                        tvec_adjusted,
                        marker_size,
                    )

                rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
                rotation = Rotation.from_matrix(rotation_matrix)
                quaternion = rotation.as_quat()

                corner_points = corners[0][0]
                center_x = int(sum([point[0] for point in corner_points]) / 4)
                center_y = int(sum([point[1] for point in corner_points]) / 4)

                x_normalized, y_normalized, z_depth = map_to_3d_space(
                    center_x, center_y, 800, 600, tvecs
                )

                data["quaternion"] = quaternion.tolist()
                data["position"] = [x_normalized, y_normalized, z_depth]

            ret, jpeg = cv2.imencode(".jpg", frame)
            jpeg_as_text = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            data["image"] = jpeg_as_text

            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.02)
    except Exception as e:
        print("Error:", e)
    finally:
        cap.release()


async def start_server():
    server = await websockets.serve(send_data, "", 8765)
    await server.wait_closed()


# Run the server
asyncio.run(start_server())
