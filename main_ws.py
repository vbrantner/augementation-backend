import asyncio
import base64
import json
import math
import pickle

import cv2
import cv2.aruco as aruco
import numpy as np
import websockets
from scipy.spatial.transform import Rotation

with open("calibration.pckl", "rb") as f:
    camera_matrix, distortion_coefficients, rvecs, tvecs = pickle.load(f)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
# set camera resolution to 800x600
cap.set(3, 800)
cap.set(4, 600)


max_distance = 0.1
marker_size = 0.01


def map_to_3d_space(center_x, center_y, image_width, image_height, tvec):
    # Convert pixel coordinates to normalized device coordinates for Three.js
    x_normalized = (center_x / image_width) * 2 - 1
    y_normalized = 1 - (center_y / image_height) * 2  # invert y for Three.js
    # Use the z component of the translation vector for depth
    z_depth = tvec[0][0][2]
    z_remap = -((z_depth / max_distance) * 11 - 1)
    z_scaled = z_depth / max_distance
    return x_normalized, y_normalized, z_remap


async def send_data(websocket, path):
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
            for rvec, tvec in zip(rvecs, tvecs):
                frame = cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    distortion_coefficients,
                    rvec,
                    tvec,
                    marker_size,
                )

            rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
            rotation = Rotation.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()

            # Calculate the center of the marker
            corner_points = corners[0][0]
            center_x = int(sum([point[0] for point in corner_points]) / 4)
            center_y = int(sum([point[1] for point in corner_points]) / 4)

            # Map the 2D coordinates to the Three.js 3D space
            x_normalized, y_normalized, z_depth = map_to_3d_space(
                center_x, center_y, 800, 600, tvecs
            )

            data["quaternion"] = quaternion.tolist()
            data["position"] = [x_normalized, y_normalized, z_depth]
            print(data["position"])

        ret, jpeg = cv2.imencode(".jpg", frame)
        jpeg_as_text = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        data["image"] = jpeg_as_text

        await websocket.send(json.dumps(data))
        await asyncio.sleep(0.02)


start_server = websockets.serve(send_data, "", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
