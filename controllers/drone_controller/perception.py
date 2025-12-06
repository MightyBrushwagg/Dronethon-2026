import cv2
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import numpy as np
import torch

class Perception():
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep

        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        

        # setup range finders or lidar for obstacle detection

    
    
    def get_state_vector(self):
        """
        Get the current state vector for MPPI: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        """
        # state = np.array([x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz])
        # return state
        pass
    
    def get_obstacle_distances(self):
        """
        Get distance to nearest obstacles in multiple directions.
        """
        # user lidar or range finders (ray casting)
        # distances = {"front": front_distance, "left": left_distance, "right": right_distance, "back": back_distance, "up": up_distance, "down": down_distance}
        # return distances
        pass
        
    
    def get_camera_depth_map(self):
        """
        Get camera depth map.
        """
        # return {
        #     "image": image,
        #     "edges": edges,
        #     "has_obstacle": np.sum(edges) > 1000  # Threshold for obstacle detection
        # }
        pass

    def get_camera_image(self):
        """Capture an image from the camera and return it as an OpenCV BGR array."""
        image = self.camera.getImage()
        print(f"Raw image size: {len(image)} bytes")

        # Convert raw BGRA buffer to a NumPy array
        img = np.frombuffer(image, dtype=np.uint8).reshape((self.height, self.width, 4))

        # Convert BGRA → BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img_bgr
    
    def detect_items(self, image=None):
        """Placeholder for item detection logic."""
        if image is None:
            image = self.get_camera_image()
        # Implement item detection logic here
        # Initialize the ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors
    
