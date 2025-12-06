import cv2
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import numpy as np
import torch

from controller import Robot

class LidarPerception:
    def __init__(self, robot: Robot, lidar_name="lidar"):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())

        # Get and enable the lidar
        self.lidar = robot.getDevice(lidar_name)
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Store specs
        self.num_layers = self.lidar.getNumberOfLayers()
        self.num_points = self.lidar.getHorizontalResolution()
        self.fov = self.lidar.getFov()

        print(f"[LIDAR] Layers: {self.num_layers}, Points per layer: {self.num_points}")

    def get_ranges(self):
        """
        Returns the full 2D array of distances:
        shape = [num_layers][num_points]
        """
        ranges = self.lidar.getRangeImage()
        return ranges   # already a Python list of lists

    def get_front_left_right(self):
        """
        Get distances in three directions:
        - front
        - left
        - right
        (for single-layer lidar)
        """
        ranges = self.lidar.getRangeImage()
        layer = ranges[0]  # single layer

        mid = self.num_points // 2
        left = self.num_points // 4
        right = 3 * self.num_points // 4

        return {
            "front": layer[mid],
            "left": layer[left],
            "right": layer[right]
        }

    def get_min_distance(self):
        """
        Shortest obstacle distance detected in front hemisphere.
        """
        ranges = self.lidar.getRangeImage()[0]
        return min(ranges)



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
        # position
        position = self.gps.getValues()  # [x, y, z]
        orientation = self.imu.getRollPitchYaw()  # [roll, pitch, yaw]
        angular_velocity = self.gyro.getValues()  # [wx, wy, wz
        linear_velocity = self.imu.getLinearVelocity()  # [vx, vy, vz]
        return np.array([
            position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2],
            linear_velocity[0], linear_velocity[1], linear_velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2]
        ])

    
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
    
