import cv2
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import numpy as np
import torch
import math

from controller import Robot

class LidarPerception:
    def __init__(self, robot: Robot, lidar_name="lidar"):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())

        # Get and enable the lidar
        self.lidar = self.robot.getDevice(lidar_name)
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Store specs
        self.num_layers = self.lidar.getNumberOfLayers()
        self.num_points = self.lidar.getHorizontalResolution()
        self.fov = self.lidar.getFov()

        print(f"[LIDAR] Layers: {self.num_layers}, Points per layer: {self.num_points}")

    def get_ranges(self):
        """
        Returns 2D array of distances [num_layers][num_points].
        """
        flat_ranges = self.lidar.getRangeImage()  # flat list or generator
        # Convert generator to list if needed
        flat_ranges = list(flat_ranges)
        
        # Reshape manually
        if self.num_layers == 1:
            return [flat_ranges]  # single layer
        else:
            ranges_2d = []
            for i in range(self.num_layers):
                start = i * self.num_points
                end = start + self.num_points
                ranges_2d.append(flat_ranges[start:end])
            return ranges_2d

    def get_min_distance(self):
        """
        Shortest obstacle distance detected in front hemisphere.
        """
        ranges = self.get_ranges()[0]  # single layer
        return min(ranges)

    def get_360_sectors(self, num_sectors=16):
        """
        Returns average distance in each sector around the robot, ignoring inf values.
        """
        ranges = self.get_ranges()  # [num_layers][num_points]
        sector_size = self.num_points // num_sectors
        sector_distances = {}

        default_names = ["front", "front-right", "right", "back-right", 
                         "back", "back-left", "left", "front-left"]
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = start_idx + sector_size
            sector_name = default_names[i % len(default_names)] if i < len(default_names) else f"sector_{i}"

            # Collect all valid points in this sector across all layers
            valid_points = []
            for layer in ranges:
                for d in layer[start_idx:end_idx]:
                    if not math.isinf(d):
                        valid_points.append(d)

            # Average distance, or inf if no points
            if valid_points:
                sector_distances[sector_name] = sum(valid_points) / len(valid_points)
            else:
                sector_distances[sector_name] = float('inf')

        return sector_distances


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

        self.lidar = LidarPerception(self.robot, "lidar")

        print("Perception module initialised successfully")
        

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
    
