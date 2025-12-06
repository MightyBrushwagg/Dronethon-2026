import cv2
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import numpy as np
import torch
import math

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
        
        # For velocity calculation from position
        self.prev_position = None
        self.prev_time = None

        # setup range finders or lidar for obstacle detection

    
    
    def get_state_vector(self):
        """
        Get the current state vector for MPPI: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        """
        # position
        position = np.array(self.gps.getValues())  # [x, y, z]
        orientation = self.imu.getRollPitchYaw()  # [roll, pitch, yaw]
        angular_velocity = self.gyro.getValues()  # [wx, wy, wz]
        
        # Calculate linear velocity from position difference
        current_time = self.robot.getTime()
        if self.prev_position is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                linear_velocity = (position - self.prev_position) / dt
            else:
                linear_velocity = np.array([0.0, 0.0, 0.0])
        else:
            # First call, no previous data
            linear_velocity = np.array([0.0, 0.0, 0.0])
        
        # Update previous values for next call
        self.prev_position = position
        self.prev_time = current_time
        
        return np.array([
            position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2],
            linear_velocity[0], linear_velocity[1], linear_velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2]
        ])

    
    def get_obstacle_distances(self):
        """
        Get distance to nearest obstacles in multiple directions using lidar.
        
        Returns:
            dict with keys: "front", "left", "right", "back", "up", "down"
            Values are minimum distances in meters, or inf if no obstacle detected
        """
        distances = {
            "front": float('inf'),
            "left": float('inf'),
            "right": float('inf'),
            "back": float('inf'),
            "up": float('inf'),
            "down": float('inf')
        }
        
        try:
            lidar_ranges = self.lidar.get_ranges()
            if not lidar_ranges or len(lidar_ranges) == 0:
                return distances
            
            # Get ranges from bottom layer (closest to obstacles)
            ranges = lidar_ranges[0] if isinstance(lidar_ranges[0], list) else lidar_ranges
            num_points = len(ranges)
            
            if num_points == 0:
                return distances
            
            # Divide lidar scan into sectors
            sector_size = num_points // 4  # 4 sectors: front, right, back, left
            
            # Front sector (0 to sector_size)
            front_ranges = [r for r in ranges[0:sector_size] if not math.isinf(r) and r > 0]
            if front_ranges:
                distances["front"] = min(front_ranges)
            
            # Right sector (sector_size to 2*sector_size)
            right_ranges = [r for r in ranges[sector_size:2*sector_size] if not math.isinf(r) and r > 0]
            if right_ranges:
                distances["right"] = min(right_ranges)
            
            # Back sector (2*sector_size to 3*sector_size)
            back_ranges = [r for r in ranges[2*sector_size:3*sector_size] if not math.isinf(r) and r > 0]
            if back_ranges:
                distances["back"] = min(back_ranges)
            
            # Left sector (3*sector_size to end)
            left_ranges = [r for r in ranges[3*sector_size:] if not math.isinf(r) and r > 0]
            if left_ranges:
                distances["left"] = min(left_ranges)
            
            # Check vertical obstacles (use different layers)
            if len(lidar_ranges) > 1:
                # Top layer for "up" obstacles
                top_ranges = lidar_ranges[-1] if isinstance(lidar_ranges[-1], list) else []
                if top_ranges:
                    up_ranges = [r for r in top_ranges if not math.isinf(r) and r > 0]
                    if up_ranges:
                        distances["up"] = min(up_ranges)
                
                # Bottom layer for "down" obstacles (ground)
                bottom_ranges = lidar_ranges[0] if isinstance(lidar_ranges[0], list) else []
                if bottom_ranges:
                    down_ranges = [r for r in bottom_ranges if not math.isinf(r) and r > 0]
                    if down_ranges:
                        distances["down"] = min(down_ranges)
            else:
                # Single layer - use it for down (ground)
                down_ranges = [r for r in ranges if not math.isinf(r) and r > 0]
                if down_ranges:
                    distances["down"] = min(down_ranges)
        
        except Exception as e:
            # If obstacle detection fails, return default distances
            pass
        
        return distances
        
    
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
    
