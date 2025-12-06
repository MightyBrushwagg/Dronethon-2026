import math
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor

def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero."""
    return 1 if x > 0 else (-1 if x < 0 else 0)

class Control():
    def __init__(self, robot, timestep):
        self.robot = robot
        
        # motors
        self.fl_motor = self.robot.getDevice("front left propeller")
        self.fr_motor = self.robot.getDevice("front right propeller")
        self.rl_motor = self.robot.getDevice("rear left propeller")
        self.rr_motor = self.robot.getDevice("rear right propeller")
        self.motors = [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]
        
        self.timestep = timestep
        
        # camera control
        self.camera_roll_motor = robot.getDevice("camera roll")
        self.camera_pitch_motor = robot.getDevice("camera pitch")
        self.camera_roll_motor.setPosition(0)
        self.camera_pitch_motor.setPosition(0)
        self.camera_roll_motor.enableForceFeedback(self.timestep)
        self.camera_pitch_motor.enableForceFeedback(self.timestep)
        
        # sensors
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(timestep)
            
        # initalising motors
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
            
        # PID gains TO TUNE!
        self.k_roll = {'p': 8.0, 'i': 0.0, 'd': 0.1}
        self.k_pitch = {'p': 10.0, 'i': 5.0, 'd': 1}
        self.k_yaw = {'p': 0.5, 'i': 0.0, 'd': 0.0}
        
        # integral accumulation
        self.integral = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
        self.prev_error = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
    
        self.k_vertical_thrust = 68.5;  # with this thrust, the drone lifts.
        self.k_vertical_offset = 0.6; 
        
        
    def process_signal(self, vel):
        print("these are the vels: ")
        """receive 4 velocities for each propeller"""
        signs = [1, -1, -1, 1]
        for motor, v, s in zip(self.motors, vel, signs):
            # motor.setPosition(float('inf'))
            clamped = max(-576, min(576, v*s))  # respect motor limits
            print(v*s, end=" ")
            motor.setVelocity(clamped)
        print()
            
    def _pid(self, error, axis, gains, dt):
        self.integral[axis] += error * dt
        derivative = (error - self.prev_error[axis]) / dt if dt > 0 else 0
        self.prev_error[axis] = error
        return gains['p'] * error + gains['i'] * self.integral[axis] + gains['d'] * derivative
            
    def stabilise(self, dt, vel=None):
        if vel is None:
            vel = [self.k_vertical_thrust] * 4 # hover by default
            
        # read sensors
        roll, pitch, _ = self.imu.getRollPitchYaw()
        altitude = self.gps.getValues()[2]
        roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
        print(f"altitude: {altitude}, roll: {roll}, pitch: {pitch}")
        
        # stabilise camera by actuating the camera motors according to the gyro feedback.
        self.camera_roll_motor.setPosition(-0.115 * roll_vel)
        self.camera_pitch_motor.setPosition(-0.1 * pitch_vel)
        
        # pid for stabilisation
        roll_corr = self._pid(max(-1, min(1, roll)), 'roll', self.k_roll, dt)
        pitch_corr = self._pid(max(-1, min(1, pitch)), 'pitch', self.k_pitch, dt)
        yaw_corr = self._pid(yaw_vel, 'yaw', self.k_yaw, dt)
        
        corrections = [
        -roll_corr + pitch_corr - yaw_corr,   # front left
        +roll_corr + pitch_corr + yaw_corr,   # front right
        -roll_corr - pitch_corr + yaw_corr,   # rear left
        +roll_corr - pitch_corr - yaw_corr,   # rear right
        ]
        
        
        return [vel[i] + corrections[i] for i in range(4)]
        
        
    def STOP(self):
        """emergency STOP ALL MOTORS"""
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
        