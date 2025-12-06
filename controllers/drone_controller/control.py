import math
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
target_altitude = 1.0  # The target altitude. Don't go lower!

def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero."""
    return 1 if x > 0 else (-1 if x < 0 else 0)

class Control(Robot):
    def __init__(self, timestep):
        super().__init__()
        self.fl_motor = Robot.getDevice("front left propeller")
        self.fr_motor = Robot.getDevice("front right propeller")
        self.rl_motor = Robot.getDevice("rear left propeller")
        self.rr_motor = Robot.getDevice("rear right propeller")
        self.motors = [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]
        
        self.timestep = timestep
        
        # camera control
        self.camera_roll_motor = self.robot.getDevice("camera roll")
        self.camera_pitch_motor = self.robot.getDevice("camera pitch")
        self.camera_roll_motor.setPosition(0)
        self.camera_pitch_motor.setPosition(0)
        self.camera_roll_motor.enable(self.timestep)
        self.camera_pitch_motor.enable(self.timestep)
        
        # initalising motors
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
            motor.enable(self.timestep)
    
    def process_signal(self, vel):
        """receive 4 velocities for each propeller"""
        for motor, v in zip(self.motor, vel):
            motor.setPosition(float('inf'))
            motor.setVelocity(v)
        
        
