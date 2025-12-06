class Control():
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep
        self.camera_roll_motor = self.robot.getDevice("camera roll")
        self.camera_pitch_motor = self.robot.getDevice("camera pitch")
        self.camera_roll_motor.setPosition(0)
        self.camera_pitch_motor.setPosition(0)
        self.camera_roll_motor.enable(self.timestep)
        self.camera_pitch_motor.enable(self.timestep)

        self.front_left_motor = self.robot.getDevice("front left propeller")
        self.front_right_motor = self.robot.getDevice("front right propeller")
        self.rear_left_motor = self.robot.getDevice("rear left propeller")
        self.rear_right_motor = self.robot.getDevice("rear right propeller")

        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))

        self.front_left_motor.setVelocity(1.0)
        self.front_right_motor.setVelocity(1.0)
        self.rear_left_motor.setVelocity(1.0)
        self.rear_right_motor.setVelocity(1.0)

        self.front_left_motor.enable(self.timestep)
        self.front_right_motor.enable(self.timestep)
        self.rear_left_motor.enable(self.timestep)
        self.rear_right_motor.enable(self.timestep)