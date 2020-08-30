import setup_path
import airsim
import time
import numpy as np
import atexit


def calc_heading(p1, p2):
    vector = p2[:2] - p1[:2]

    angle = np.arctan2(vector[1], vector[0])
    angle *= (180 / np.pi)
    # print('output:', angle)

    if angle < 0:
        return round(angle + 360, 2)
    else:
        return round(angle, 2)


class Drone:
    def __init__(self, name='Drone1'):
        self.client = airsim.MultirotorClient(port=41451)
        self.name = name
        self.init_client()
        self.current_goal = [0, 0]
        self.current_pose = None
        self.heading = None
        atexit.register(self.disconnect)

    def init_client(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

    def move(self, pos, yaw, offset_x=0, offset_y=0):
        # self.client.enableApiControl(True, self.name)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0]+offset_x, pos[1]+offset_y, pos[2]), airsim.to_quaternion(0, 0, yaw)), True, vehicle_name=self.name)
        # self.client.enableApiControl(False, self.name)

    def set_speed(self, speed):
        '''
        send a command to drone to keep flying to current goal but with new speed
        :arg speed: scalar value between 0-1 (0 is hover in place)
        '''
        height_default = -2
        speed_const = 12
        print(f'Executing action: setting speed to {int(speed*speed_const)}, heading to {self.current_goal}')
        self.client.enableApiControl(True, self.name)
        self.client.moveToPositionAsync(self.current_goal[0], self.current_goal[1], height_default, int(speed*speed_const), vehicle_name=self.name)
        # self.client.enableApiControl(False, self.name)

    def update_pose(self, new_pose):
        '''
        Update drone current position and heading (based on prev pose)
        '''
        if self.current_pose is not None:
            if np.linalg.norm(self.current_pose[:2] - new_pose[:2]) > 2:
                self.heading = calc_heading(self.current_pose, new_pose)
        else:
            self.heading = 0

        self.current_pose = new_pose


    def dist(self, position):
        if self.current_pose is None:
            return False
        else:
            return np.linalg.norm(self.current_pose[:2] - position)

    def get_img2d(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
        response = responses[0]
        # get numpy array
        img1d = response.image_data_float
        # reshape array to 2D array H X W
        try:
            img2d = np.reshape(img1d, (response.height, response.width))
            # print(response.camera_position)
            return img2d
        except:
            return False

    def disconnect(self):
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
        print(f'{self.name} - disconnected')


class Car:
    def __init__(self, name='Car1'):
        self.client = airsim.CarClient(port=41452)
        self.name = name
        self.init_client()
        self.car_controls = airsim.CarControls()
        self.current_pose = None
        self.heading = None
        atexit.register(self.disconnect)

    def init_client(self):
        self.client.confirmConnection()
        # self.client.enableApiControl(True, self.name)

    def move(self, pos, yaw, offset_x=0, offset_y=0):
        # pos Z coordinate is overridden to -1 (or 0, need to test)
        self.client.enableApiControl(True, self.name)
        # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0]+offset_x, pos[1]+offset_y, -1), airsim.to_quaternion(0, 0, yaw)), True, vehicle_name=self.name)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0]+offset_x, pos[1]+offset_y, -1)), True, vehicle_name=self.name)
        time.sleep(0.3)
        self.client.enableApiControl(False, self.name)
        self.client.armDisarm(False, self.name)

    def dist(self, position):
        if self.current_pose is None:
            return False
        else:
            return np.norm(self.current_pose - position)

    def update_pose(self, new_pose):
        '''
        Update drone current position and heading (based on prev pose)
        '''
        if self.current_pose is not None:
            if np.linalg.norm(self.current_pose[:2] - new_pose[:2]) > 1:
                self.heading = calc_heading(self.current_pose, new_pose)
        else:
            self.heading = 0

        self.current_pose = new_pose

    def disconnect(self):
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
        print(f'{self.name} - disconnected')

