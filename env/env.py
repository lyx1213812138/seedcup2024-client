import os
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from ccalc import Calc
from .reward import reward
from .utils import predict_pos, relative_dir, next_tar_step
import random
from team_algorithm import MyCustomAlgorithm

calc = Calc()

class Env(gym.Env):
    def __init__(self,is_senior=False,seed=423, gui=False, pos='all'):
        super().__init__()
        self.reward = reward
        self.pos = pos
        # self.unwrapped = self
        self.seed = seed
        self.is_senior = is_senior
        self.step_num = 0
        self.max_steps = 200
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.random_velocity = np.random.uniform(-0.02, 0.02, 2)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        # XXX observation space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 47), dtype=np.float64)
        self.metadata = {'render.modes': []}

        self.target_step = 94 # 未来目标位置的步数
        self.prior_plan = MyCustomAlgorithm()

        self.init_env()

    def init_env(self):
        # np.random.seed(self.seed)  
        self.fr5 = self.p.loadURDF("fr5_description/urdf/fr5v6.urdf", useFixedBase=True, basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), flags=p.URDF_USE_SELF_COLLISION)
        self.table = self.p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        collision_target_id = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_target_id, basePosition=[0.5, 0.8, 2])
        collision_obstacle_id = self.p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
        self.obstacle1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_obstacle_id, basePosition=[0.5, 0.5, 2])
        self.reset()

    # def reset(self, goal={'x': (-0.2, 0.2), 'y': (0.8, 0.9), 'z': (0.1, 0.3)}, obstacle={'x': (-0.2, 0.2), 'z': (0.1, 0.3)}, seed, options):
    def reset(self, seed = None, options = None):
        self.step_num = 0
        self.success_reward = 0
        self.terminated = False
        self.obstacle_contact = False   
        self.last_dis = -1
        self.last_obs_dis = -1
        self.total_reward = 0

        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]

        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        # TEST
        # self.target_position = predict_pos(self.target_position, self.random_velocity, 120)
        # self.step_num = 120
        # idle_angle = calc.idlePos(self.target_position, self.obstacle1_position)
        # neutral_angle = [(x * 2 - 1) * math.pi + random.uniform(-0.1, 0.1) for x in idle_angle] + [0, 0]
        # print(neutral_angle)
        
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        dir = relative_dir(
            {'x': self.obstacle1_position[0][0], 'y': self.obstacle1_position[1]}, 
            {'x': self.target_position[0], 'y': self.target_position[1]}
        )
        if self.pos == 'right':
            if dir == 'left':
                self.obstacle1_position[0] = self.target_position[0] - 0.1
        elif self.pos == 'center':
            if dir != 'center':
                self.obstacle1_position[0] = self.target_position[0]
        elif self.pos == 'left':
            if dir == 'right':
                self.obstacle1_position[0] = self.target_position[0]
        # print("1118", self.obstacle1_position, self.target_position)

        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])

        # 设置目标朝x z平面赋予随机速度
        self.random_velocity = np.random.uniform(-0.02, 0.02, 2)
        self.p.resetBaseVelocity(self.target, linearVelocity=[self.random_velocity[0], 0, self.random_velocity[1]])

        for _ in range(100):
            self.p.stepSimulation()
            # self.reward(self)
            # if self.terminated:
            #     print('reset')
            #     return self.reset()
        

        return (self.get_observation(), self._get_info())

    def get_observation(self):
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 7)]
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
    
        future_tar_pos = predict_pos(target_position, self.random_velocity, max(0, self.target_step-self.step_num))
        end_tar_pos = predict_pos(target_position, self.random_velocity, max(0, self.max_steps-self.step_num))
        dir_future = relative_dir(
            {'x': future_tar_pos[0], 'y': future_tar_pos[1]},
            {'x': obstacle1_position[0], 'y': obstacle1_position[1]}, 
            True
        )
        dir_end = relative_dir(
            {'x': end_tar_pos[0], 'y': end_tar_pos[1]},
            {'x': obstacle1_position[0], 'y': obstacle1_position[1]}, 
            True
        )
        # XXX observation        
        self.observation = np.hstack((
            obs_joint_angles, # [0:6] # 机械臂角度
            future_tar_pos, obstacle1_position, # [6:12] # 环境物体位置
            calc.LastPos(obs_joint_angles),  # [12:15] # 各关节位置
            calc.WristPos(obs_joint_angles), # [15:18]
            calc.jointPos(obs_joint_angles), # [18:36]
            calc.idlePos(future_tar_pos, obstacle1_position), #[36:42] # 预测机械臂角度
            [self.random_velocity[0], 0, self.random_velocity[1]], # [42:45] # 目标物体速度
            [dir_future, dir_end] # [45:47] # 未来目标位置和最终目标位置的方向
        )).flatten().reshape(1, -1)
        # print('obs: ', self.observation, self.observation.shape)
        
        return self.observation

    def step(self, action):
        if self.terminated:
            return self.reset_episode()
        
        self.step_num += 1

        # TEST
        if self.step_num <= self.target_step:
            action = self.prior_plan.get_action(self.get_observation())

        joint_angles = [self.p.getJointState(self.fr5, i)[0] for i in range(1, 7)]
        action = np.clip(action, -1, 1)
        fr5_joint_angles = np.array(joint_angles) + (np.array(action[:6]) / 180 * np.pi)
        gripper = np.array([0, 0])
        angle_now = np.hstack([fr5_joint_angles, gripper])
        reward = self.reward(self)
        # TEST
        if self.step_num <= self.target_step:
            reward = 0
            
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=angle_now)

        for _ in range(20):
            self.p.stepSimulation()

        # 检查目标位置并反向速度
        target_position = self.p.getBasePositionAndOrientation(self.target)[0]
        if target_position[0] > 0.5 or target_position[0] < -0.5:
            self.p.resetBaseVelocity(self.target, linearVelocity=[-self.random_velocity[0], 0, self.random_velocity[1]])
        if target_position[2] > 0.5 or target_position[2] < 0.1:
            self.p.resetBaseVelocity(self.target, linearVelocity=[self.random_velocity[0], 0, -self.random_velocity[1]])

        # TEST 查看100步时的target_position
        # if self.step_num == self.target_step:
        #     tp = np.array(target_position)
        #     v = [self.random_velocity[0], 0, self.random_velocity[1]]
        #     if np.linalg.norm(predict_pos(self.target_position, self.random_velocity, self.target_step) - tp) > 0.03:
        #         print('target position error: ', self.predict_pos, tp)
        #         print('start :', self.target_position, 'velocity: ', v)
        #         exit(1)

        return (self.get_observation(), reward, self.terminated, self.terminated, self._get_info())
        # return self.observation


    # step!=0: 抓手位置和step步(一共走了)的目标位置的距离, 如果 now step > step 则按照当前目标位置
    def get_dis(self, if_future=False): 
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        if not if_future:
            return np.linalg.norm(gripper_centre_pos - target_position)
        return np.linalg.norm(gripper_centre_pos 
            - predict_pos(target_position, self.random_velocity, next_tar_step(self.step_num, self.target_step, self.max_steps)))

    def get_obs_dis(self):
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        return np.linalg.norm(gripper_centre_pos - target_position)


    def reset_episode(self):
        self.reset()
        return self.step_num, self.get_dis()

    def close(self):
        self.p.disconnect()

    def _get_info(self):
        return {'dis': self.get_dis(), 'score': self.success_reward}

    def _get_reward(self):
        return 

    
