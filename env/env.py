import os
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from ccalc import Calc

calc = Calc()

class Env:
    def __init__(self,is_senior=False,seed=123, gui=False, pos='left'):
        self.pos = pos
        self.unwrapped = self
        self.seed = seed
        self.is_senior = is_senior
        self.step_num = 0
        self.max_steps = 100
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        # XXX observation space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 42), dtype=np.float64)
        self.metadata = {'render.modes': []}

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
        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]

        obs_angle = np.arctan2(self.obstacle1_position[1], self.obstacle1_position[0][0])
        target_angle = np.arctan2(self.target_position[1], self.target_position[0])
        if self.pos == 'right':
            if obs_angle - target_angle < 0:
                self.obstacle1_position[0] = self.target_position[0] - 0.1
        elif self.pos == 'center':
            if abs(obs_angle - target_angle) > 0.2:
                self.obstacle1_position[0] = self.target_position[0]
        elif self.pos == 'left':
            if obs_angle - target_angle > 0:
                self.obstacle1_position[0] = self.target_position[0]
        # print("1118", self.obstacle1_position, self.target_position)

        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])
        for _ in range(100):
            self.p.stepSimulation()

        return (self.get_observation(), self._get_info())

    def get_observation(self):
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 7)]
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])

        # XXX observation        
        self.observation = np.hstack((
            obs_joint_angles, # [0:6]
            target_position, obstacle1_position, # [6:12]
            calc.LastPos(obs_joint_angles),  # [12:15]
            calc.WristPos(obs_joint_angles), # [15:18]
            calc.jointPos(obs_joint_angles), # [18:36]
            calc.idlePos(target_position, obstacle1_position) #[36:42]
        )).flatten().reshape(1, -1)
        # print('obs: ', self.observation, self.observation.shape)
        
        return self.observation

    def step(self, action):
        if self.terminated:
            return self.reset_episode()
        
        self.step_num += 1
        joint_angles = [self.p.getJointState(self.fr5, i)[0] for i in range(1, 7)]
        action = np.clip(action, -1, 1)
        fr5_joint_angles = np.array(joint_angles) + (np.array(action[:6]) / 180 * np.pi)
        gripper = np.array([0, 0])
        angle_now = np.hstack([fr5_joint_angles, gripper])
        reward = self.reward()
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=angle_now)

        for _ in range(20):
            self.p.stepSimulation()

        return (self.get_observation(), reward, self.terminated, self.terminated, self._get_info())
        # return self.observation

    def get_dis(self):
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        return np.linalg.norm(gripper_centre_pos - target_position)

    def get_obs_dis(self):
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        return np.linalg.norm(gripper_centre_pos - target_position)


    def reward(self):
        reward = 0

        # XXX 计算距离reward
        total_dis = self.get_dis()
        obs = self.get_observation()
        total_dis = np.linalg.norm(obs[0][0:6] - obs[0][36:42]) + self.get_dis()
        if self.last_dis < 0:
            reward = 0
        else:
            reward = 1000*(self.last_dis - total_dis)
        self.last_dis = total_dis

        # XXX 接近障碍物
        dis = calc.near_obs(self.get_observation(), self.get_obs_dis())
        if self.last_obs_dis > 0:
            if dis < 0.2:
                reward -= 500*(self.last_obs_dis - dis)
            elif dis < 0.25:
                reward -= 200*(self.last_obs_dis - dis)
        self.last_obs_dis = dis

        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]
            if link_index not in [0, 1]:
                if not self.obstacle_contact:
                    # XXX contact
                    reward = -300
                    self.obstacle_contact = True
                # reward = -10

        # 计算结束
        if self.get_dis() < 0.05 and self.step_num <= self.max_steps:
            self.success_reward = 100
            if self.obstacle_contact:
                if self.is_senior:
                    self.success_reward = 20
                elif not self.is_senior:
                    self.success_reward = 50
                else:
                    return 
                
            # XXX reach target
            if self.obstacle_contact:
                reward = 300
            else:
                reward = 1000
            self.terminated = True
            print("Terminated for reaching target", 
                  'touch :' , self.obstacle_contact,
                  '\n\ttotal_reward: ', self.total_reward)

        elif self.step_num >= self.max_steps:
            distance = self.get_dis()
            if 0.05 <= distance <= 0.2:
                self.success_reward = 100 * (1 - ((distance - 0.05) / 0.15))
            else:
                self.success_reward = 0
            if self.obstacle_contact:
                if self.is_senior:
                    self.success_reward *= 0.2 
                elif not self.is_senior:
                    self.success_reward *= 0.5
                    
            self.terminated = True
            # XXX reach max steps
            reward = self.success_reward
            # if self.success_reward < 30:
            #     reward = -300
            print("Terminated for reaching max steps, dis: ", self.get_dis(), 
                  'obs_dis: ', self.get_obs_dis(),
                  'touch :' , self.obstacle_contact,
                #   'pos dis: ', total_dis,
                  '\n\ttotal_reward: ', self.total_reward)

        if self.terminated:
            reward = self.success_reward - 50
            print('success_reward', self.success_reward)
        else:
            reward = 0
        self.total_reward += reward

        # XXX calc reward
        return reward


    def reset_episode(self):
        self.reset()
        return self.step_num, self.get_dis()

    def close(self):
        self.p.disconnect()

    def _get_info(self):
        return {'dis': self.get_dis(), 'score': self.success_reward}

    def _get_reward(self):
        return 

    
