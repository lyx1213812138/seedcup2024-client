import os
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
# import calc

class Env:
    def __init__(self,is_senior=False,seed=100, gui=False):
        self.unwrapped = self
        self.seed = seed
        self.is_senior = is_senior
        self.step_num = 0
        self.max_steps = 100
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 15), dtype=np.float64)
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
        self.total_reward = 0
        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        # neutral_angle = [-49.45849125928217, -155.601209583849, -36, -164.0052115563118, 0, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        # # self.goalx = np.array([0])
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        # self.goalx = np.random.uniform(goal['x'][0], goal['x'][1], 1)
        # self.goaly = np.random.uniform(goal['y'][0], goal['y'][1], 1)
        # self.goalz = np.random.uniform(goal['z'][0], goal['z'][1], 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])

        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]
        # self.obstacle1_position = [0, 0.6, np.random.uniform(0.2, 0.2, 1)]
        # self.obstacle1_position = [np.random.uniform(obstacle['x'][0], obstacle['x'][1], 1) + self.goalx[0], 0.6, np.random.uniform(obstacle['z'][0], obstacle['z'][1], 1)]
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])
        for _ in range(100):
            self.p.stepSimulation()

        return (self.get_observation(), self._get_info())

    def get_observation(self):
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 7)]
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        self.observation = np.hstack((obs_joint_angles, target_position, obstacle1_position, calc.LastPos(obs_joint_angles))).flatten().reshape(1, -1)
        print('obs: ', self.observation, self.observation.shape)
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

    def reward(self):
        reward_score = 0

        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]
            if link_index not in [0, 1]:
                if not self.obstacle_contact:
                    reward_score = - 200
                    print('touch obstacle!')
                self.obstacle_contact = True

        # 计算奖励
        # print('step num: ', self.step_num)
        if self.get_dis() < 0.05 and self.step_num <= self.max_steps:
            self.success_reward = 100
            if self.obstacle_contact:
                if self.is_senior:
                    self.success_reward = 20
                elif not self.is_senior:
                    self.success_reward = 50
                else:
                    return 
            reward_score = 1000
            self.terminated = True
            print("Terminated for reaching target")

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
                    
            reward_score = - 100
            self.terminated = True
            print("Terminated for reaching max steps, dis: ", self.get_dis(), 'total_reward: ', self.total_reward)

        total_dis = self.get_dis() + 0.1 * self.get_idlepos_dis()
        if self.last_dis < 0:
            dis_reward = 0
        else:
            dis_reward = 1000*(self.last_dis - total_dis)
        self.last_dis = total_dis
        # print('traditional dis: ', self.get_dis(), 'idlepos dis: ', self.get_idlepos_dis())
        # print('dis: ', total_dis, 'dis_reward: ', dis_reward)
        reward = reward_score + dis_reward
        self.total_reward += reward

        return reward * (0.5 if self.obstacle_contact else 1)


    def reset_episode(self):
        self.reset()
        return self.step_num, self.get_dis()

    def close(self):
        self.p.disconnect()

    def _get_info(self):
        return {'dis': self.get_dis(), 'score': self.success_reward}

    def _get_reward(self):
        return 

    def get_idlepos_dis(self):
        a = self.get_observation()[0][:6]
        t = self.get_observation()[0][6:9]
        o = self.get_observation()[0][9:]
        r = 0.9
        l1 = 0.865 * r
        l2 = 0.225 * r
        l3 = 0.121  * r
        l4 = 0.168 * r
        a_t_n3 = -60*(t[2]**4)+51.3333*(t[2]**3)-15.65*(t[2]**2)+2.161*t[2]-0.045
        l = np.sqrt(l1 ** 2 + l3 ** 2)
        s = np.sqrt(l ** 2 + l4 ** 2)
        l = s * np.cos(a_t_n3 * math.pi / 2)
        d = np.sqrt(t[0] ** 2 + t[1] ** 2)
        if l + l2 < d: d = l + l2 - 0.0000001
        a1 = math.acos((d**2 + l**2 - l2**2) / (2 * d * l))
        a2 = np.arctan2(t[1], t[0])
        oa = np.arctan2(o[1], o[0])
        a3 = np.arctan2(l3, l1)
        a_t_n = 0
        a_t_n2 = 0
        if o[0] > t[0] - 0.1:
            if oa > a2 + 0.01:
                a_t = a2 - a1 + a3
            else:
                a_t = a1 + a2 + a3
            a_t_n = (a_t / math.pi) / 2%1
            #print(a1, a2, a3)

            # action[4]
            a4 = math.acos((l**2 + l2**2 - d**2) / (2 * l * l2))
            if oa > a2 + 0.01:
                a_t_2 = math.pi/2+a4+a3
            else:
                a_t_2 = math.pi/2-a4+a3
            a_t_n2 = (a_t_2/math.pi + 1)/2%1

        target = np.array([a_t_n, 0.07, 0.40097904, 0.04461199, a_t_n2, 0.5])

        return np.sum((a - target) ** 2)
