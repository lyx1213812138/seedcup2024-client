import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import math
from time import sleep
import env
from ccalc import Calc
import os

calc = Calc()

class BaseAlgorithm(ABC):
    @abstractmethod 
    def get_action(self, observation):
        """
        输入观测值，返回动作
        Args:
            observation: numpy array of shape (1, 12) 包含:
                - 6个关节角度 (归一化到[0,1])
                - 3个目标位置坐标
                - 3个障碍物位置坐标
        Returns:
            action: numpy array of shape (6,) 范围在[-1,1]之间
        """
        pass

now = 0

# 示例：使用PPO预训练模型
class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        path_right = os.path.join(os.path.dirname(__file__), "ppo_eval_logs_42/batch256/right_perfect/best_model")
        path_left = os.path.join(os.path.dirname(__file__), "ppo_eval_logs_42/batch256/left_perfect/best_model")
        print("ppo load path: ", path_right, path_left)
        sleep(1)
        self.model_r = PPO.load(path_right, device="cpu")
        self.model_l = PPO.load(path_left, device="cpu")

    def get_action(self, observation):
        n_angle = observation[0, :6]
        target_position = observation[0][6:9]
        obstacle1_position = observation[0][9:12]
        my_obs = np.hstack((
            observation[0], 
            calc.LastPos(n_angle), 
            calc.WristPos(n_angle),
            calc.jointPos(n_angle),
            calc.idlePos(target_position, obstacle1_position)
        )).reshape(1, -1)
        
        obs_angle = np.arctan2(obstacle1_position[1], obstacle1_position[0])
        target_angle = np.arctan2(target_position[1], target_position[0])
        if obs_angle - target_angle > 0.05:
            action, _ = self.model_r.predict(my_obs)
        else:
            action, _ = self.model_l.predict(my_obs)
        return np.reshape(action, (6, ))
    

class TriangleAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.rest_step = 95

    def to_target(self, now, t):
        x = t - now
        if x > 0.5:
            x = x - 1
        elif x < -0.5:
            x = x + 1
        if self.rest_step > 0 and x > 1:
            return x # / self.rest_step
        else: 
            return x

    def to_target2(self, now_arr, t_arr):
        arr = []
        for now, t in zip(now_arr, t_arr):
            arr.append(self.to_target(now, t))
        return arr

    def get_action(self, observation):
        self.rest_step = self.rest_step - 1
        a = observation[0, :6] 
        t = observation[0, 6:9]
        o = observation[0, 9:]
        action = self.to_target2(a, calc.idlePos(t, o))
    
        if np.max(np.abs(action)) == 0:
            action = np.random.rand(6) - 0.5
        # print(action / (np.max(np.abs(action))))
        # action = np.array(self.to_target2(a, calc.idlePos(t, o)))
        return action / (np.max(np.abs(action)))
        

class RobotAlgorithm():
    def __init__(self):            #DH参数
        self.a=[0 , -0.425 , -0.395 ,0 , 0 , 0]
        self.theta=[0 , 0 , 0 , 0 , 0 , 0]
        self.d=[0.152 , 0 , 0 , 0.102 , 0.102 , 0.100]
        self.alpha=[math.pi/2 , 0, 0, math.pi/2 , -math.pi/2 , 0]


    def transferQue(self,i,j):      #齐次变换方程
        c1 = math.cos(self.theta[j-1])
        s1 = math.sin(self.theta[j-1])
        c2 = math.cos(self.alpha[j-1])
        s2 = math.sin(self.alpha[j-1])
        d = self.d[j-1]
        a = self.a[j-1]
        A=np.array([[c1, -s1*c2, s1*s2, a*c1],
                    [s1, c1*c2, -c1*s2, a*s1], 
                    [0, s2, c2, d], 
                    [0, 0, 0, 1]])
        return A
    

    def PositiveKine(self,n_angle=[1, 1, 1, 1, 1, 1]):   #正运动学求解
        self.theta=n_angle
        T1=self.transferQue(0,1)
        T2=self.transferQue(1,2)
        T3=self.transferQue(2,3)
        T4=self.transferQue(3,4)
        T5=self.transferQue(4,5)
        T6=self.transferQue(5,6)
        T7=T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
        print(T7)
        return T7
    

    def InverseKine(self,T):           #逆运动学分析
        N = np.array([T[0][0], T[1][0], T[2][0]])
        o = np.array([T[0][1], T[1][1], T[2][1]])
        a = np.array([T[0][2], T[1][2], T[2][2]])
        p = np.array([T[0][3], T[1][3], T[2][3]])
        m = self.d[5]*a[1]-p[1]; n=self.d[5]*a[0]-p[0]
        theta1 = round(math.atan2(m,n)-math.atan2(self.d[3],math.sqrt(m**2+n**2-self.d[3]**2)),4)
        theta5 = round(math.acos(a[0]*math.sin(theta1)-a[1]*math.cos(theta1)),4)
        m = N[0]*math.sin(theta1)-N[1]*math.cos(theta1)
        n = o[0]*math.sin(theta1)-o[1]*math.cos(theta1)
        theta6 = round(math.atan2(m,n)-math.atan2(math.sin(theta5),math.sqrt(m**2+n**2-math.sin(theta5)**2)),4)
        m = self.d[4]*(math.sin(theta6)*(N[0]*math.cos(theta1)+N[1]*math.sin(theta1)) + math.cos(theta6)*(o[0]*math.cos(theta1)+o[1]*math.sin(theta1)))
        - self.d[5]*(a[0]*math.cos(theta1)+a[1]*math.sin(theta1)) 
        + p[0]*math.cos(theta1)+p[1]*math.sin(theta1)
        n = p[2]-self.d[0]-a[2]*self.d[5]+self.d[4]*(o[2]*math.cos(theta6)+N[2]*math.sin(theta6))
        theta3 = math.acos((m**2+n**2-self.a[1]**2-self.a[2]**2)/(2*self.a[1]*self.a[2]))
        s2 = ((self.a[2]*math.cos(theta3)+self.a[1])*n-self.a[2]*math.sin(theta3)*m)/(self.a[1]**2+self.a[2]**2+2*self.a[1]*self.a[2]*math.cos(theta3))
        c2 = (m+self.a[2]*math.sin(theta3)*s2)/(self.a[2]*math.cos(theta3)+self.a[1])
        theta2 = math.atan2(s2,c2)
        theta4 = math.atan2(-math.sin(theta6)*(N[0]*math.cos(theta1)+N[1]*math.sin(theta1))
        -math.cos(theta6)*(o[0]*math.cos(theta1)+o[1]*math.sin(theta1)),
        o[2]*math.cos(theta6)+N[2]*math.sin(theta6))-theta2-theta3
        now_theta = np.array([theta1,theta1,theta3,theta4,theta5,theta6])
        print(now_theta)
        return now_theta

        

        