import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import math
from time import sleep
import os

#DH参数
a=[0 , -0.425 , -0.395 ,0 , 0 , 0]
theta=[0 , 0 , 0 , 0 , 0 , 0]
d=[0.152 , 0 , 0 , 0.102 , 0.102 , 0.100]
alpha=[math.pi/2 , 0, 0, math.pi/2 , -math.pi/2 , 0]

class Calc:
    def __init__(self):
        pass
    def transferQue(self,i,j):      #齐次变换方程
        global a, theta, d, alpha
        c1 = math.cos(theta[j-1])
        s1 = math.sin(theta[j-1])
        c2 = math.cos(alpha[j-1])
        s2 = math.sin(alpha[j-1])
        dd = d[j-1]
        aa = a[j-1]
        A=np.array([[c1, -s1*c2, s1*s2, aa*c1],
                    [s1, c1*c2, -c1*s2, aa*s1], 
                    [0, s2, c2, dd], 
                    [0, 0, 0, 1]])
        return A


    def PositiveKine(self,n_angle=[1, 1, 1, 1, 1, 1], i=6):   #正运动学求解
        global a, theta, d, alpha
        theta = math.pi*(2*n_angle-1)
        # T1=transferQue(0,1)
        # T2=transferQue(1,2)
        # T3=transferQue(2,3)
        # T4=transferQue(3,4)
        # T5=transferQue(4,5)
        # T6=transferQue(5,6)
        # T7=T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
        # print(T7)
        T = [0]*7
        T[1] = self.transferQue(0,1)
        T[2] = T[1].dot(self.transferQue(1,2))
        T[3] = T[2].dot(self.transferQue(2,3))
        T[4] = T[3].dot(self.transferQue(3,4))
        T[5] = T[4].dot(self.transferQue(4,5))
        T[6] = T[5].dot(self.transferQue(5,6))
        return T[i]

    def LastPos(self,n_angle):
        a = self.PositiveKine(n_angle)
        return np.array([-a[0][3], -a[1][3], a[2][3]])

    def WristPos(self,n_angle):
        global a, theta, d, alpha
        T=self.PositiveKine(n_angle)
        # print(T[0][3],T[1][3],T[2][3])
        pw=np.array([-(T[0][3]-d[5]*T[0][2]), -(T[1][3]-d[5]*T[1][2]), T[2][3]-d[5]*T[2][2]])
        return pw

    def jointPos(self,n_angle): # 18
        a = []
        for i in range(1,7):
            t = self.PositiveKine(n_angle,i)
            a = a + [-t[0][3], -t[1][3], t[2][3]]
        a = np.array(a).flatten()
        # b = (a[0:5] + a[1:6])/2
        # print("all joint pos", a, a.shape)
        return a

    def collisionAngle(self,now_angle):         #判断是否碰撞
        theta2=now_angle[1]
        theta3=now_angle[2]
        l1=0.425
        l2=0.395
        R=0.1
        x=0.6
        w=0.08
        theta2L=-math.asin((2*R+w)/2*x)
        theta2H=math.asin((2*R+w)/2*x)
        m=math.sqrt((l1*math.sin(theta2))**2+(x-l1*math.cos(theta2))**2)
        theta3L=math.acos((l1**2+m**2-x**2)/(2*l1*m))-math.asin((w+2*R)/(2*m))-math.pi
        theta3H=math.acos((l1**2+m**2-x**2)/(2*l1*m))+math.asin((w+2*R)/(2*m))-math.pi
        if theta2<theta2H and theta2>theta2L:
            return True
        if theta3<theta3H and theta3>theta3L:
            return True
        return False

    def near_obs(self,my_obs, hand_obs_dis, if_print=False) -> float:
        near_r = 0.15 # param 球边离关节中心的距离
        obs = my_obs[0] 
        obs_pos = obs[9:12]
        res = 0
        for i in range(12, len(obs)+3, 3):
            if i >= len(obs):
                dis = hand_obs_dis
            else:
                dis = np.linalg.norm(obs_pos - obs[i:i+3])
            if dis < 0.1 + near_r:
                if if_print:
                    print(obs_pos, obs[i:i+3])
                    print("touch obs")
                res = max(res, 0.5/(dis-near_r))
        return res


    def idlePos(self,t, o):
        target = np.array([0.5, 0.07, 0.40097904, 0.04461199, 0.5, 0.5])
        try:
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
        except:
            print("error")
        return target
    
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

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        path_right = os.path.join(os.path.dirname(__file__), "right_model")
        path_left = os.path.join(os.path.dirname(__file__), "left_model")
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