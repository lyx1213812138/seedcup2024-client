import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import math
from time import sleep
import env


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

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        now = 0
        # 自定义初始化
        pass
        
    def get_action(self, observation):
        global now
        a = observation[0, :6] # 六个关节角度
        t = observation[0, 6:9] # 目标位置
        o = observation[0, 9:] # 障碍物位置
        print(a)
        # 初始
        # [0.36261892 0.3400045  0.11557633 0.04461199 0.36262435 0.49998325]
        action = [0, 0, 0, 0, 0, 0]

        # 旋转
        angle_t = np.arctan2(t[1], t[0])
        angle_t_n = (angle_t / math.pi) / 2 # 归一化
        print(angle_t, angle_t_n)
        action[0] = to_target(a[0], angle_t_n)
        
        now = 3
        # 前后 & 上下 a[1] a[2]
        if now == 0: # 上抬
            action[1] = -1
            action[2] = 1
            if a[2] > 0:
                now = now + 1
        elif now == 1: # 向前
            action[1] = -1
            action[2] = 1
            # 0.07   0.40097904
            # -155 -36
            if a[2] > 0.25:
                now = now + 1
        elif now == 2: # 动手
            action[1] = -1
            action[2] = 1
            action[3] = -1
        
        # action = [0, -1.1, 1, 0, 0, 0]
        # 0.20361236 0.25160331


        # 取消向上过去
        # if a[0] > 0.25:
        #     action[0] = -1
        # if a[1] > 0.15:
        #     action[1] = -1
        # if a[2] < 0.2:
        #     action[2] = 1.5
        # elif a[2] < 0.35:
        #     action[2] = 1
        # if a[3] < 0.25:
        #     action[3] = 1
        # if a[4] < 1:
        #     action[4] = 1
        # end 0.25 0.14940223 0.34809947 0.25 0.5
        # [0.2452434  0.21460503 0.24620441 0.11941871 0.48773259 0.53049225]
        return action / (np.max(np.abs(action)))

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        # self.model = PPO.load("model.zip", device="cpu")
        self.model = PPO.load("ppo", device="cpu")

    def get_action(self, observation):
        action, _ = self.model.predict(observation)
        print('action: ', action)
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
        action = [0, 0, 0, 0, 0, 0]
        print(a)

        # # action[0]
        # r = 0.9
        # l1 = 0.865 * r
        # l2 = 0.225 * r
        # l3 = 0.121  * r
        # l = np.sqrt(l1 ** 2 + l3 ** 2)
        # d = np.sqrt(t[0] ** 2 + t[1] ** 2)
        # if l + l2 < d: d = l + l2 - 0.0000001
        # a1 = math.acos((d**2 + l**2 - l2**2) / (2 * d * l))
        # a2 = np.arctan2(t[1], t[0])
        # oa = np.arctan2(o[1], o[0])
        # a3 = np.arctan2(l3, l1)
        # if oa > a2 + 0.01:
        #     a_t = a2 - a1 + a3
        # else:
        #     a_t = a1 + a2 + a3
        # a_t_n = (a_t / math.pi) / 2%1
        # # print(a1, a2, a3)

        # # action[4]
        # a4 = math.acos((l**2 + l2**2 - d**2) / (2 * l * l2))
        # if oa > a2 + 0.01:
        #     a_t_2 = math.pi/2+a4+a3
        # else:
        #     a_t_2 = math.pi/2-a4+a3
        # a_t_n2 = (a_t_2/math.pi + 1)/2%1
        # # print(a4, a_t_n2)

        # action = np.array(self.to_target2(a, np.array([a_t_n, 0.07, 0.40097904, 0.04461199, a_t_n2, 0.5])))
        # # [0.36261892 0.3400045  0.11557633 0.04461199 0.36262435 0.49998325]
        # ### [0.33914432 0.11602792 0.3494153  0.06071219 0.47847939 0.51293832]
        # print('action: ', action / (np.max(np.abs(action))))

        # action[0]
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
            #print(a4, a_t_n2)
            #action[5] && z
            if abs(a[4]-a_t_n2) > 0.04:
                a_t_n3 -= 0.01
                action = np.array(self.to_target2(a, np.array([a_t_n, a_t_n3, 0.40097904, 0.04461199 - a_t_n3 + 0.07, a_t_n2, 0.5])))

        # print('action: ', action)
        if np.max(np.abs(action)) == 0:
            action = np.random.rand(6) - 0.5
        return action / (np.max(np.abs(action)))
        


        

        