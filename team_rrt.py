import numpy as np
from abc import ABC, abstractmethod
import math
import os
from env.utils import relative_dir, predict_pos, next_tar_step
from env.ccalc import Calc
from queue import PriorityQueue
    
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
        self.flag = 0
        self.vx = 0 
        self.vz =0
        self.target = [0, 0, 0]
        self.num = 0
        self.n_obs = [0, 0, 0]

        self.max_steps = 200
        self.target_step = 94 # 未来目标位置的步数
            

    def get_action(self, observation):
        self.set_ball_pos = observation[1] # !!!
        observation = observation[0]
        n_angle = observation[0, :6]
        target_position = observation[0][6:9]
        obstacle1_position = observation[0][9:12]
        self.num += 1
        if self.flag == 0:
            self.n_obs = obstacle1_position
            self.st = target_position
            self.target = target_position
            self.flag = 1
            self.num = 0
            self.last_dis = -1
            return np.array([0, 0, 0, 0, 0, 0])
        if self.n_obs[0] != obstacle1_position[0]:
            # print('restart')
            self.n_obs = obstacle1_position
            self.st = target_position
            self.target = target_position
            self.flag = 0
            self.num = 0
            self.last_dis = -1
            return np.array([0, 0, 0, 0, 0, 0])
        if self.flag == 1:
            # XXX init
            self.vx = target_position[0]-self.st[0]
            self.vz = target_position[2]-self.st[2]
            self.end_tar = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.max_steps)
            self.target = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.target_step)
            self.path = self.a_star(n_angle, self.target, obstacle1_position)
            self.now_tar_idx = 0 # the index of now target in path
            self.flag = -1
            return np.array([0, 0, 0, 0, 0, 0])
        if self.flag == -1:
            # if self.set_ball_pos != None: # !!! ball
            #     self.set_ball_pos(calc.LastPos(n_angle))
            action = (self.path[self.now_tar_idx] - n_angle) * 360
            self.now_tar_idx = self.now_tar_idx + 1 if self.now_tar_idx < len(self.path) - 1 else self.now_tar_idx
            return np.reshape(action, (6, ))

    def a_star(self, start, tarpos, obspos):
        """
        Args:
            start: list of 6 : angles
            tarpos: list of 3 : position

        Space:
            360 ^ 6
        """
        start = np.array(start)
        q = PriorityQueue()
        g = {} # g[ang] = 已走过的步数
        pre = {} # pre[ang] = 上一个状态
        hm = {} # hm[ang] = h
        g[tuple(start)] = 0
        pre[tuple(start)] = None
        hm[tuple(start)] = self.h(start, tarpos)
        
        q.put((hm[tuple(start)],) + tuple(start))
        while not q.empty():
            cur = q.get()[1:7]
            # print(cur)
            # self.set_ball_pos(calc.LastPos(np.array(cur)))
            if self.is_end(cur, tarpos):
                return self.find_path(pre, cur)
            i = [0] * 6
            for i[0] in range(-1, 2):
                for i[1] in range(-1, 2):
                    for i[2] in range(-1, 2):
                        for i[3] in range(-1, 2):
                            for i[4] in range(-1, 2):
                                for i[5] in range(-1, 2):
                                    new_ang = np.array([cur[j] + i[j] / 60 for j in range(6)])
                                    if self.not_valid(new_ang, obspos):
                                        continue
                                    new_g = g[tuple(cur)] + 1
                                    if tuple(new_ang) not in g or new_g < g[tuple(new_ang)]:
                                        g[tuple(new_ang)] = new_g
                                        pre[tuple(new_ang)] = cur
                                        if tuple(new_ang) not in hm:
                                            hm[tuple(new_ang)] = self.h(new_ang, tarpos)
                                        q.put((new_g + hm[tuple(new_ang)],) + tuple(new_ang))
        return None
      

    def h(self, ang1, tarpos): 
        if not isinstance(ang1, np.ndarray):
            ang1 = np.array(ang1)
        dis_per_step = 0.01
        return np.linalg.norm(calc.LastPos(ang1) - tarpos) / dis_per_step
    
    
    def is_end(self, ang, tarpos):
        if not isinstance(ang, np.ndarray):
            ang = np.array(ang)
        dis = np.linalg.norm(calc.LastPos(ang) - tarpos)
        print(dis)
        return dis < 0.05
    

    def not_valid(self, ang, obspos):
        if not isinstance(ang, np.ndarray):
            ang = np.array(ang)
        return calc.collisionAngle(ang, obspos)
    

    def find_path(self, premap, end):
        path = []
        while end != None:
            path.append(end)
            end = premap[end]
        return path[::-1]
        
