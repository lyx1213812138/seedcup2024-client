import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import math
from time import sleep
import os
from env.utils import relative_dir, predict_pos, next_tar_step
from env.ccalc import Calc
import torch
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
        path_right = os.path.join(os.path.dirname(__file__), "zip/right_model.zip")
        path_left = os.path.join(os.path.dirname(__file__), "ppo_eval_logs_42/left/best_model.zip")
        # path_tot = os.path.join(os.path.dirname(__file__), "ppo_eval_logs_47/test/best_model.zip")
        # path_end = os.path.join(os.path.dirname(__file__), "model/ppo_eval_logs_18/end2_x_g_0_left/best_model.zip")
        # print("ppo load path: ", path_tot, path_end)
        sleep(1)
        self.model_r = PPO.load(path_right, device="cpu")
        self.model_l = PPO.load(path_left, device="cpu")
        # self.model = PPO.load(path_tot, device="cpu")
        # self.model_end = PPO.load(path_end, device="cpu")
        
        self.flag = 0
        self.vx = 0 
        self.vz =0
        self.target = [0, 0, 0]
        self.num = 0
        self.n_obs = [0, 0, 0]

        self.max_steps = 200
        self.target_step = 120 # 未来目标位置的步数


    # v : [vx, vz]
    def get_future_pos(self, now_tar_pos, v, step):
        step = max(0, step - self.num)
        # print('step:', step, self.num)
        now_tar_pos = now_tar_pos.copy()
        v = v.copy()
        for _ in range(step):
            now_tar_pos[0] += v[0]
            now_tar_pos[1] += v[1]
            if now_tar_pos[0] > 0.5 or now_tar_pos[0] < -0.5:
                v[0] = -v[0]
            if now_tar_pos[2] > 0.5 or now_tar_pos[2] < 0.1:
                v[1] = -v[1]
        # print(now_tar_pos)
        return now_tar_pos
            
    def get_action(self, observation):
        self.set_ball_pos = observation[1]
        observation = observation[0]
        n_angle = observation[0, :6]
        target_position = observation[0][6:9]
        obstacle1_position = observation[0][9:12]
        # print(self.num,target_position)
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
            # init
            self.vx = target_position[0]-self.st[0]
            self.vz = target_position[2]-self.st[2]
            self.end_tar = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.max_steps)
            self.target = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.target_step)
            self.path = []
            # if np.linalg.norm(self.target - obstacle1_position) > 0.3:
            #     self.target_step = 70
            #     self.target = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.target_step)
            #     print('be quick')
            self.dir_future = relative_dir(
                {'x': self.target[0], 'y': self.target[2]},
                {'x': obstacle1_position[0], 'y': obstacle1_position[1]}, 
                True
            )
            self.dir_end = relative_dir(
                {'x': self.end_tar[0], 'y': self.end_tar[2]},
                {'x': obstacle1_position[0], 'y': obstacle1_position[1]}, 
                True
            )
            print("dir", self.dir_future, self.dir_end)
            self.now_state = 0 # 0: RL, 1: A*, 2: torch
            self.init_state = 1
            self.flag = -1
            return np.array([0, 0, 0, 0, 0, 0])
        if self.flag == -1:
            # TODO 如果方向不一致, 改变目标
            # if self.num <= self.target_step and self.dir_future == self.dir_end:
            #     real_target = self.target
            #     real_dir = self.dir_future
            # elif self.dir_future != self.dir_end:
            #     real_target = self.end_tar
            #     real_dir = self.dir_end
            # elif self.num > self.target_step:
            #     real_target = target_position 
            #     real_dir = self.dir_end
            # print(np.linalg.norm(target_position - obstacle1_position))
            if np.linalg.norm(self.target - obstacle1_position) < 0.23 or (self.dir_future == 0 and self.dir_end != 0):
                real_target = self.end_tar
                real_dir = self.dir_end
                self.target_step = 130
            else:
                real_target = self.target
                real_dir = self.dir_future

            # print(self.num, target_position)
            # if self.num == self.target_step:
            #     if np.linalg.norm(np.array(target_position) - np.array(self.target)) > 0.01:
            #         print('!target',self.target, '\n\t', target_position)
            #         exit(1)
            # elif self.num == self.max_steps-1:
            #     if np.linalg.norm(np.array(target_position) - np.array(self.end_tar)) > 0.01:
            #         print('!end',self.end_tar, '\n\t', target_position)
            #         exit(1)

            # if self.set_ball_pos != None:
            #     self.set_ball_pos(calc.LastPos(n_angle))
                # set_ball_pos(target_angle)
            my_obs = np.hstack((
                n_angle, real_target, obstacle1_position,
                calc.LastPos(n_angle), 
                calc.WristPos(n_angle),
                calc.jointPos(n_angle),
                calc.idlePos(real_target, obstacle1_position),
                [self.vx * 12, 0, self.vz * 12],
                [self.dir_future, self.dir_end],
            )).reshape(1, -1)


            my_obs2 = np.hstack((
                n_angle, target_position, obstacle1_position
            )).reshape(1, -1)
        
            obs_angle = np.arctan2(obstacle1_position[1], obstacle1_position[0])
            target_angle = np.arctan2(target_position[1], target_position[0])

            action = [0] * 6
            
            change_target_step = self.target_step 
            a_star_target_step = self.target_step + 25
            if self.now_state == 0:
                if real_dir >= 0:
                    action, _ = self.model_r.predict(my_obs[:,:42])
                else:
                    action, _ = self.model_l.predict(my_obs[:,:42])
                # if calc.collisionAngle(n_angle + action, obstacle1_position):
                #     action = -action
                #     self.now_state = 1
                #     self.init_state = 1
                if self.num > 90:
                    self.now_state = 1
                    self.init_state = 1
            elif self.now_state == 1:
                if self.num >= self.max_steps - 15:
                    self.now_state = 2
                    self.init_state = 1

                if self.init_state:
                    self.init_state = 0
                    print('start A*')
                    self.path = self.a_star(n_angle, real_target, obstacle1_position)
                    self.now_tar_idx = 0

                if self.path == None: # go to traditional
                    self.now_state = 2
                    self.init_state = 1
                    return np.zeros(6)

                if np.linalg.norm(self.path[self.now_tar_idx] - n_angle) < 0.003:
                    self.now_tar_idx = self.now_tar_idx + 1
                    if self.now_tar_idx >= len(self.path):
                        self.now_state = 2
                        self.init_state = 1
                        return np.zeros(6)
                    

                action = (self.path[self.now_tar_idx] - n_angle) * 360
                # print(n_angle)
                # if calc.collisionAngle(n_angle + np.array(action), obstacle1_position):
                #     self.path = self.a_star(n_angle, self.path[self.now_tar_idx], obstacle1_position) + self.path
                # FIXME: path target 之间可能碰撞
            elif self.now_state == 2:
                if self.init_state:
                    self.init_state = 0
                    print('traditional')
                action = self.traditional_get_action(my_obs2)
                if calc.collisionAngle(n_angle + np.array(action), obstacle1_position):
                    self.now_state = 1
                    self.init_state = 1
                    return np.zeros(6)
                    
            return np.reshape(action, (6, ))

    def traditional_get_action(self, observation):
        n_angle = observation[0, :6]
        target_position = observation[0][6:9]
        obstacle1_position = observation[0][9:12]
        a = [0] * 6
        
        torch.enable_grad()
        n_angle_tensor = torch.tensor(n_angle, dtype=torch.float32, requires_grad=True)
        dis = - torch.sum( (calc.LastPos_torch(n_angle_tensor) - torch.tensor(target_position)) ** 2 )

        if self.last_dis > 0 and dis - self.last_dis >= 1e-4: # maybe collision
            return torch.zeros(6)

        dis.backward()
        if n_angle_tensor.grad == None or torch.any(torch.isnan(n_angle_tensor.grad)):
            return torch.zeros(6)
        with torch.no_grad():
            # print('grad', n_angle_tensor.grad)
            action = n_angle_tensor.grad / torch.max(torch.abs(n_angle_tensor.grad))
        n_angle_tensor.grad.zero_()
        # print(action)
        return action

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
        lp = {} # lp[ang] = lastpos(ang)
        dis_ob = {}
        g[tuple(start)] = 0
        pre[tuple(start)] = None
        lp[tuple(start)] = calc.LastPos(start)
        total_cnt = 0
        
        q.put((self.h(lp[tuple(start)], tarpos),) + tuple(start))
        while not q.empty():
            cur = q.get()[1:7]
            if tuple(cur) not in lp:
                lp[tuple(cur)] = calc.LastPos(cur)
            curpos = lp[tuple(cur)]
            # print(cur)
            # self.set_ball_pos(curpos)

            dis = np.linalg.norm(curpos - tarpos)
            now_dis_ob = dis_ob.get(tuple(cur), 0)
            print(dis, now_dis_ob)
            total_cnt += 1
            if dis < 0.06 or (total_cnt > 30 and dis < 0.12):
                return self.find_path(pre, cur)
            if total_cnt > 50:
                return self.find_path(pre, cur)
            
            if min(dis, now_dis_ob) > 0.3:
                step_len = 0.05
            elif now_dis_ob < 0.05 or dis < 0.15:
                step_len = 0.01
            else:
                step_len = 0.02

            new_g = g[tuple(cur)] + 1
            i = [0] * 6
            for i[0] in range(-1, 2):
                for i[1] in range(-1, 2):
                    for i[2] in range(-1, 2):
                        for i[3] in range(-1, 2):
                            for i[4] in range(-1, 2):
                                for i[5] in range(-1, 2):
                                    new_ang = np.array([cur[j] + i[j] * step_len for j in range(6)])
                                    if tuple(new_ang) == tuple(cur):
                                        continue
                                    if tuple(new_ang) not in g and self.not_valid(new_ang, obspos, dis_ob):
                                        continue
                                    if tuple(new_ang) not in g or new_g < g[tuple(new_ang)]:
                                        g[tuple(new_ang)] = new_g
                                        pre[tuple(new_ang)] = cur
                                        if tuple(new_ang) not in lp:
                                            lp[tuple(new_ang)] = calc.LastPos(new_ang)
                                        q.put((new_g + self.h(lp[tuple(new_ang)], tarpos),) + tuple(new_ang))
        return None
      

    def h(self, lastpos, tarpos): 
        dis_per_step = 0.007
        return np.linalg.norm(lastpos - tarpos) / dis_per_step
    

    def not_valid(self, ang, obspos, dis_ob):
        if not isinstance(ang, np.ndarray):
            ang = np.array(ang)
        return calc.collisionAngle(ang, obspos, dis_ob)
    

    def find_path(self, premap, end):
        path = []
        while end != None:
            path.append(end)
            end = premap[end]
        return path[::-1]

        