import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import math
from time import sleep
import os
# from env.utils import relative_dir, predict_pos, next_tar_step
# from env.ccalc import Calc
import torch

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
        pp = np.array([-a[0][3], -a[1][3], a[2][3]])
        return 2.5 * pp - 1.5 * self.WristPos(n_angle)
    
    def transferQue_torch(self, i, j, theta):      #齐次变换方程
        global a, d
        alpha = torch.tensor([torch.pi/2 , 0, 0, torch.pi/2 , -torch.pi/2 , 0])
        c1 = torch.cos(theta[j-1])
        s1 = torch.sin(theta[j-1])
        c2 = torch.cos(alpha[j-1])
        s2 = torch.sin(alpha[j-1])
        # print(c2, ',', s2)
        dd = torch.tensor(d[j-1])
        aa = torch.tensor(a[j-1])
        A2 = torch.tensor([[c1, -s1*c2, s1*s2, aa*c1],
                    [s1, c1*c2, -c1*s2, aa*s1], 
                    [0, s2, c2, dd], 
                    [0, 0, 0, 1]], dtype=torch.float32)
        # print('A2', A2)
        A = torch.mul(c1, torch.tensor([[1, 0, 0, aa],[0, c2, -s2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])) + torch.mul(s1, torch.tensor([[0, -c2, s2, 0], [1, 0, 0, aa], [0, 0, 0, 0], [0, 0, 0, 0]])) + torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, s2, c2, dd], [0, 0, 0, 1]])
        # print('A', j, ':', A)
        return A


    def PositiveKine_torch(self, n_angle, i=6):   #正运动学求解
        global a, d, alpha
        theta = torch.pi*(2*n_angle-1)
        T = [0]*7
        T[1] = self.transferQue_torch(0, 1, theta)
        T[2] = torch.mm(T[1], self.transferQue_torch(1,2, theta))
        T[3] = torch.mm(T[2], self.transferQue_torch(2,3, theta))
        T[4] = torch.mm(T[3], self.transferQue_torch(3,4, theta))
        T[5] = torch.mm(T[4], self.transferQue_torch(4,5, theta))
        T[6] = torch.mm(T[5], self.transferQue_torch(5,6, theta))
        return T[i]

    def LastPos_torch(self, n_angle):
        global d
        T = self.PositiveKine_torch(n_angle)
        # print(T)
        # pp = torch.tensor([-a[0][3], -a[1][3], a[2][3]])
        T = T.t()
        pp = torch.mul(T[3][0:3], torch.tensor([-1, -1, 1]))
        # print('n_angle', n_angle)
        # print('pp', pp)
        return pp + 1.5 * torch.mul(torch.mul(T[2][0:3], d[5]),torch.tensor([-1, -1, 1]))
    
    def WristPos(self,n_angle):
        global a, theta, d, alpha
        T=self.PositiveKine(n_angle)
        # print(T[0][3],T[1][3],T[2][3])
        pw=np.array([-(T[0][3]-d[5]*T[0][2]), -(T[1][3]-d[5]*T[1][2]), T[2][3]-d[5]*T[2][2]])
        return pw

    def WristPos_torch(self,n_angle):
        global a, theta, d, alpha
        T=self.PositiveKine_torch(n_angle)
        T = T.t()
        pw = torch.mul(T[3][0:3], torch.tensor([-1, -1, 1])) - torch.mul(T[2][0:3], d[5])
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

    # def collisionAngle(self,now_angle):         #判断是否碰撞
    #     theta2=now_angle[1]
    #     theta3=now_angle[2]
    #     l1=0.425
    #     l2=0.395
    #     R=0.1
    #     x=0.6
    #     w=0.08
    #     theta2L=-math.asin((2*R+w)/2*x)
    #     theta2H=math.asin((2*R+w)/2*x)
    #     m=math.sqrt((l1*math.sin(theta2))**2+(x-l1*math.cos(theta2))**2)
    #     theta3L=math.acos((l1**2+m**2-x**2)/(2*l1*m))-math.asin((w+2*R)/(2*m))-math.pi
    #     theta3H=math.acos((l1**2+m**2-x**2)/(2*l1*m))+math.asin((w+2*R)/(2*m))-math.pi
    #     if theta2<theta2H and theta2>theta2L:
    #         return True
    #     if theta3<theta3H and theta3>theta3L:
    #         return True
    #     return False

    def near_obs(self,my_obs, hand_obs_dis, if_print=False) -> float:
        near_r = 0.1 # param 球边离关节中心的距离
        near_r2 = 0.15
        obs = my_obs[0] 
        obs_pos = obs[9:12]
        res = 0
        for i in range(12, 39, 3):
            if i >= 36:
                dis = hand_obs_dis
            else:
                dis = np.linalg.norm(obs_pos - obs[i:i+3])
            if dis < 0.1 + near_r2:
                if if_print:
                    print(obs_pos, obs[i:i+3])
                    print("touch obs")
                res = min(res, dis)

        return res


    def idlePos(self,t, o):
        # action[0]
        r = 0.9
        l1 = 0.865 * r
        l2 = 0.225 * r
        l3 = 0.121  * r
        l = np.sqrt(l1 ** 2 + l3 ** 2)
        d = np.sqrt(t[0] ** 2 + t[1] ** 2)
        if l + l2 < d: d = l + l2 - 0.0000001
        a1 = math.acos((d**2 + l**2 - l2**2) / (2 * d * l))
        a2 = np.arctan2(t[1], t[0])
        oa = np.arctan2(o[1], o[0])
        a3 = np.arctan2(l3, l1)
        if oa > a2 + 0.01:
            a_t = a2 - a1 + a3
        else:
            a_t = a1 + a2 + a3
        a_t_n = (a_t / math.pi) / 2%1

        # action[4]
        a4 = math.acos((l**2 + l2**2 - d**2) / (2 * l * l2))
        if oa > a2 + 0.01:
            a_t_2 = math.pi/2+a4+a3
        else:
            a_t_2 = math.pi/2-a4+a3
        a_t_n2 = (a_t_2/math.pi + 1)/2%1

        # action[5] && z
        a_t_n3 = -60*(t[2]**4)+51.3333*(t[2]**3)-15.65*(t[2]**2)+2.161*t[2]-0.045
        return np.array([a_t_n, a_t_n3, 0.40097904, 0.04461199 - a_t_n3 + 0.07, a_t_n2, 0.5])
    
    def disjo(self,j1,j2,ob):
        s=np.array([j1[0]-j2[0], j1[1]-j2[1], j2[2]-j2[2]])
        v=np.array([ob[0]-j2[0], ob[1]-j2[1], ob[2]-j2[2]])
        d=np.linalg.norm((np.cross(s,v)))/math.sqrt(s[0]**2+s[1]**2+s[2]**2)
        return d
    
    def collisionAngle(self,n_angle,ob):         #判断是否碰撞
        global theta
        theta = math.pi*(2*n_angle-1)
        T1=self.transferQue(0,1)
        T2=self.transferQue(1,2)
        T3=self.transferQue(2,3)
        T4=self.transferQue(3,4)
        T5=self.transferQue(4,5)
        T6=self.transferQue(5,6)
        T11=T1
        T22=T1.dot(T2)
        T33=T1.dot(T2).dot(T3)
        T44=T1.dot(T2).dot(T3).dot(T4)
        T55=T1.dot(T2).dot(T3).dot(T4).dot(T5)
        T66=T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
        p1=np.array([-T11[0][3], -T11[1][3], T11[2][3]])
        p2=np.array([-T22[0][3], -T22[1][3], T22[2][3]])
        p3=np.array([-T33[0][3], -T33[1][3], T33[2][3]])
        p4=np.array([-T44[0][3], -T44[1][3], T44[2][3]])
        p5=np.array([-T55[0][3], -T55[1][3], T55[2][3]])
        p6=np.array([-T66[0][3], -T66[1][3], T66[2][3]])
        d1=self.disjo(p1,p2,ob)
        d2=self.disjo(p2,p3,ob)
        d3=self.disjo(p3,p4,ob)
        d4=self.disjo(p4,p5,ob)
        d5=self.disjo(p5,p6,ob)
        x = np.min([d1,d2,d3,d4,d5])
        return x
    
calc = Calc()

def predict_pos(now, v, step):
  v = np.array(v)
  if v.shape[0] != 3:
    v = np.array([v[0], 0, v[1]])
  t = np.array(now) + v * step / 12
  if t[0] > 0.5:
    t[0] = 1 - t[0]
  if t[0] < -0.5:
    t[0] = -1 - t[0]
  if t[2] > 0.5:
    t[2] = 1 - t[2]
  if t[2] < 0.1:
    t[2] = 0.2 - t[2]
  return t


# pos1 在 pos2 的哪个方向, 只看x和y坐标
#  o
# / \
# 2  \
#    1
# 1 在 2 的右边(right)
# right = 1, left = -1, center = 0
def relative_dir(pos1, pos2, use_int=False):
    angle1 = np.arctan2(pos1['y'], pos1['x'])
    angle2 = np.arctan2(pos2['y'], pos2['x'])
    if abs(angle1 - angle2) < 0.1:
        return 'center' if not use_int else 0
    elif angle1 - angle2 > 0:
        return 'right' if not use_int else 1
    elif angle1 - angle2 < 0:
        return 'left' if not use_int else -1
    return 'center'


def next_tar_step(now, tar1, max):
  if now <= tar1:
    return tar1 - now 
  return min(max, now + 1) - now
  

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
        path_right = os.path.join(os.path.dirname(__file__), "left_model.zip")
        path_left = os.path.join(os.path.dirname(__file__), "right_model.zip")
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
        self.target_step = 94 # 未来目标位置的步数


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
        # set_ball_pos = observation[1]
        # observation = observation[0]
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
            self.vx = target_position[0]-self.st[0]
            self.vz = target_position[2]-self.st[2]
            self.end_tar = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.max_steps)
            self.target = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.target_step)
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
            if np.linalg.norm(self.target - obstacle1_position) > 0.3 and (self.dir_future != 0 or obstacle1_position[2]<=0.2):
                self.target_step = 70
                self.target = predict_pos(self.st, [self.vx * 12, self.vz * 12], self.target_step)
                print('be quick')
            print("dir", self.dir_future, self.dir_end)
            self.flag = -1
            return np.array([0, 0, 0, 0, 0, 0])
        if self.flag == -1:
            # TODO 如果方向不一致, 改变目标
            if self.num <= self.target_step and self.dir_future == self.dir_end:
                real_target = self.target
                real_dir = self.dir_future
            elif self.dir_future != self.dir_end:
                real_target = self.end_tar
                real_dir = self.dir_end
            elif self.num > self.target_step:
                real_target = target_position 
                real_dir = self.dir_end

            # print(self.num, target_position)
            if self.num == self.target_step:
                if np.linalg.norm(np.array(target_position) - np.array(self.target)) > 0.01:
                    print('!target',self.target, '\n\t', target_position)
                    exit(1)
            elif self.num == self.max_steps-1:
                if np.linalg.norm(np.array(target_position) - np.array(self.end_tar)) > 0.01:
                    print('!end',self.end_tar, '\n\t', target_position)
                    exit(1)

            # if set_ball_pos != None:
            #     set_ball_pos(calc.LastPos(n_angle))
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
            if self.num == self.target_step:
                print("change model")



            if self.num <= self.target_step:
                if real_dir >= 0:
                    action, _ = self.model_r.predict(my_obs[:,:42])
                else:
                    action, _ = self.model_l.predict(my_obs[:,:42])
            else:
                # action, _ = self.model_end.predict(my_obs2)  
                action = self.traditional_get_action(my_obs2)
                if torch.all(action == 0):
                    if real_dir >= 0:
                        action, _ = self.model_r.predict(my_obs[:,:42])
                    else:
                        action, _ = self.model_l.predict(my_obs[:,:42])
            
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

        