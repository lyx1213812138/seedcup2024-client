import math
import numpy as np
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
        if not isinstance(n_angle, np.ndarray):
            n_angle = np.array(n_angle)
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
        AP = ob - j1
        AB = j2 - j1
        t = np.dot(AP,AB)/np.dot(AB,AB)
        t = max(0, min(1,t))
        Q = j1 + t*AB
        d=np.linalg.norm(ob - Q)
        return d
    
    def collisionAngle(self,n_angle,ob, dis_ob=None):         #判断是否碰撞
        global theta
        theta = math.pi*(2*n_angle-1)
        T1= self.transferQue(0,1)
        T2 = T1.dot(self.transferQue(1,2))
        T3 = T2.dot(self.transferQue(2,3))
        T4 = T3.dot(self.transferQue(3,4))
        T5 = T4.dot(self.transferQue(4,5))
        T6 = T5.dot(self.transferQue(5,6))
        p1=np.array([-T1[0][3], -T1[1][3], T1[2][3]])
        p2=np.array([-T2[0][3], -T2[1][3], T2[2][3]])
        p3=np.array([-T3[0][3], -T3[1][3], T3[2][3]])
        p4=np.array([-T4[0][3], -T4[1][3], T4[2][3]])
        p5=np.array([-T5[0][3], -T5[1][3], T5[2][3]])
        p6=np.array([-T6[0][3], -T6[1][3], T6[2][3]])
        p8 = self.LastPos(n_angle)
        p9 = np.array([-0.138*math.sin(theta[0]), 0.138*math.cos(theta[0]), p1[2]])
        p7=np.array([p2[0]+p9[0]-p1[0], p2[1]+p9[1]-p1[1], p2[2]+p9[2]-p1[2]])
        d1=self.disjo(p9,p7,ob)
        d2=self.disjo(p7,p2,ob)
        d3=self.disjo(p2,p3,ob)
        d4=self.disjo(p3,p4,ob)
        d5=self.disjo(p4,p5,ob)
        d6=self.disjo(p5,p6,ob)
        d7=self.disjo(p6,p8,ob)
        d8=np.linalg.norm(p3 - ob)
        d9=np.linalg.norm(p4 - ob)
        d10=np.linalg.norm(p5 - ob)
        d11=np.linalg.norm(p8 - ob)
        # print(p1,p9,p7,p2,p3,p4,p5,p6,p8)
        # print(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11)
        # print('5 ', d10)
        # print('5 6', d6)
        if min(d1,d2,d3) < 0.13:
            # print("&", min(d1,d2,d3))
            return True
        if min(d4, d5, d6) < 0.16:
            # print("&&", min(d4, d5, d6))
            return True
        if min(d8, d9, d10) < 0.19:
            # print("&&&", min(d8, d9, d10))
            return True
        if min(d7,d8) < 0.20:
            # print("&&&", d7)
            return True
        if dis_ob != None:
            dis_ob[tuple(n_angle)] = min(min(d1,d2,d3) - 0.13, min(d4, d5, d6) - 0.16, min(d8, d9, d10) - 0.19, min(d7,d8) - 0.20)
        return False