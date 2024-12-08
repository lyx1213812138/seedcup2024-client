import math
import numpy as np

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
        if d1 <= 0.14 or d2 <= 0.14 or d3 <= 0.14 or d4 <= 0.14 or d5 <= 0.14:
            return True
        return False
