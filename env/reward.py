import numpy as np
from .ccalc import Calc

def reward(self) -> float:
    calc = Calc()
    reward = 0
    end_reach = False
    end_max_steps = False

    # XXX 计算距离reward
    obs = self.get_observation()
    total_dis = self.get_dis() + np.linalg.norm(obs[0][36:42] - obs[0][0:6])
    # print('total_dis: ', total_dis)
    if self.last_dis >= 0:
        reward += 500*(self.last_dis - total_dis)
        # print(500*(self.last_dis - total_dis))
    self.last_dis = total_dis

    # XXX 接近障碍物
    # dis =  calc.collisionAngle(obs[0:6], obs[9:12])
    # dis = calc.near_obs(self.get_observation(), self.get_obs_dis())
    # if self.last_obs_dis > 0:
    #     if dis < 0.2:
    #         reward -= max(200*(self.last_obs_dis - dis), 10)
    #     elif dis < 0.25:
    #         reward -= max(100*(self.last_obs_dis - dis), 20)
    # self.last_obs_dis = dis

    # 获取与桌子和障碍物的接触点
    table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
    obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

    for contact_point in table_contact_points or obstacle1_contact_points:
        link_index = contact_point[3]
        if link_index not in [0, 1]:
            if not self.obstacle_contact:
                # XXX contact
                reward = -100 # XXXXXXXXXX
                self.obstacle_contact = True
            reward = -15 # XXXXXXXXXX

    # 计算结束
    # TEST
    if self.get_dis() < 0.05 and self.step_num <= self.max_steps:
        self.success_reward = 100
        if self.obstacle_contact:
            if self.is_senior:
                self.success_reward = 20
            elif not self.is_senior:
                self.success_reward = 50
            else:
                return 
        self.terminated = True
        end_reach = True
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
        self.truncated = True
        end_max_steps = True

    if end_reach or end_max_steps:
        if end_reach: # XXX reach target
            if self.obstacle_contact:
                reward = -150
            else:
                reward = 200
            print("# Terminated for reaching target")
        elif end_max_steps: # XXX reach max steps
            reward = self.success_reward * 0.5
            if self.success_reward < 30:
                reward = -200
            print("# Terminated for reaching max steps")
        # reward = self.success_reward * 3
        # if self.step_num >= self.target_step:
            # reward = reward * self.front_dis * 10
            # if self.success_reward < 30:
            #     reward = min(-500, -self.total_reward-100)
        self.total_reward += reward
        print("dis: ", self.get_dis(),
            'obs_dis: ', self.get_obs_dis(),
            'touch: ' , self.obstacle_contact,
            'step_num: ', self.step_num,
            '\n\ttotal_reward: ', self.total_reward,
            '\n\tsuccess_reward: ', self.success_reward)

    # XXX calc reward
    # if self.step_num <= self.target_step + 1:
    #     reward = 0
    #     self.front_dis = self.get_dis()
    # if reward > 0:
    #     reward = reward * self.front_dis * 10
    self.total_reward += reward
    return reward