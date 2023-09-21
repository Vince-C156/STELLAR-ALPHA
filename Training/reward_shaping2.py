import numpy as np
import pdb
#import cupy as np

class reward_conditions:

    def __init__(self, chaser):
        self.chaser = chaser
        self.time_limit = 1500
        #self.obstacle_list = obstacle_list
    def inbounds(self): 
        if self.chaser_current_distance() < 1500.0:
           return True
        else:
           return False

    def target_collision(self):
        pass

    def chaser_current_distance(self):
        target_point = self.chaser.docking_point
        curr_pos = self.chaser.state[:3]

        curr_sumsq = np.sum(np.square(target_point - curr_pos))
        curr_dist = np.sqrt(curr_sumsq)

        return curr_dist

    def chaser_last_distance(self):
        target_point = self.chaser.docking_point
        if len(self.chaser.state_trace) < 2:
            return np.linalg.norm(self.chaser.state[:3] - target_point)
        last_pos = self.chaser.state_trace[-2][:3]

        last_sumsq = np.sum(np.square(target_point - last_pos))
        last_dist = np.sqrt(last_sumsq)

        return last_dist

    def is_closer(self):
        """
        if curr_dist < last_dist then curr_dist - last_dist
        represents forward progress
        """
        curr_dist = self.chaser_current_distance()
        last_dist = self.chaser_last_distance()

        if (curr_dist - last_dist) <= 0:
            return True
        else:
            return False

    def in_los(self):
        los_len = 800.0
        dock_pos = np.array(self.chaser.docking_point, copy=False)
        theta = self.chaser.theta_cone

        p = np.array(self.chaser.state[:3], dtype=np.float64, copy=False) - dock_pos

        #c_hat = dock_pos / np.linalg.norm(dock_pos)
        #c = c_hat * los_len
        c = np.array([0.0, 800, 0.0])
        #p dot c = mag p * mag c cos(theta)
        condition = np.cos(theta / 2.0)
        value = np.dot(p,c) / ( np.linalg.norm(p) * np.linalg.norm(c) )
        #value = np.dot(p,c) / (np.absolute(p) @ np.absolute(c))

        if p[1] < 0.0 or p[1] > 800:
            return False

        if value >= condition:
            return True
        return False

    def is_velocity_limit(self):
        state = np.array(self.chaser.state, copy=True)
        #print(f'testing state {self.chaser.state}')
        vel = state[3:]

        distance = self.chaser_current_distance()
        if distance < self.chaser.phase3_d:
            if np.linalg.norm(vel) > 0.05:
                 return False
        elif distance < self.chaser.slowzone_d:
            if np.linalg.norm(vel) > 8.0:
                print(f'failed chaser velocity {np.linalg.norm(vel)}')
                return False
        return True

    def in_phase3(self):
        dist = self.chaser_current_distance()
        if dist <= self.chaser.phase3_d:
            return True
        return False

    def in_slowzone(self):
        dist = self.chaser_current_distance()
        if dist <= self.chaser.slowzone_d and dist >= self.chaser.phase3_d:
            return True
        return False

    def in_time(self):
        """

        """
        if self.chaser.current_step <= self.time_limit:
            return True
        return False

    def l2norm_state(self):
        chaser_p = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        chaser_v = np.array(self.chaser.state[3:], dtype=np.float64, copy=True)
        #pos = self.chaser.docking_point - chaser_p
        pos = chaser_p - self.chaser.docking_point
        vel = chaser_v
        #print(f'chaser_p {chaser_p}')
        #print(f'difference {pos}')
        #print(f'vel {vel}')
        #state = np.concatenate((pos,vel))
        l2_norm_pos = np.linalg.norm(pos)
        l2_norm_vel = np.linalg.norm(vel)

        l2_norm_vel *= 5.0

        l2_norm_state = l2_norm_pos + l2_norm_vel
        #print(f'state {state}')
        #l2_norm_state = np.linalg.norm(state)
        #print(f'l2 norm {l2_norm}')
        return l2_norm_state

    def l2norm_p(self):
        chaser_p = np.array(self.chaser.state[:3], dtype=np.float64, copy=False)
        pos = chaser_p - self.chaser.docking_point
        return np.linalg.norm(pos)

    def l2norm_v(self):
        chaser_v = np.array(self.chaser.state[3:], dtype=np.float64, copy=False)
        vel = chaser_v
        return np.linalg.norm(vel)


    def is_docked(self):
        pos = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        vel = np.array(self.chaser.state[3:], dtype=np.float64, copy=True)
        if self.in_phase3() and self.in_los():
            pos_cm = (pos - self.chaser.docking_point) * 100.0
            vel_cm = vel * 100.0
            #print(f'position in centimeters {pos_cm}')
            #print(f'velocity in centimeters {vel_cm}')
            #if less than 20cm away and less than 1cm/s
            if np.linalg.norm(pos_cm) < 20 and np.linalg.norm(vel_cm) < 1:
                return True
        return False

    def docking_collision(self):
        pass

class reward_formulation(reward_conditions):

    def __init__(self, chaser):
        super().__init__(chaser)
        self.time_inlos = 0
        self.time_slowzone = 0
        self.time_inlos_slowzone = 0
        self.time_inlos_phase3 = 0
        self.entered_los =  False
        self.entered_phase3 = False

    def terminal_conditions(self):
        """
        check in bounds
        check target collision
        check timelimit

        *check obstacle collision

        return accumulated penality and done status
        """
        
        penality = 0
        done = False
        l2norm_p = super().l2norm_p()
        l2norm_v = super().l2norm_v()
        
        pos = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        docking_point = self.chaser.docking_point
        
        p = pos - docking_point
        
        
        if not super().in_time():
            print('mission time limit exceeded')
            penality = -1
            done = True
            return penality, done
            
        if l2norm_p < 100:
            #ensuring phase3 objectives
            if not super().in_los() or not l2norm_v < 0.05:
                penality -= 75
                done = True
                print("CHASER VIOLATING PHASE 3 CONSTRAINT")
                print(f"IN LOS:{super().in_los()} DIST:{l2norm_p} VEL:{l2norm_v}")
            #classify collision / no zone as terminal region
            elif p[1] < 0 and -50 > p[1]:
            	print("CHASER VIOLATING KEEP OUT COLLISION")
            	print(f"STATE:{pos}")
            	done = True
            	penality -= 100
            
                       
        return penality, done


    def quadratic_objective(self):

        x = self.chaser.state - np.concatenate((self.chaser.docking_point, np.array([0,0,0], dtype=np.float64)), axis=0)
        Q = np.array([[3, 0, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0, 0],
                      [0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 2]], dtype=np.float64)

        xTQ = x.T @ Q
        xTQx = xTQ @ x

        #print(f'quadratic result is {xTQx}')
        return xTQx

    def soft_penalities(self):
        """
        check if distance is not increased
        check if is not in los
        check if exceeding velocity limit
        """
        penality = 0
        l2norm_p = super().l2norm_p()
        l2norm_v = super().l2norm_v()

        
        #incentivising staying in LOS (per step)
        if not super().in_los() and self.entered_los == True:
            penality -= 1    
        return penality

    def soft_rewards(self):
        """
        check if closer
        check if los
        """
        #spare rewards priincple
        #0.99 100 steps with gae gamma * lambda 1/1-gamma*lambda
        #reward increases


        #increase gamma 0.999
        #initalize within 300m
        #control freq 1
        #episode time 1000
        reward = 0

        
        R = 50
        c_hat = np.array([0, 860, 0])
        pos = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        vel = np.array(self.chaser.state[3:], dtype=np.float64, copy=True)

        l2norm_p = super().l2norm_p()
        l2norm_v = super().l2norm_v()

        #evaluating centerness of LOS entrance (1 time)
        if super().in_los() and self.entered_los == False:
            self.entered_los = True
            LOS_Error = np.linalg.norm(pos - c_hat)
            reward += min(50, max(0, R - LOS_Error))
            
        #incentivising the phase 3 constraint (1 time)
        if l2norm_p < 100 and self.entered_phase3 == False:
            if super().in_los() and l2norm_v < 0.05:
                self.entered_phase3 = True
                reward += 50

        return reward

    def win_conditions(self):
        l2norm_p = super().l2norm_p()
        l2norm_v = super().l2norm_v()
        reward = 0 
        done = False
        
        if l2norm_p < 0.05 and l2norm_v < 0.01:
            print("CHASER DOCKED")
            pos = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
            print(f"STATE:{pos} DIST:{l2norm_p} VEL:{l2norm_v}")
            reward += 200
            done = True
            return reward, done
        return reward, done

    def reset_counts(self):
        self.time_inlos = 0
        self.time_slowzone = 0
        self.time_inlos_slowzone = 0
        self.time_inlos_phase3 = 0
        
    def reset_checkpoints(self):
        self.entered_los =  False
        self.entered_phase3 = False
