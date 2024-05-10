from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w,h,dt=1/30):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.kf.R = R
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h

        self.status = TrackStatus.Tentative
        self.is_new = True
        self.max_ios = 0.0
        self.in_pos = -1
        self.out_pos = -1

    def update(self, y, R):
        self.kf.update(y,R)

    def predict(self):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0,0] + logdet
    
    def update_pos(self, bb_det, bbs):
        pos = self.in_pos if self.is_new else self.out_pos
        for i in range(len(bbs)):
            cur_ios = self.ios(bb_det, bbs[i])
            if cur_ios > self.max_ios:
                self.max_ios = cur_ios
                pos = i
        
        self.max_ios *= 0.95
        if self.max_ios == 0.0:
            return False
        if self.is_new:
            self.in_pos = pos
            self.is_new = False
            return True
        self.out_pos = pos
        return False

    # intersection over self (bb_det)
    def ios(self, bb_det,bb_comp):
        x_l = max(bb_det[0], bb_comp[0])
        y_t = max(bb_det[1], bb_comp[1])
        x_r = min(bb_det[2], bb_comp[2])
        y_b = min(bb_det[3], bb_comp[3])

        if x_r < x_l or y_b < y_t:
            return 0.0
        i = (x_r - x_l) * (y_b - y_t)
        s = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
        return i/s