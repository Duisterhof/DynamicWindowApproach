#!/usr/bin/env python3

import time

import numpy as np
import cv2
import dwa
from numpy import random
import imageio

class Demo(object):
    def __init__(self):
        cv2.namedWindow('cvwindow')
        cv2.setMouseCallback('cvwindow', self.callback)

        # setting initial setting for vehicle and environment
        self.reset() 

        # Planner Settings
        self.vel = (0.0, 0.0)
        self.pose = (-30.0, 30.0, 0)
        self.goal = None
        self.base = [-1.5, -1.25, +1.5, +1.25]
        self.config = dwa.Config(
                max_speed = 3.0,
                min_speed = -1.0,
                max_yawrate = np.radians(40.0),
                max_accel = 15.0,
                max_dyawrate = np.radians(110.0),
                velocity_resolution = 0.1,
                yawrate_resolution = np.radians(1.0),
                dt = 0.1,
                predict_time = 3.0,
                heading = 0.15,
                clearance = 1.0,
                velocity = 1.0,
                base = self.base)

    # resets env and robot placement
    def reset(self):
        self.placed_robot = False
        self.pose = (-30.0, 30.0, 0)
        self.env_num = random.randint(100)
        self.bin_map = imageio.imread('imgs/rand_env_'+str(self.env_num)+'.png')
        bin_shape = np.shape(self.bin_map)
        conversion_factor = 600./(bin_shape[0])

        self.point_cloud = []
        self.draw_points = []

        for i in range(bin_shape[0]):
            for j in range(bin_shape[1]):
                if np.all(self.bin_map[i][j] == 255):
                    self.point_cloud.append([j*conversion_factor/10,i*conversion_factor/10])
                    self.draw_points.append([int(j*conversion_factor),int(i*conversion_factor)])

        self.point_cloud = self.point_cloud[::3]
        self.draw_points = self.draw_points[::3]

        self.placed_robot = False
        self.goal = None

    # mouse callback
    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.placed_robot:
                self.pose = (x/10,y/10,0)
                self.placed_robot = True
            else:
                self.goal = (x/10,y/10)            


    def main(self):
        import argparse
        parser = argparse.ArgumentParser(description='DWA Demo')
        parser.add_argument('--save', dest='save', action='store_true')
        parser.set_defaults(save=False)
        args = parser.parse_args()
        if args.save:
            import imageio
            writer = imageio.get_writer('./dwa.gif', mode='I', duration=0.05)
        while True:
            prev_time = time.time()
            self.map = np.zeros((600, 600, 3), dtype=np.uint8)
            self.scaled_bin = cv2.resize(self.bin_map,(600,600),interpolation=cv2.INTER_AREA)
            self.map += self.scaled_bin

            if self.goal is not None:
                cv2.circle(self.map, (int(self.goal[0]*10), int(self.goal[1]*10)),
                        4, (0, 255, 0), -1)
                if len(self.point_cloud):
                    # Planning
                    self.vel = dwa.planning(self.pose, self.vel, self.goal,
                            np.array(self.point_cloud, np.float32), self.config)
                    # Simulate motion
                    self.pose = dwa.motion(self.pose, self.vel, self.config.dt)

            pose = np.ndarray((3,))
            pose[0:2] = np.array(self.pose[0:2]) * 10
            pose[2] = self.pose[2]

            base = np.array(self.base) * 10
            base[0:2] += pose[0:2]
            base[2:4] += pose[0:2]

            # Not the correct rectangle but good enough for the demo
            width = base[2] - base[0]
            height = base[3] - base[1]
            rect = ((pose[0], pose[1]), (width, height), np.degrees(pose[2]))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.map,[box],0,(0,0,255),-1)

            # Prevent divide by zero
            fps = int(1.0 / (time.time() - prev_time + 1e-10))
            cv2.putText(self.map, f'FPS: {fps}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(self.map, f'Point Cloud Size: {len(self.point_cloud)}',
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if not self.placed_robot:
                cv2.putText(self.map, f'Click to place robot.',
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(self.map, f'Click to place goal.',
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if args.save:
                writer.append_data(self.map)
            cv2.imshow('cvwindow', self.map)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):
                self.reset()
        if args.save:
            writer.close()

if __name__ == '__main__':
    Demo().main()
