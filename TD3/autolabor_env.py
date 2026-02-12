import math
import os
import random
import subprocess
import time
import signal
import atexit
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import logging

# Suppress ROS TF_REPEATED_DATA warnings
logging.getLogger("rosgraph.xmlrpc").setLevel(logging.ERROR)
logging.getLogger("rosgraph").setLevel(logging.ERROR)
logging.getLogger("tf2").setLevel(logging.ERROR)

# Autolabor Pro1 specific parameters
# Pro1 dimensions: 0.7248m (L) x 0.5286m (W) x 0.195m (H)
# Half width = 0.2643m, so collision should trigger before hitting edges
GOAL_REACHED_DIST = 0.3     # Goal reached when within 0.3m
COLLISION_DIST = 0.57       # Collision when lidar < 0.55m
                            # Verified empirically: min_laser reaches ~0.45m on actual collision
                            # Set to 0.55m to reliably catch collisions with Ouster LiDAR
TIME_DELTA = 0.1

# LiDAR height filter for Ouster
# More aggressive filter to detect low obstacles
LIDAR_HEIGHT_FILTER = -0.55  # Filter points well below ground to catch low obstacles
                             # Original -0.2 / -0.15 were too strict and missed short obstacles


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8: goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2: goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3: goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2: goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2: goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2: goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5: goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5: goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5: goal_ok = False
    return goal_ok

class AutolaborEnv:
    """Gazebo environment for Autolabor Pro1 robot with Ouster LiDAR."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.lidar_data = np.ones(self.environment_dim) * 10
        self.lidar_received = False
        self.last_odom = None
        self.prev_angular = 0.0
        self._is_shutdown = False
        
        # 进程句柄初始化
        self.roscore_process = None
        self.launch_process = None
        
        # 用于停滞检测的位置历史
        self.position_history = []  # 存储最近的位置
        self.max_history_size = 10  # 保留最近10个位置
        self.stall_threshold = 0.02  # 停滞判定：10步内移动 < 0.02m

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        # 【修复1】保存进程句柄，放到单独进程组以便于统一终止
        try:
            self.roscore_process = subprocess.Popen(["roscore", "-p", port], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        except Exception:
            self.roscore_process = None
        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        
        # Register cleanup handler for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        atexit.register(self._cleanup)
        
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            self._cleanup()
            raise IOError("File " + fullpath + " does not exist")

        # 【修复2】保存进程句柄，放到单独进程组
        try:
            self.launch_process = subprocess.Popen(["roslaunch", "-p", port, fullpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        except Exception:
            self.launch_process = None
        print("Gazebo launched!")

        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        # 【修复3】自动检测雷达话题 (融合 V1 的智能与 V3 的逻辑)
        candidates = [
            "/os_cloud_node/points", "/r1/os_cloud_node/points",
            "/velodyne_points", "/points_raw", "/velodyne_points_raw",
        ]
        chosen_topic = None
        try:
            published = [t for (t, _) in rospy.get_published_topics()]
            for c in candidates:
                if c in published:
                    chosen_topic = c
                    break
        except Exception:
            pass
        
        if chosen_topic is None:
            chosen_topic = candidates[0] # Default fallback
            print(f"Warning: Auto-detect failed, using default: {chosen_topic}")
        else:
            print(f"Auto-detected LiDAR topic: {chosen_topic}")

        self.lidar = rospy.Subscriber(chosen_topic, PointCloud2, self.lidar_callback, queue_size=1)
        self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)
        
        print("Waiting for ROS topics...")
        start_time = time.time()
        timeout = start_time + 20 # Give Gazebo slightly more time
        while (self.last_odom is None or not self.lidar_received) and time.time() < timeout:
            elapsed = time.time() - start_time
            print(f"  Waiting... ({elapsed:.1f}s) - Odom: {'✓' if self.last_odom else '✗'} | LiDAR: {'✓' if self.lidar_received else '✗'}", end='\r', flush=True)
            time.sleep(0.5)
        
        print("\n")
        if self.last_odom is None:
            self._cleanup()
            raise RuntimeError("ERROR: Odometry not received. Check /r1/odom topic.")
        if not self.lidar_received:
            self._cleanup()
            raise RuntimeError(f"ERROR: LiDAR data not received on {chosen_topic}.")

    # V3 版本的高效数学计算逻辑
    def lidar_callback(self, v):
        try:
            iterator = pc2.read_points(v, field_names=("x", "y", "z"), skip_nans=True)
        except Exception:
            return

        self.lidar_data = np.ones(self.environment_dim) * 10
        got = False
        for pt in iterator:
            try:
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z): continue
            if z <= LIDAR_HEIGHT_FILTER: continue
            
            dist = math.hypot(x, y)
            if dist == 0: continue
            
            beta = math.atan2(y, x)
            for j in range(len(self.gaps)):
                if self.gaps[j][0] <= beta < self.gaps[j][1]:
                    self.lidar_data[j] = min(self.lidar_data[j], dist)
                    got = True
                    break

        if got:
            self.lidar_received = True

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        target = False
        if self._is_shutdown: raise RuntimeError("Environment is shutting down")

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)
            self.unpause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            if self._is_shutdown: raise RuntimeError("ROS shutdown during unpause")
            print(f"Unpause failed: {e}")

        time.sleep(TIME_DELTA)

        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=2.0)
            self.pause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            if self._is_shutdown: raise RuntimeError("ROS shutdown during pause")
            print(f"Pause failed: {e}")

        # V3 版本的严格同步逻辑：等待新一帧雷达数据
        if not self.lidar_received:
            for _ in range(20):
                if self.lidar_received: break
                time.sleep(0.01)
        
        done, collision, min_laser = self.observe_collision(self.lidar_data)
        v_state = self.lidar_data.copy()
        laser_state = v_state

        timeout = time.time() + 5
        while self.last_odom is None and time.time() < timeout:
            time.sleep(0.01)
        
        if self.last_odom is None: raise RuntimeError("Odometry timeout")

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        
        if not collision and self.check_stall_collision(self.odom_x, self.odom_y, action):
            collision = True
            done = True
            
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0: beta = -beta
            else: beta = 0 - beta
        theta = beta - angle
        if theta > np.pi: theta = np.pi - theta; theta = -np.pi - theta
        if theta < -np.pi: theta = -np.pi - theta; theta = np.pi - theta

        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        # 1. 计算距离引导 (Distance Guidance)
        if not hasattr(self, 'last_distance'):
            self.last_distance = distance

        # distance_rate > 0 代表靠近目标，< 0 代表远离目标
        distance_rate = self.last_distance - distance 
        self.last_distance = distance  # 更新 last_distance 供下一步使用
        
        angular_jerk = abs(action[1] - self.prev_angular)
        self.prev_angular = action[1]

        # reward = self.get_reward(target, collision, action, min_laser, angular_jerk)
        reward = self.get_reward(target, collision, action, min_laser, distance_rate)

        robot_state = np.array([distance, theta, action[0], action[1]])
        state = np.concatenate((laser_state, robot_state))
        return state, reward, done, target

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("Reset simulation failed")
        
        self.position_history = []

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0; y = 0; position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation = quaternion
        self.set_state.publish(object_state)

        self.odom_x = x; self.odom_y = y

        self.change_goal()
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try: self.unpause()
        except rospy.ServiceException: pass
        
        time.sleep(TIME_DELTA)
        
        rospy.wait_for_service("/gazebo/pause_physics")
        try: self.pause()
        except rospy.ServiceException: pass
        
        timeout = time.time() + 5
        while self.last_odom is None and time.time() < timeout: time.sleep(0.01)
        if self.last_odom is None: raise RuntimeError("Odometry timeout")
        
        v_state = self.lidar_data.copy()
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0: beta = -beta
            else: beta = 0 - beta
        theta = beta - angle
        if theta > np.pi: theta = np.pi - theta; theta = -np.pi - theta
        if theta < -np.pi: theta = -np.pi - theta; theta = np.pi - theta

        robot_state = np.array([distance, theta, 0.0, 0.0])
        state = np.concatenate((v_state, robot_state))
        return state

    def change_goal(self):
        if self.upper < 10: self.upper += 0.004
        if self.lower > -10: self.lower -= 0.004
        goal_ok = False
        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        for i in range(4):
            name = "cardboard_box_" + str(i)
            x = 0; y = 0; box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                if np.linalg.norm([x - self.odom_x, y - self.odom_y]) < 1.5: box_ok = False
                if np.linalg.norm([x - self.goal_x, y - self.goal_y]) < 1.5: box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.01
        marker.color.a = 1.0; marker.color.g = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST: return True, True, min_laser
        return False, False, min_laser
    
    def check_stall_collision(self, current_x, current_y, action):
        self.position_history.append((current_x, current_y))
        if len(self.position_history) > self.max_history_size: self.position_history.pop(0)
        if len(self.position_history) < 5: return False
        
        start_pos = self.position_history[0]
        current_pos = self.position_history[-1]
        dist = np.linalg.norm([current_pos[0]-start_pos[0], current_pos[1]-start_pos[1]])
        
        if action[0] > 0.3 and dist < self.stall_threshold: return True
        return False

    # def get_reward(self, target, collision, action, min_laser, angular_jerk):
    #     if target: return 100.0
    #     elif collision: return -100.0
    #     else:
    #         r3 = lambda x: 1 - x if x < 1 else 0.0
    #         return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 + (-0.5 * angular_jerk)
    
    def _shutdown_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self._is_shutdown = True
        self._cleanup()
        exit(0)

    def _cleanup(self):
        """Clean up ROS resources and subprocesses properly"""
        # Ensure cleanup runs only once
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        self._cleaned = True
        self._is_shutdown = True
        
        # 1. Stop Robot
        try:
            if hasattr(self, 'vel_pub'): self.vel_pub.publish(Twist())
        except Exception: pass
            
        # 2. Kill ROS connections
        try:
            if hasattr(self, 'lidar'): self.lidar.unregister()
            if hasattr(self, 'odom'): self.odom.unregister()
            if not rospy.is_shutdown(): rospy.signal_shutdown("Cleanup")
        except Exception: pass

        # 3. Kill Subprocesses (The Fix for Ctrl+C hanging)
        for proc, name in [(self.launch_process, "Gazebo/roslaunch"), (self.roscore_process, "Roscore")]:
            if proc:
                try:
                    pgid = os.getpgid(proc.pid)
                except Exception:
                    pgid = None
                print(f"Terminating {name}...")
                try:
                    # first try graceful termination of the whole process group
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGTERM)
                    else:
                        proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    try:
                        if pgid is not None:
                            os.killpg(pgid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except Exception:
                        pass
        # Additionally ensure gazebo clients/servers are killed (some installs spawn them
        # in ways not captured by roslaunch process groups)
        try:
            subprocess.call(["pkill", "-f", "gzclient"])
        except Exception:
            pass
        try:
            subprocess.call(["pkill", "-f", "gzserver"])
        except Exception:
            pass

        print("Cleanup complete.")


    def get_reward(self, target, collision, action, min_laser, distance_rate):
        if target:
            # 到达目标，给大奖
            return 100.0
        elif collision:
            # 撞墙，给大惩罚
            return -100.0
        else:
            # -----------------------------------------------------------
            # 根据 circuit.world 文件分析调整后的参数
            # 最窄处路宽约 0.67米，半宽 0.33米。
            # 因此 safe_distance 必须小于 0.33米，否则机器人无法通过。
            # -----------------------------------------------------------
            
            # 1. 距离奖励 (Distance Reward)
            # 保持足够的诱惑力，让它想往前走
            r_distance = 10.0 * distance_rate
            
            # 2. 避障惩罚 (Obstacle Penalty) - 核心修改
            # 设定安全阈值为 0.30米 (30厘米)
            safe_distance = 0.35
            
            r_obstacle = 0.0

            if min_laser < 0.25:
                r_obstacle = -20.0
            # if min_laser < safe_distance:
                # 进入 0.3米 危险区，开始指数级扣分
                # 离得越近，扣分越狠。
                # 公式：(安全距 - 当前距) / 安全距
                # 例如：当前 0.15米 -> (0.3-0.15)/0.3 = 0.5 -> 0.5 * 20 = -10分
                # r_obstacle = -20.0 * ((safe_distance - min_laser) / safe_distance)
                
                # 如果贴脸了 (小于 0.15米)，追加重罚，防止撞转角
                if min_laser < 0.15:
                    r_obstacle -= 10.0

            # 3. 速度/动作奖励 (Action Reward) - 解决“犹豫”
            # 只要它给出了前进速度 (action[0] > 0)，就给一点点微小的奖励
            # 这能抵消微小的计算误差，鼓励它动起来
            # r_action = action[0] * 0.2
            # r_action = -0.05 * abs(action[1])  # 罚角速度
            
            # 4. 固定的步数惩罚
            r_step = -0.1
            
            return r_distance + r_obstacle + r_step # + r_action