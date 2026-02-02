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
LIDAR_HEIGHT_FILTER = -0.30  # Filter points well below ground to catch low obstacles
                             # Original -0.2 / -0.15 were too strict and missed short obstacles


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

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
        
        # 用于停滞检测的位置历史
        self.position_history = []  # 存储最近的位置
        self.max_history_size = 10  # 保留最近10个位置
        self.stall_threshold = 0.02  # 停滞判定：10步内移动 < 0.02m

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

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
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        # Autolabor Pro1 uses /r1/cmd_vel for velocity commands
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        # Subscribe to a PointCloud2 topic. try several common candidates and
        # fall back to the default if none are currently published.
        candidates = [
            "/os_cloud_node/points",
            "/r1/os_cloud_node/points",
            "/velodyne_points",
            "/points_raw",
            "/velodyne_points_raw",
        ]
        chosen_topic = None
        try:
            published = [t for (t, _) in rospy.get_published_topics()]
            for c in candidates:
                if c in published:
                    chosen_topic = c
                    break
        except Exception:
            chosen_topic = None

        if chosen_topic is None:
            chosen_topic = candidates[0]
            rospy.logwarn("PointCloud2 topic not found among published topics; subscribing to default %s", chosen_topic)
        else:
            rospy.loginfo("Subscribing to PointCloud2 topic: %s", chosen_topic)

        self.lidar = rospy.Subscriber(chosen_topic, PointCloud2, self.lidar_callback, queue_size=1)
        
        # Odometry from Gazebo
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        
        # Wait for initial odometry and LiDAR data
        print("Waiting for ROS topics...")
        start_time = time.time()
        timeout = start_time + 10  # 10 second timeout
        while (self.last_odom is None or not self.lidar_received) and time.time() < timeout:
            elapsed = time.time() - start_time
            print(f"  Waiting... ({elapsed:.1f}s) - Odom: {'✓' if self.last_odom else '✗'} | LiDAR: {'✓' if self.lidar_received else '✗'}", end='\r', flush=True)
            time.sleep(0.5)
        
        print("\n")  # Clear the progress line
        if self.last_odom is None:
            print("Warning: Odometry data not received within timeout period")
        if np.all(self.lidar_data == 10):
            print("Warning: LiDAR data not received within timeout period")

    # Read Ouster LiDAR pointcloud and convert to distance data
    def lidar_callback(self, v):
        # Read points, skip NaNs to avoid invalid entries
        try:
            iterator = pc2.read_points(v, field_names=("x", "y", "z"), skip_nans=True)
        except Exception:
            return

        # reset to default far distances
        self.lidar_data = np.ones(self.environment_dim) * 10
        got = False
        for pt in iterator:
            try:
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
                continue
            if z <= LIDAR_HEIGHT_FILTER:
                continue
            # horizontal distance
            dist = math.hypot(x, y)
            if dist == 0:
                continue
            # use atan2 to compute angle in [-pi, pi]
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

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Check if we're shutting down
        if self._is_shutdown:
            raise RuntimeError("Environment is shutting down")

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Use timeout for ROS service calls to avoid hanging during shutdown
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)
            self.unpause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            if self._is_shutdown:
                raise RuntimeError("ROS shutdown during unpause_physics")
            print(f"/gazebo/unpause_physics service call failed: {e}")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=2.0)
            self.pause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            if self._is_shutdown:
                raise RuntimeError("ROS shutdown during pause_physics")
            print(f"/gazebo/pause_physics service call failed: {e}")

        # read LiDAR laser state
        done, collision, min_laser = self.observe_collision(self.lidar_data)
        # build a flat numeric state vector: [lidar( environment_dim ), distance, theta, v, w]
        v_state = self.lidar_data.copy()
        laser_state = v_state

        # Wait for odometry data to be available
        timeout = time.time() + 5  # 5 second timeout
        while self.last_odom is None and time.time() < timeout:
            time.sleep(0.01)
        
        if self.last_odom is None:
            raise RuntimeError("Odometry data not received within timeout period")

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        
        # 检测停滞碰撞（机器人被卡住）
        if not collision and self.check_stall_collision(self.odom_x, self.odom_y, action):
            collision = True
            done = True
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        
        # Calculate angular velocity change (anti-circle)
        angular_jerk = abs(action[1] - self.prev_angular)
        self.prev_angular = action[1]

        reward = self.get_reward(
            target, collision, action, min_laser, angular_jerk
        )

        robot_state = np.array([distance, theta, action[0], action[1]])
        state = np.concatenate((laser_state, robot_state))
        return state, reward, done, target

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        
        # 清除位置历史用于停滞检测
        self.position_history = []

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        # Wait for odometry data to be available
        timeout = time.time() + 5  # 5 second timeout
        while self.last_odom is None and time.time() < timeout:
            time.sleep(0.01)
        
        if self.last_odom is None:
            raise RuntimeError("Odometry data not received within timeout period")
        
        v_state = self.lidar_data.copy()

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = np.array([distance, theta, 0.0, 0.0])
        state = np.concatenate((v_state, robot_state))
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser
    
    def check_stall_collision(self, current_x, current_y, action):
        """
        检测机器人是否被卡住（停滞碰撞）
        即使激光数据显示前方清晰，但机器人无法移动
        """
        # 记录当前位置
        self.position_history.append((current_x, current_y))
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        
        # 需要至少 5 个历史位置才能判定
        if len(self.position_history) < 5:
            return False
        
        # 计算过去几步的位移
        start_pos = self.position_history[0]
        current_pos = self.position_history[-1]
        distance_moved = np.linalg.norm([current_pos[0] - start_pos[0], 
                                        current_pos[1] - start_pos[1]])
        
        # 如果命令了前进但几乎没有移动，判定为停滞
        if action[0] > 0.3 and distance_moved < self.stall_threshold:
            return True
        
        return False

    
    def get_reward(self, target, collision, action, min_laser, angular_jerk):

        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            r_jerk = -0.5 * angular_jerk
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 + r_jerk
    
    def print_collision_debug_info(self):
        """打印碰撞检测的调试信息"""
        min_laser = min(self.lidar_data)
        print(f"\n=== 碰撞检测调试信息 ===")
        print(f"最小激光距离: {min_laser:.3f}m")
        print(f"碰撞阈值: {COLLISION_DIST}m")
        print(f"碰撞状态: {'❌ 碰撞!' if min_laser < COLLISION_DIST else '✓ 安全'}")
        print(f"激光数据范围: min={min(self.lidar_data):.3f}, max={max(self.lidar_data):.3f}, mean={np.mean(self.lidar_data):.3f}")
        print(f"激光数据统计: {sum(1 for x in self.lidar_data if x < 1.0)} 个点 < 1.0m")
        print("=======================\n")
    def _shutdown_handler(self, signum, frame):
        """Handle SIGINT signal gracefully"""
        print("\nShutting down gracefully...")
        self._is_shutdown = True
        self._cleanup()
        exit(0)

    def _cleanup(self):
        """Clean up ROS resources properly"""
        if self._is_shutdown:
            return
        self._is_shutdown = True
        try:
            if hasattr(self, 'vel_pub'):
                # Stop the robot
                vel_cmd = Twist()
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
                time.sleep(0.1)
            
            # Unsubscribe from topics
            if hasattr(self, 'lidar'):
                self.lidar.unregister()
            if hasattr(self, 'odom'):
                self.odom.unregister()
            
            # Shutdown ROS node gracefully
            if not rospy.is_shutdown():
                rospy.signal_shutdown("Environment cleanup")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")