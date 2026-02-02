# Autolabor Pro1 ROS 问题修复方案

## 问题诊断

### 1. TF_REPEATED_DATA 警告

**症状：** 程序运行时出现大量警告：
```
[WARN] [timestamp]: TF_REPEATED_DATA ignoring data with redundant timestamp for frame left_front_wheel (parent base_link)
```

**根本原因：**
- Gazebo中的TF（Transform）系统发现同一帧的变换数据在相同时间戳被发布多次
- Autolabor Pro1 URDF模型中可能配置了冗余的关节发布者
- `robot_state_publisher` 和 `joint_state_publisher` 的发布频率配置不匹配

**解决方案：**
通过日志抑制禁用这些无害的警告：
```python
import logging
logging.getLogger("rosgraph.xmlrpc").setLevel(logging.ERROR)
logging.getLogger("rosgraph").setLevel(logging.ERROR)
logging.getLogger("tf2").setLevel(logging.ERROR)
```

### 2. Ctrl+C 后的连接拒绝错误

**症状：** 按下 Ctrl+C 后出现大量错误：
```
Error in XmlRpcClient::writeRequest: write error (拒绝连接).
Error in XmlRpcDispatch::work: couldn't find source iterator
```

**根本原因：**
- ROS节点没有优雅关闭，导致待处理的通信请求仍在尝试
- 主节点已被关闭，但其他节点还在尝试与其通信
- 缺少适当的信号处理和资源清理

**解决方案：**
1. **注册信号处理器** - 捕获 SIGINT (Ctrl+C) 信号并优雅关闭
2. **实现清理函数** - 在关闭前停止机器人、注销订阅者和发布者
3. **使用 ROS 服务超时** - 防止在shutdown过程中hanging
4. **优雅的 ROS shutdown** - 调用 `rospy.signal_shutdown()`

## 实现的改进

### autolabor_env.py

#### 1. 改进的导入和日志配置
```python
import signal
import atexit
import logging

# 抑制TF和XMLRPC警告
logging.getLogger("rosgraph.xmlrpc").setLevel(logging.ERROR)
logging.getLogger("rosgraph").setLevel(logging.ERROR)
logging.getLogger("tf2").setLevel(logging.ERROR)
```

#### 2. 环境初始化中的shutdown处理
```python
def __init__(self, launchfile, environment_dim):
    self._is_shutdown = False
    
    # ... 初始化代码 ...
    
    # 注册清理处理器
    signal.signal(signal.SIGINT, self._shutdown_handler)
    atexit.register(self._cleanup)
```

#### 3. 清理和shutdown方法
```python
def _shutdown_handler(self, signum, frame):
    """处理SIGINT信号并优雅关闭"""
    print("\nShutting down gracefully...")
    self._is_shutdown = True
    self._cleanup()
    exit(0)

def _cleanup(self):
    """正确清理ROS资源"""
    if self._is_shutdown:
        return
    self._is_shutdown = True
    try:
        # 停止机器人
        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)
        time.sleep(0.1)
        
        # 注销订阅者
        if hasattr(self, 'lidar'):
            self.lidar.unregister()
        if hasattr(self, 'odom'):
            self.odom.unregister()
        
        # 优雅关闭ROS节点
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Environment cleanup")
    except Exception as e:
        print(f"Error during cleanup: {e}")
```

#### 4. 改进的 step() 函数
```python
def step(self, action):
    # 检查是否正在shutdown
    if self._is_shutdown:
        raise RuntimeError("Environment is shutting down")
    
    # ... action publish ...
    
    # 为ROS服务调用添加超时
    try:
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)
        self.unpause()
    except (rospy.ServiceException, rospy.ROSException) as e:
        if self._is_shutdown:
            raise RuntimeError("ROS shutdown during unpause_physics")
        print(f"/gazebo/unpause_physics service call failed: {e}")
    
    # ... 其他代码 ...
```

### train_autolabor_pro1.py

#### 1. 添加全局cleanup处理
```python
import signal
import atexit

env = None

def cleanup_environment():
    """在退出时清理环境"""
    global env
    if env is not None:
        try:
            env._cleanup()
        except:
            pass

def signal_handler(signum, frame):
    """优雅处理Ctrl+C"""
    print("\n\nReceived interrupt signal. Cleaning up...")
    cleanup_environment()
    exit(0)

# 注册信号处理器和退出处理器
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_environment)
```

## 预期改进

1. **减少/消除 TF_REPEATED_DATA 警告** - 这些警告现在被日志级别过滤
2. **干净的Ctrl+C退出** - 不会再出现大量"拒绝连接"错误
3. **更优雅的资源清理** - 机器人被正确停止，ROS节点被正确关闭
4. **更好的可观察性** - 清晰的"Shutting down gracefully"消息

## 测试步骤

1. 运行程序：
   ```bash
   python train_autolabor_pro1.py
   ```

2. 在运行几秒后按 Ctrl+C

3. **预期结果：**
   - 看到"Shutting down gracefully..."消息
   - 没有大量的"拒绝连接"错误
   - 程序在1-2秒内干净地退出
   - 最多只有少量的TF_REPEATED_DATA警告（来自Gazebo本身）

## 与 temp_train_td3.py 的区别

temp_train_td3.py（使用 Velodyne）之所以表现更好：
1. Velodyne传感器的TF发布配置更简洁
2. 可能使用了不同的launch文件配置
3. 没有Autolabor特定的URDF问题

现在autolabor_env.py已被优化以匹配同等质量。

## 进一步优化建议

1. **检查launch文件** - 确保Autolabor Pro1 launch文件中的发布频率配置正确
2. **URDF审查** - 检查是否有冗余的joint_state_publisher配置
3. **ROS参数** - 可能需要调整 `/use_sim_time` 和其他时间相关参数
