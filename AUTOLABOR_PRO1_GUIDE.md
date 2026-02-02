# Autolabor Pro1 + Ouster LiDAR 整合说明

## 概述
本文档说明如何使用新创建的文件来支持 Autolabor Pro1 机器人和 Ouster LiDAR 传感器。

## 创建的文件

### 1. **Launch Files**

#### `/catkin_ws/src/multi_robot_scenario/launch/autolabor_pro1.gazebo.launch`
- **用途**: Gazebo 物理引擎专用启动文件
- **功能**:
  - 加载 `pja/urdf/pro1.urdf.xacro` (包含 Ouster LiDAR 配置)
  - 生成机器人模型到 Gazebo
  - 启动 robot_state_publisher 和 joint_state_publisher
- **参数**:
  - `robot_name`: 机器人名称，默认 `autolabor_pro1`
  - `robot_position`: 初始位置，默认原点

#### `/home/mia/DRL-robot-navigation/TD3/assets/autolabor_pro1_scenario.launch`
- **用途**: 高级启动文件（包装 empty_world 和 gazebo 启动）
- **功能**:
  - 启动 Gazebo empty world
  - 加载 autolabor_pro1.gazebo.launch
  - 启动 RVIZ 可视化
- **用法**: 在 Python 代码中指定 `autolabor_pro1_scenario.launch`

### 2. **Environment Class**

#### `/home/mia/DRL-robot-navigation/TD3/autolabor_env.py`
- **类名**: `AutolaborEnv`
- **替换对象**: `GazeboEnv`（原始的 velodyne_env.py）
- **关键改动**:

| 项目 | Pioneer3DX | Autolabor Pro1 |
|------|-----------|-----------------|
| 命令话题 | `/r1/cmd_vel` | `/r1/cmd_vel` (相同) |
| 激光点云话题 | `/velodyne_points` | `/os_cloud_node/points` |
| 里程计话题 | `/r1/odom` | `/r1/odom` (相同) |
| 传感器类别 | Velodyne | Ouster OS1-64 |
| 环境变量名 | `velodyne_data` | `lidar_data` |
| 回调函数 | `velodyne_callback` | `lidar_callback` |

### 3. **示例训练脚本**

#### `/home/mia/DRL-robot-navigation/TD3/train_autolabor_pro1.py`
- **用途**: 展示如何使用新环境的示例脚本
- **修改要点**:
  ```python
  # 旧代码（Pioneer3DX）:
  from velodyne_env import GazeboEnv
  env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
  
  # 新代码（Autolabor Pro1）:
  from autolabor_env import AutolaborEnv
  env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
  ```

## 修改现有代码

### 步骤 1: 更新导入语句
在所有训练脚本中（`temp_train_td3.py`, `train.py` 等）：

```python
# 改为：
from autolabor_env import AutolaborEnv

# 改为：
env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
```

### 步骤 2: 保持参数兼容
`AutolaborEnv` 与 `GazeboEnv` 的接口完全兼容：

```python
# 这些调用方式不变：
state = env.reset()
next_state, reward, done, target = env.step(action)
```

### 步骤 3: 状态维度
- 保持 `environment_dim = 20`（与 Pioneer3DX 相同）
- Ouster OS1-64 在 Gazebo 模拟中也生成 360 个扫描点

## ROS 话题参考

### Autolabor Pro1 在仿真中的话题

```
发布者 (Publishers):
- /r1/cmd_vel          : 速度命令 (geometry_msgs/Twist)
- /tf                  : 坐标变换
- /odom                : 里程计
- /odom_combined       : 融合里程计

订阅者 (Subscribers):
- /r1/odom            : 里程计数据 (nav_msgs/Odometry)
- /os_cloud_node/points : Ouster 激光点云 (sensor_msgs/PointCloud2)
```

## 运行训练

### 方式 1: 直接修改现有脚本
```python
# 在 temp_train_td3.py 中修改：
from autolabor_env import AutolaborEnv

# 并改这行：
env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)

# 运行：
python temp_train_td3.py
```

### 方式 2: 使用新的示例脚本
```bash
python train_autolabor_pro1.py
```

## 文件清单

| 文件路径 | 类型 | 用途 |
|---------|------|------|
| `catkin_ws/src/multi_robot_scenario/launch/autolabor_pro1.gazebo.launch` | Launch | Gazebo启动 |
| `TD3/assets/autolabor_pro1_scenario.launch` | Launch | 完整场景启动 |
| `TD3/autolabor_env.py` | Python | 环境类 |
| `TD3/train_autolabor_pro1.py` | Python | 示例训练脚本 |

## 常见问题

### Q: 如何同时支持 Pioneer3DX 和 Autolabor Pro1？
A: 保留两个环境类和两个 launch 文件，根据需要导入不同的环境。

### Q: Ouster 激光点云话题是否需要配置？
A: 不需要。`pro1.urdf.xacro` 已经配置了 Ouster 激光雷达，话题名称是固定的 `/os_cloud_node/points`。

### Q: 可以改变环境维度吗？
A: 可以，修改 `environment_dim` 参数即可。但推荐保持 20，与现有训练模型兼容。

### Q: 需要重新编译 ROS 包吗？
A: 需要。运行：
```bash
cd ~/DRL-robot-navigation/catkin_ws
catkin_make
```

## 测试命令

### 验证 Launch 文件语法
```bash
roslaunch multi_robot_scenario autolabor_pro1.gazebo.launch
```

### 查看可用的 ROS 话题
```bash
# 开启一个终端启动模拟
python train_autolabor_pro1.py &

# 在另一个终端查看话题
rostopic list | grep -E "(cmd_vel|odom|os_)"
```

## 后续优化

1. **激光点云处理优化**: 调整 `lidar_callback()` 中的角度范围和距离阈值
2. **碰撞检测参数**: 根据 Autolabor Pro1 的实际大小调整 `COLLISION_DIST`
3. **奖励函数调整**: 在 `get_reward()` 中针对新机器人特性优化奖励逻辑
4. **模型检查点**: 不能直接使用 Pioneer3DX 的预训练模型，需要重新训练

---

**创建日期**: 2026年1月29日
**对应机器人**: Autolabor Pro1
**传感器**: Ouster OS1-64 LiDAR
