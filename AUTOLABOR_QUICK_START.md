# Autolabor Pro1 快速开始指南

## 📋 已为您创建的文件

```
✓ /catkin_ws/src/multi_robot_scenario/launch/autolabor_pro1.gazebo.launch
✓ /TD3/assets/autolabor_pro1_scenario.launch
✓ /TD3/autolabor_env.py
✓ /TD3/train_autolabor_pro1.py
✓ /AUTOLABOR_PRO1_GUIDE.md (详细文档)
```

## 🚀 快速使用步骤

### 1️⃣ 编译 ROS 包
```bash
cd ~/DRL-robot-navigation/catkin_ws
catkin_make
```

### 2️⃣ 修改您的现有训练脚本

只需要改 **3 行代码**！

#### 原代码 (temp_train_td3.py):
```python
from velodyne_env import GazeboEnv

# ... 其他代码 ...

environment_dim = 20
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
```

#### 改为：
```python
from autolabor_env import AutolaborEnv

# ... 其他代码 ... (无需改动)

environment_dim = 20
env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
```

### 3️⃣ 运行训练
```bash
python temp_train_td3.py
# 或使用新的示例脚本：
python train_autolabor_pro1.py
```

## 📊 关键配置对比

| 配置项 | Pioneer3DX | Autolabor Pro1 |
|--------|-----------|-----------------|
| **环境类** | `GazeboEnv` | `AutolaborEnv` |
| **Launch文件** | `multi_robot_scenario.launch` | `autolabor_pro1_scenario.launch` |
| **激光话题** | `/velodyne_points` | `/os_cloud_node/points` |
| **速度话题** | `/r1/cmd_vel` | `/r1/cmd_vel` (同) |
| **里程计话题** | `/r1/odom` | `/r1/odom` (同) |
| **状态维度** | 24 (20+4) | 24 (20+4) |

## ✅ 验证安装

### 检查 Launch 文件
```bash
roslaunch multi_robot_scenario autolabor_pro1.gazebo.launch
```

### 查看话题
```bash
# 终端1: 启动环境
python train_autolabor_pro1.py &

# 终端2: 查看可用话题
rostopic list | grep -E "(cmd_vel|odom|os_cloud)"
```

## 🔧 API 兼容性

`AutolaborEnv` 与 `GazeboEnv` 完全兼容，无需改动其他代码：

```python
# 这些调用完全相同：
state = env.reset()
next_state, reward, done, target = env.step(action)
```

## 📝 需要修改的所有文件

使用 Find & Replace（Ctrl+Shift+H）替换以下内容：

### 对象: temp_train_td3.py
```
查找: from velodyne_env import GazeboEnv
替换为: from autolabor_env import AutolaborEnv

查找: env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
替换为: env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
```

### 对象: train.py
```
查找: from velodyne_env import GazeboEnv
替换为: from autolabor_env import AutolaborEnv

查找: env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
替换为: env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
```

### 对象: train_velodyne_td3.py、test_temp.py、test_velodyne_td3.py
```
同上操作
```

## 🐛 常见问题排查

### ❌ 错误: `ModuleNotFoundError: No module named 'autolabor_env'`
**解决**: 确保 `autolabor_env.py` 在 `/TD3/` 目录中
```bash
ls -la ~/DRL-robot-navigation/TD3/autolabor_env.py
```

### ❌ 错误: `FileNotFoundError: File .../autolabor_pro1_scenario.launch does not exist`
**解决**: 
1. 确保文件存在：
```bash
ls -la ~/DRL-robot-navigation/TD3/assets/autolabor_pro1_scenario.launch
```
2. 确保已编译 ROS 包：
```bash
cd ~/DRL-robot-navigation/catkin_ws && catkin_make
```

### ❌ 错误: `No module named '/os_cloud_node/points'`
**解决**: Ouster 激光话题已在 URDF 中配置，自动发布。检查 Gazebo 是否正确启动。

## 📚 详细文档

查看: [AUTOLABOR_PRO1_GUIDE.md](AUTOLABOR_PRO1_GUIDE.md)

---

**需要帮助？** 按照快速步骤 1-3 操作即可！
