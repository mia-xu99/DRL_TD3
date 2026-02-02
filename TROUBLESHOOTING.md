# Autolabor Pro1 ROS 连接故障排除指南

## 问题：里程计话题未发布

### 症状
```
Warning: Odometry data not received within timeout period
```

---

## 诊断步骤

### 步骤 1: 清理旧进程
```bash
pkill -f roscore || true
pkill -f gazebo || true
sleep 2
```

### 步骤 2: 重新编译 ROS 包
```bash
cd ~/DRL-robot-navigation/catkin_ws
catkin_make
source devel/setup.bash
```

### 步骤 3: 查看生成的 URDF
```bash
# 验证 pro1.urdf.xacro 是否正确解析
cd ~/DRL-robot-navigation/catkin_ws/src/pja/urdf
xacro pro1.urdf.xacro > /tmp/pro1_expanded.urdf

# 检查是否有错误
echo $?  # 应该返回 0
```

### 步骤 4: 启动诊断脚本

**终端 1** - 启动训练环境:
```bash
cd ~/DRL-robot-navigation/TD3
python train_autolabor_pro1.py &
sleep 10  # 等待 Gazebo 启动
```

**终端 2** - 运行诊断:
```bash
cd ~/DRL-robot-navigation/TD3
python diagnose_topics.py
```

---

## 常见问题和解决方案

### ❌ 问题 1: "resource not found: ouster_description"

**原因**: 缺少 Ouster 描述包

**解决**:
```bash
# 安装 ouster_description (如果还未安装)
# 或使用已有的配置

# 如果 ouster 包不可用，可以禁用它：
# 修改 pro1.urdf.xacro，注释掉 ouster 行：
# <xacro:include filename="$(find ouster_description)/urdf/OS1-64.urdf.xacro"/>
```

### ❌ 问题 2: "/r1/odom 话题未发布"

**检查点**:
1. Gazebo 中是否看到机器人模型？
   ```bash
   # 在另一个终端检查
   rostopic list | grep -i odom
   ```

2. Gazebo 插件是否加载？
   ```bash
   # 查看 Gazebo 日志
   tail -f ~/.ros/log/**/gazebo*.log | grep -i "differential\|odom\|plugin"
   ```

3. 关节名称是否正确？
   ```bash
   # 查看 URDF 中的关节
   xacro ~/DRL-robot-navigation/catkin_ws/src/pja/urdf/pro1.urdf.xacro | grep "joint name"
   ```

**解决**:
确保 pro1.urdf.xacro 末尾的插件配置中：
```xml
<leftJoint>joint_left_front</leftJoint>
<rightJoint>joint_right_front</rightJoint>
```

这些关节必须在 URDF 中存在！

### ❌ 问题 3: "libgazebo_ros_diff_drive.so not found"

**检查**:
```bash
find /opt/ros -name "libgazebo_ros_diff_drive.so" 2>/dev/null
```

**解决**:
```bash
# 安装 gazebo_ros 包
sudo apt-get install ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-gazebo-plugins
```

### ❌ 问题 4: 激光点云话题未发布

**检查**:
```bash
# 验证 Ouster 配置
xacro ~/DRL-robot-navigation/catkin_ws/src/pja/urdf/pro1.urdf.xacro | grep -A5 "os_cloud_node"
```

**注意**: 如果 Ouster 传感器配置有问题，可以先禁用它，只关注里程计工作。

---

## 验证步骤

### 1️⃣ 验证里程计发布

```bash
# 等待环境启动后，运行：
rostopic echo -n 5 /r1/odom

# 应该看到类似输出：
# header:
#   seq: 1
#   stamp:
#     secs: ...
#   frame_id: "odom"
# child_frame_id: "base_link"
# pose:
#   pose:
#     position:
#       x: ...
#       y: ...
#       z: ...
```

### 2️⃣ 验证速度命令接收

```bash
# 发送测试速度命令
rostopic pub /r1/cmd_vel geometry_msgs/Twist -r 10 \
  '{linear: {x: 0.5, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0.2}}'

# 观察 Gazebo 中机器人是否移动
# 再观察 /r1/odom 是否更新
```

### 3️⃣ 查看坐标变换

```bash
# 查看 TF 树
rosrun tf view_frames

# 应该看到 odom -> base_link 的变换
```

---

## 调试命令集合

```bash
# 1. 列出所有话题
rostopic list

# 2. 查看特定话题类型
rostopic type /r1/odom

# 3. 监视话题信息
rostopic info /r1/odom

# 4. 实时显示话题数据
rostopic echo /r1/odom

# 5. 测量话题频率
rostopic hz /r1/odom

# 6. 查看 TF 树
rosrun tf view_frames

# 7. 查看两个帧之间的变换
rosrun tf tf_echo odom base_link

# 8. 查看 Gazebo 日志
tail -f ~/.ros/log/**/gazebo*.log

# 9. 获取 Gazebo 物理引擎信息
rosservice list | grep gazebo

# 10. 检查机器人模型
xacro ~/DRL-robot-navigation/catkin_ws/src/pja/urdf/pro1.urdf.xacro
```

---

## 最终检查清单

- [ ] Gazebo 正常启动 (能看到 empty_world)
- [ ] 机器人模型出现在 Gazebo 中
- [ ] `/r1/odom` 话题存在
- [ ] `/r1/odom` 话题有数据发布 (频率 > 0)
- [ ] `/tf` 话题中有 odom -> base_link 变换
- [ ] 发送 `/r1/cmd_vel` 后机器人移动
- [ ] 机器人移动时 `/r1/odom` 数据更新

如果以上全部通过，环境应该能正常使用！

---

## 快速修复方案

如果问题持续存在，尝试以下方案：

### 方案 A: 使用简化的 URDF (禁用 Ouster)
```bash
cp /home/mia/DRL-robot-navigation/catkin_ws/src/pja/urdf/pro1.urdf.xacro \
   /home/mia/DRL-robot-navigation/catkin_ws/src/pja/urdf/pro1.urdf.xacro.bak

# 编辑 pro1.urdf.xacro，注释掉 Ouster 行：
# <xacro:include filename="$(find ouster_description)/urdf/OS1-64.urdf.xacro"/>
# <xacro:OS1-64 parent="base_link" name="os_sensor" ...>
```

### 方案 B: 回退到 Pioneer3DX
```bash
# 临时回到 Pioneer3DX 验证环境是否正常
# 在 train_autolabor_pro1.py 中改为：
# from velodyne_env import GazeboEnv
# env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
```

---

**需要帮助？** 收集以下信息：
1. 完整的错误日志（从 ~/.ros/log 中）
2. `xacro pro1.urdf.xacro` 的输出
3. `rostopic list` 的完整输出
4. Gazebo 窗口的截图
