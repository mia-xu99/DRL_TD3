# Pro3 里程计发布机制详解

## 快速答案

Pro3 发布里程计的方式与 Pioneer3DX 完全相同：

**通过在 Gazebo 配置文件中加载 `libgazebo_ros_diff_drive.so` 插件**

---

## Pro3 的实现结构

### 文件组织

```
pro3.xacro (主文件)
    ↓
<xacro:include filename="$(find pja)/urdf/turtlebot3_burger.gazebo.xacro"/>
    ↓
turtlebot3_burger.gazebo.xacro (Gazebo 配置)
    ↓
<plugin name="turtlebot3_burger_controller" 
        filename="libgazebo_ros_diff_drive.so">
    ↓
发布 /wheel_odom 话题
```

### 核心插件配置

文件：`turtlebot3_burger.gazebo.xacro`

```xml
<gazebo>
  <plugin name="turtlebot3_burger_controller" filename="libgazebo_ros_diff_drive.so">
    <commandTopic>/cmd_vel</commandTopic>
    <odometryTopic>wheel_odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <odometrySource>world</odometrySource>
    <publishOdomTF>true</publishOdomTF>
    <robotBaseFrame>base_footprint</robotBaseFrame>
    <publishWheelTF>false</publishWheelTF>
    <publishTf>true</publishTf>
    <publishWheelJointState>true</publishWheelJointState>
    <legacyMode>false</legacyMode>
    <updateRate>50</updateRate>
    <leftJoint>wheel_left_joint</leftJoint>
    <rightJoint>wheel_right_joint</rightJoint>
    <wheelSeparation>0.160</wheelSeparation>
    <wheelDiameter>0.066</wheelDiameter>
    <wheelAcceleration>100</wheelAcceleration>
    <wheelTorque>100</wheelTorque>
  </plugin>
</gazebo>
```

---

## Pro3 vs Pioneer3DX vs Autolabor Pro1 对比

### 三种方式的核心相同点

都使用：`libgazebo_ros_diff_drive.so` 插件

### 三种方式的组织差异

| 方式 | Pioneer3DX | Pro3 | Autolabor Pro1 |
|-----|-----------|------|----------------|
| **组织** | 分离宏 | 独立配置文件 | 直接嵌入 |
| **文件** | pioneer3dx_plugins.xacro | turtlebot3_burger.gazebo.xacro | pro1.urdf.xacro末尾 |
| **包含方式** | 宏调用 | 直接引入 | 无需引入 |
| **灵活性** | 高 (可复用) | 中 (独立文件) | 低 (写死了) |

### 参数对比

| 参数 | Pioneer3DX | Pro3 | Autolabor Pro1 |
|-----|-----------|------|----------------|
| leftJoint | left_hub_joint | wheel_left_joint | joint_left_front |
| rightJoint | right_hub_joint | wheel_right_joint | joint_right_front |
| wheelSeparation | 0.3 m | 0.16 m | 0.5286 m |
| wheelDiameter | 0.18 m | 0.066 m | 0.254 m |
| odometryTopic | odom | wheel_odom | odom |
| updateRate | 50 | 50 | 30 |
| robotNamespace | (空) | (空) | /r1 |

---

## Pro3 的关键特性

### 1. 话题名称

```xml
<commandTopic>/cmd_vel</commandTopic>
<odometryTopic>wheel_odom</odometryTopic>
```

**特点**: 里程计发布到 `wheel_odom`（不是 `/odom`）

### 2. 坐标系定义

```xml
<robotBaseFrame>base_footprint</robotBaseFrame>
<odometryFrame>odom</odometryFrame>
```

**特点**: 使用 `base_footprint` 作为基础框架（不是 `base_link`）

### 3. TF 发布

```xml
<publishOdomTF>true</publishOdomTF>
<publishTf>true</publishTf>
```

**特点**: 发布 TF 变换 `odom -> base_footprint`

### 4. 轮子参数

```xml
<leftJoint>wheel_left_joint</leftJoint>
<rightJoint>wheel_right_joint</rightJoint>
<wheelSeparation>0.160</wheelSeparation>
<wheelDiameter>0.066</wheelDiameter>
```

**特点**: 小轮子（TurtleBot3 Burger 机器人尺寸）

---

## Pro3 发布的话题

### 主要话题

```
/wheel_odom          ← 里程计数据 (nav_msgs/Odometry)
/tf                  ← 坐标变换 (包含 odom -> base_footprint)
/joint_states        ← 关节状态
/cmd_vel             ← 接收速度命令
/imu                 ← IMU 数据 (来自 imu_plugin)
```

---

## Pro3 的完整发布流程

```
用户发送: rostopic pub /cmd_vel ...
            ↓
turtlebot3_burger_controller 插件接收
            ↓
计算轮子转速
            ↓
从 Gazebo 物理引擎获取实时位置
            ↓
计算 position 和 velocity
            ↓
发布到 /wheel_odom 话题
            ↓
发布 TF: odom -> base_footprint
            ↓
Python 代码接收数据
            rospy.Subscriber("/wheel_odom", Odometry, callback)
```

---

## 如何在 Pro3 基础上修改为 Autolabor Pro1

### 已完成的修改：

在 `pro1.urdf.xacro` 末尾添加了类似的插件配置：

```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <alwaysOn>true</alwaysOn>
    <updateRate>30</updateRate>
    <leftJoint>joint_left_front</leftJoint>
    <rightJoint>joint_right_front</rightJoint>
    <wheelSeparation>${wheel_spacing_2 * 2}</wheelSeparation>
    <wheelDiameter>0.254</wheelDiameter>
    <torque>10.0</torque>
    <maxLinearSpeed>1.0</maxLinearSpeed>
    <maxAngularSpeed>2.0</maxAngularSpeed>
    <publishWheelTF>true</publishWheelTF>
    <publishWheelJointState>true</publishWheelJointState>
    <robotNamespace>/r1</robotNamespace>
    <odometryTopic>odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <robotBaseFrame>base_link</robotBaseFrame>
    <commandTopic>cmd_vel</commandTopic>
    <publishTf>1</publishTf>
    <odometrySource>world</odometrySource>
  </plugin>
</gazebo>
```

### 与 Pro3 的关键差异：

| 项目 | Pro3 | Autolabor Pro1 |
|-----|------|----------------|
| 插件名 | turtlebot3_burger_controller | diff_drive |
| odometryTopic | wheel_odom | odom |
| robotBaseFrame | base_footprint | base_link |
| robotNamespace | (空) | /r1 |
| wheelSeparation | 0.16 m | 0.5286 m |
| wheelDiameter | 0.066 m | 0.254 m |

---

## 总结：三种机器人的里程计发布方式

### 核心机制
✅ 全部都使用 `libgazebo_ros_diff_drive.so` 插件  
✅ 全部都监听 `cmd_vel` 话题  
✅ 全部都发布里程计话题  
✅ 全部都通过差分驱动模型计算  

### 组织方式
1. **Pioneer3DX**: 分离宏（最灵活）
2. **Pro3**: 独立配置文件（中等灵活）
3. **Autolabor Pro1**: 直接嵌入（最简单）

### 参数调整
每个机器人的参数不同，基于其实际物理尺寸

### 话题名称
- Pioneer3DX: `/odom`
- Pro3: `/wheel_odom`
- Autolabor Pro1: `/r1/odom`

**这些差异只是参数配置，核心发布机制完全相同！**

---

## 您现在可以理解的流程

```
Pioneer3DX 方式 (宏 → 复用)
    ↓
Pro3 方式 (独立文件 → 灵活)
    ↓
Autolabor Pro1 方式 (直接嵌入 → 简洁)
    ↓
所有方式都使用同一个 Gazebo 插件发布里程计
```

是不是豁然开朗了？😄
