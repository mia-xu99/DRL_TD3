# Pioneer3DX vs Autolabor Pro1 里程计配置对比

## Pioneer3DX 里程计发布原理

### 1. 文件结构
```
pioneer3dx.xacro (主文件)
  └─ pioneer3dx_body.xacro
      ├─ pioneer3dx_chassis.xacro     (底盘定义)
      ├─ pioneer3dx_wheel.xacro       (轮子定义)
      ├─ pioneer3dx_plugins.xacro     (插件定义) ← 关键！
      └─ 其他配置文件
```

### 2. 核心插件配置 (pioneer3dx_plugins.xacro)

```xml
<xacro:macro name="pioneer3dx_diff_drive">
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <robotNamespace></robotNamespace>
      <leftJoint>left_hub_joint</leftJoint>
      <rightJoint>right_hub_joint</rightJoint>
      <wheelSeparation>0.3</wheelSeparation>
      <wheelDiameter>0.18</wheelDiameter>
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <robotBaseFrame>base_link</robotBaseFrame>
      <odometrySource>world</odometrySource>
      <updateRate>50</updateRate>
    </plugin>
  </gazebo>
</xacro:macro>
```

### 3. 如何应用插件
在 pioneer3dx_body.xacro 中：
```xml
<!-- Motor plugin -->
<xacro:pioneer3dx_diff_drive />  ← 调用宏来添加插件
```

### 4. 里程计发布流程

```
Gazebo 差分驱动插件 (libgazebo_ros_diff_drive.so)
    ↓
监听 /r1/cmd_vel 话题 (速度命令)
    ↓
计算轮子转速和位置
    ↓
发布 /r1/odom 话题 (里程计数据)
    ↓
计算机器人位置、速度等信息
```

### 5. 关键参数说明

| 参数 | 值 | 说明 |
|-----|-----|-----|
| `leftJoint` | left_hub_joint | 左轮关节名称 |
| `rightJoint` | right_hub_joint | 右轮关节名称 |
| `wheelSeparation` | 0.3 | 两轮间距 (米) |
| `wheelDiameter` | 0.18 | 轮子直径 (米) |
| `commandTopic` | cmd_vel | 接收速度命令的话题 |
| `odometryTopic` | odom | 发布里程计的话题 |
| `robotNamespace` | (空) | 机器人命名空间 (空表示全局) |
| `odometrySource` | world | 以世界坐标系为参考 |
| `updateRate` | 50 | 更新频率 (Hz) |

---

## Autolabor Pro1 里程计配置

### 1. 配置位置
文件: `/catkin_ws/src/pja/urdf/pro1.urdf.xacro` (末尾)

### 2. 添加的插件配置

```xml
<!-- Gazebo plugins for simulation -->
<gazebo>
  <!-- Differential drive controller for two-wheel drive -->
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

### 3. 参数对比

| 参数 | Pioneer3DX | Autolabor Pro1 | 说明 |
|-----|-----------|----------------|-----|
| leftJoint | left_hub_joint | joint_left_front | 左轮关节名称 |
| rightJoint | right_hub_joint | joint_right_front | 右轮关节名称 |
| wheelSeparation | 0.3 | ${wheel_spacing_2 * 2} = 0.5286 | 两轮间距 |
| wheelDiameter | 0.18 | 0.254 | 轮子直径 |
| updateRate | 50 | 30 | 更新频率 |
| robotNamespace | (空) | /r1 | 命名空间 |
| commandTopic | cmd_vel | cmd_vel | 速度命令话题 |
| odometryTopic | odom | odom | 里程计话题 |

### 4. Autolabor Pro1 的关节信息

从 pro1.urdf.xacro 中提取：
```xml
<xacro:property name="wheel_spacing_2" value="0.2643" />  <!-- 单边宽度 -->

<joint name="joint_left_front" type="continuous">
  <parent link="base_link"/>
  <child link="left_front_wheel"/>
  ...
</joint>

<joint name="joint_right_front" type="continuous">
  <parent link="base_link"/>
  <child link="right_front_wheel"/>
  ...
</joint>
```

---

## 话题对比

### 发布的话题

| 话题 | Pioneer3DX | Autolabor Pro1 | 内容 |
|-----|-----------|----------------|-----|
| /odom | ✓ | ✓ | 里程计数据 (position, velocity) |
| /tf | ✓ | ✓ | 坐标变换 |
| /joint_states | ✓ | ✓ | 关节状态 |
| /cmd_vel | 监听 | 监听 | 速度命令 |

### 激光传感器话题

| 话题 | Pioneer3DX | Autolabor Pro1 |
|-----|-----------|----------------|
| /velodyne_points | ✓ (Velodyne VLP-16) | ✗ |
| /os_cloud_node/points | ✗ | ✓ (Ouster OS1-64) |

---

## 总结

### Pioneer3DX 方式
1. 在 `pioneer3dx_plugins.xacro` 中**定义宏** (代码复用)
2. 在 `pioneer3dx_body.xacro` 中**调用宏** (应用插件)
3. 使用 libgazebo_ros_diff_drive.so 发布 `/odom`

### Autolabor Pro1 方式 (已实现)
1. 直接在 `pro1.urdf.xacro` **末尾添加** `<gazebo>` 标签
2. 配置同样的 libgazebo_ros_diff_drive.so 插件
3. 发布 `/r1/odom` (带命名空间)

### 关键点
✓ 两个都使用相同的 Gazebo 插件: `libgazebo_ros_diff_drive.so`  
✓ 两个都通过差分驱动模型发布里程计  
✓ 主要区别在组织方式 (宏 vs 直接定义)  
✓ Autolabor Pro1 的配置已经完成，应该能正常工作  

---

如果还有里程计问题，请检查：
1. ✓ 差分驱动插件是否正确加载 (日志中应该能看到)
2. ✓ 关节名称是否匹配 (`joint_left_front`, `joint_right_front`)
3. ✓ 轮子参数是否合理 (直径 0.254m, 间距 0.5286m)
4. ✓ 机器人命名空间是否设置为 `/r1`
