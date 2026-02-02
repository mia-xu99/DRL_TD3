#!/usr/bin/env python3
"""
诊断脚本：检查 Autolabor Pro1 的 ROS 话题
用于调试里程计和激光点云发布问题
"""

import subprocess
import time
import sys

def check_topics():
    """检查关键话题是否存在"""
    print("\n" + "="*70)
    print("Autolabor Pro1 ROS 话题诊断")
    print("="*70 + "\n")
    
    # 需要检查的话题
    expected_topics = {
        "/r1/cmd_vel": "速度命令话题",
        "/r1/odom": "里程计话题 (必须有!)",
        "/os_cloud_node/points": "Ouster 激光点云",
        "/os_cloud_node/imu": "Ouster IMU 数据",
        "/tf": "坐标变换",
        "/joint_states": "关节状态",
        "/gazebo/link_states": "Gazebo 链接状态",
    }
    
    print("🔍 检查 ROS 话题...\n")
    
    try:
        # 运行 rostopic list 命令
        result = subprocess.run(
            ["rostopic", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        available_topics = result.stdout.strip().split('\n')
        available_topics = [t.strip() for t in available_topics if t.strip()]
        
        print(f"找到 {len(available_topics)} 个话题:\n")
        
        # 检查每个预期的话题
        for topic, description in expected_topics.items():
            if topic in available_topics:
                print(f"  ✓ {topic:30s} - {description}")
            else:
                print(f"  ✗ {topic:30s} - {description}")
        
        # 显示所有找到的话题
        print(f"\n📋 所有可用话题：")
        for topic in sorted(available_topics):
            print(f"  • {topic}")
        
        # 检查关键话题
        print("\n" + "="*70)
        print("关键检查")
        print("="*70 + "\n")
        
        if "/r1/odom" in available_topics:
            print("✓ 里程计话题存在！")
            
            # 尝试查看里程计数据
            print("\n  尝试读取里程计数据...\n")
            try:
                result = subprocess.run(
                    ["rostopic", "echo", "-n", "1", "/r1/odom"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')[:10]  # 显示前 10 行
                    for line in lines:
                        if line.strip():
                            print(f"    {line}")
                    print("\n  ✓ 里程计数据正常发布")
            except subprocess.TimeoutExpired:
                print("  ✗ 无法读取里程计数据 (超时)")
        else:
            print("✗ 里程计话题不存在！")
            print("  可能原因：")
            print("  1. Gazebo 中的机器人模型未正确生成")
            print("  2. 差分驱动插件未正确加载")
            print("  3. pro1.urdf.xacro 中的插件配置有问题")
        
        if "/os_cloud_node/points" in available_topics:
            print("\n✓ Ouster 激光点云话题存在！")
        else:
            print("\n✗ Ouster 激光点云话题不存在")
            print("  这在 Gazebo 模拟中可能是正常的（取决于传感器配置）")
        
    except subprocess.TimeoutExpired:
        print("✗ rostopic list 超时 - ROS 可能未启动")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False
    
    print("\n" + "="*70 + "\n")
    return True

def check_robot_model():
    """检查 Gazebo 中是否加载了机器人模型"""
    print("🤖 检查机器人模型...\n")
    
    try:
        result = subprocess.run(
            ["rosservice", "call", "/gazebo/get_world_properties"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "autolabor_pro1" in result.stdout or "r1" in result.stdout:
            print("✓ 机器人模型已加载到 Gazebo")
        else:
            print("✗ 机器人模型可能未加载")
            print(f"  返回信息: {result.stdout[:200]}")
        
    except Exception as e:
        print(f"⚠️  无法检查模型: {e}")

if __name__ == "__main__":
    print("\n" + "⠿"*70)
    print("Autolabor Pro1 诊断工具")
    print("使用此脚本检查 ROS 连接和话题发布情况")
    print("⠿"*70)
    
    # 等待 ROS 启动
    print("\n⏳ 等待 ROS 完全启动...\n")
    time.sleep(3)
    
    # 运行诊断
    if check_topics():
        check_robot_model()
        print("\n✓ 诊断完成")
        sys.exit(0)
    else:
        print("\n✗ 诊断失败 - 请确保已运行:")
        print("   python train_autolabor_pro1.py")
        sys.exit(1)
