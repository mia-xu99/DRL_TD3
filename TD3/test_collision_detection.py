#!/usr/bin/env python3
"""
碰撞检测诊断脚本
用于测试和调试 Autolabor Pro1 的碰撞检测是否正常工作
"""

import os
import time
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from autolabor_env import AutolaborEnv, COLLISION_DIST, LIDAR_HEIGHT_FILTER

def test_collision_detection():
    """测试碰撞检测功能"""
    
    print("=" * 60)
    print("Autolabor Pro1 碰撞检测测试")
    print("=" * 60)
    print(f"\n当前配置:")
    print(f"  碰撞阈值 (COLLISION_DIST): {COLLISION_DIST} m")
    print(f"  LiDAR 高度过滤 (LIDAR_HEIGHT_FILTER): {LIDAR_HEIGHT_FILTER} m")
    print("\n" + "=" * 60)
    
    # 初始化环境
    print("\n初始化环境...")
    environment_dim = 20
    env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
    
    print("\n等待环境就绪...")
    time.sleep(10)
    
    try:
        print("\n开始碰撞检测测试...")
        print("程序将：")
        print("  1. 向前移动")
        print("  2. 打印 LiDAR 数据")
        print("  3. 显示碰撞检测状态")
        print("\n按 Ctrl+C 停止测试\n")
        
        episode = 0
        while True:
            episode += 1
            print(f"\n--- 第 {episode} 局 ---")
            
            # 重置环境
            raw_state = env.reset()
            print(f"重置完成，初始状态维度: {len(raw_state)}")
            
            # 测试碰撞检测 - 持续向前直到碰撞
            collision_detected = False
            steps = 0
            max_steps = 200
            min_laser_history = []
            
            while steps < max_steps and not collision_detected:
                # 简单的前进动作
                action = [0.5, 0.0]  # 前进，不转向
                raw_next_state, reward, done, target = env.step(action)
                
                steps += 1
                min_laser = min(env.lidar_data)
                min_laser_history.append(min_laser)
                
                # 检查是否碰撞
                if steps % 20 == 0:  # 每20步打印一次信息
                    status = "❌ 碰撞!" if done else "✓ 安全"
                    print(f"  步数 {steps}: min_laser={min_laser:.3f}m (threshold={COLLISION_DIST}m), reward={reward:.1f}, done={done} {status}")
                
                if done:
                    collision_detected = True
                    print(f"\n✓ 第 {steps} 步检测到碰撞!")
                    print(f"  最小激光距离: {min_laser:.3f}m")
                    print(f"  碰撞阈值: {COLLISION_DIST}m")
                    print(f"  奖励: {reward}")
                    print(f"  最小激光历史: {[f'{x:.3f}' for x in min_laser_history[-5:]]}")
                    env.print_collision_debug_info()
            
            if steps >= max_steps:
                print(f"\n⚠ 达到最大步数 {max_steps}，未检测到碰撞")
                print(f"  最小激光距离: {min_laser:.3f}m")
                print(f"  碰撞阈值: {COLLISION_DIST}m")
                print(f"  比较结果: min_laser={min_laser:.3f} vs threshold={COLLISION_DIST}")
                print(f"  判断: min_laser < threshold = {min_laser < COLLISION_DIST}")
                print(f"  最小激光历史: {[f'{x:.3f}' for x in min_laser_history[-5:]]}")
                print(f"  最小值: {min(min_laser_history):.3f}m")
                env.print_collision_debug_info()
                
                # 诊断信息
                print(f"\n🔧 诊断信息:")
                if min_laser < COLLISION_DIST:
                    print(f"   ✓ 激光距离 < 阈值，但 done 仍为 False")
                    print(f"   这可能表示数据更新延迟或其他问题")
                else:
                    print(f"   ❌ 激光距离 ({min_laser:.3f}m) >= 阈值 ({COLLISION_DIST}m)")
                    print(f"   需要增加碰撞阈值或调整其他参数")
            
            # 暂停后继续下一局
            print("\n等待 2 秒后开始下一局...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n测试已停止")
    finally:
        print("\n清理资源...")
        env._cleanup()
        print("测试完成")

if __name__ == "__main__":
    test_collision_detection()

