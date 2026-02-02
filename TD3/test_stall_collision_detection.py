#!/usr/bin/env python3
"""
停滞碰撞检测诊断脚本
用于测试机器人是否能检测到被卡住的情况（即使激光数据显示前方清晰）
"""

import time
import numpy as np
from autolabor_env import AutolaborEnv

def test_stall_detection():
    """测试停滞碰撞检测"""
    
    print("=" * 60)
    print("Autolabor Pro1 停滞碰撞检测测试")
    print("=" * 60)
    print("\n此测试将检测机器人是否能发现自己被卡住的情况")
    print("（即使激光数据显示前方没有障碍物）\n")
    print("=" * 60)
    
    # 初始化环境
    environment_dim = 20
    env = AutolaborEnv("autolabor_pro1_scenario.launch", environment_dim)
    print("\n初始化环境...")
    time.sleep(10)
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- 第 {episode} 局 ---")
            
            # 重置环境
            raw_state = env.reset()
            print(f"重置完成")
            
            # 运行一个回合
            steps = 0
            max_steps = 200
            
            while steps < max_steps:
                # 持续向前驾驶
                action = [0.5, 0.0]
                raw_next_state, reward, done, target = env.step(action)
                
                steps += 1
                min_laser = min(env.lidar_data)
                distance_moved = np.linalg.norm([
                    env.odom_x - env.position_history[0][0],
                    env.odom_y - env.position_history[0][1]
                ]) if len(env.position_history) > 0 else 0
                
                if steps % 20 == 0 or done:
                    pos_history_size = len(env.position_history)
                    collision_source = ""
                    if done and reward == -100.0:
                        if min_laser < 0.57:
                            collision_source = "🔴 激光碰撞"
                        else:
                            collision_source = "🟡 停滞碰撞"
                    
                    print(f"  步数 {steps}: min_laser={min_laser:.3f}m, pos_history={pos_history_size}, "
                          f"done={done}, reward={reward:.1f} {collision_source}")
                
                if done:
                    if reward == -100.0:
                        print(f"\n✓ 第 {steps} 步检测到碰撞!")
                        print(f"  激光最小距离: {min_laser:.3f}m")
                        if min_laser < 0.57:
                            print(f"  碰撞类型: 🔴 激光检测到障碍物")
                        else:
                            print(f"  碰撞类型: 🟡 停滞检测到卡住")
                    break
            
            if steps >= max_steps:
                print(f"\n⚠ 达到最大步数，未检测到碰撞")
                print(f"  最小激光距离: {min_laser:.3f}m")
            
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
    test_stall_detection()
