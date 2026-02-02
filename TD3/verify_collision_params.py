#!/usr/bin/env python3
"""
快速参数验证脚本
检查碰撞检测参数是否正确设置
"""

from autolabor_env import COLLISION_DIST, LIDAR_HEIGHT_FILTER

print("=" * 50)
print("参数验证")
print("=" * 50)

# 检查碰撞阈值
print(f"\n✓ 碰撞阈值 (COLLISION_DIST): {COLLISION_DIST} m")
if COLLISION_DIST >= 0.50:
    print("  ✓ 阈值正确设置 (>= 0.50m)")
else:
    print(f"  ❌ 阈值太低 (当前: {COLLISION_DIST}m, 应该 >= 0.50m)")

# 检查高度过滤
print(f"\n✓ LiDAR 高度过滤 (LIDAR_HEIGHT_FILTER): {LIDAR_HEIGHT_FILTER} m")
if LIDAR_HEIGHT_FILTER <= -0.15:
    print("  ✓ 过滤器正确设置 (<= -0.15m)")
else:
    print(f"  ❌ 过滤器太宽松 (当前: {LIDAR_HEIGHT_FILTER}m, 应该 <= -0.15m)")

print("\n" + "=" * 50)
print("验证完成")
print("=" * 50)
