#!/bin/bash
set -e
cd ~/comp0250_s26_labs
echo "=== killing any leftover ROS/Gazebo procs ==="
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true
pkill -9 -f rviz2 2>/dev/null || true
pkill -9 -f move_group 2>/dev/null || true
pkill -9 -f cw2_solution_node 2>/dev/null || true
pkill -9 -f world_spawner 2>/dev/null || true
sleep 1
echo "=== sourcing ROS ==="
source /opt/ros/humble/setup.bash
echo "=== building cw2_team_13 ==="
colcon build --packages-select cw2_team_13 --symlink-install
BUILD_RC=$?
echo "=== build exit code: $BUILD_RC ==="
if [ "$BUILD_RC" -ne 0 ]; then
  echo "=== BUILD FAILED ==="
  exit 1
fi
source install/setup.bash
echo "=== relaunching cw2 solution ==="
export PATH=/usr/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
exec ros2 launch cw2_team_13 run_solution.launch.py use_gazebo_gui:=true use_rviz:=true
