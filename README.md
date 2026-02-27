# comp0250_s26_labs
Labs and coursework code for COMP0250

Currently contains the code for coursework 1. Coursework 2 will follow.

In your home directory (locally or on the GPU servers) type
```
git clone https://github.com/surgical-vision/comp0250_s26_labs.git
```

If you've done this previously do the following to pull code that has been added (in the master branch):

```
cd ~/comp0250_s26_labs
git pull origin master
```

## ROS 2 dependency install (required once)

```bash
./scripts/install_ros2_deps.sh
sudo apt-get install -y \
  ros-humble-moveit \
  ros-humble-moveit-core \
  ros-humble-moveit-ros-planning-interface
sudo apt-get install -y \
  ros-humble-pcl-ros \
  ros-humble-pcl-conversions \
  ros-humble-point-cloud-transport
sudo apt-get install -y ros-humble-gazebo-ros2-control
sudo apt-get install -y python3-catkin-pkg python3-catkin-pkg-modules
```

Optional camera-processing packages (not needed for current baseline):

```bash
sudo apt-get install -y \
  ros-humble-image-proc \
  ros-humble-depth-image-proc
```

If you use Conda, run `conda deactivate` before ROS2 build/launch.


## ROS 2 build

```bash
cd ~/comp0250_S26_labs
source /opt/ros/humble/setup.bash
colcon build --mixin release
source install/setup.bash
```


**YOU MUST RENAME THE DIRECTORY cw1_team_x to your team number and amend the following command appropriately**
## ROS 2 launch: `cw1_team_x`

Use this command to run your solution to the coursework:

```bash
cd ~/comp0250_S26_labs
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
ros2 launch cw1_team_x run_solution.launch.py \
  use_gazebo_gui:=true use_rviz:=true \
  enable_realsense:=true enable_camera_processing:=false \
  control_mode:=effort
```
## ROS 2 command: Task 1

```bash
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 1}"
```

Task 1 scenario 2 only:

```bash
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 111}"
```

## ROS 2 test command: Task 2

```bash
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 2}"
```

## ROS 2 test command: Task 3

```bash
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 3}"



