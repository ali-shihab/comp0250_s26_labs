Package:     cw2_team_13
License:     MIT
Description: ROS 2 coursework package for robotic pick-and-place and shape
             classification tasks using a Panda arm in Gazebo.

AUTHORS & CONTRIBUTION
----------------------
  Ali Shihab          <ali.shihab.25@ucl.ac.uk>       33%
  [Team member 2]     [email]                          33%
  [Team member 3]     [email]                          33%

SETUP & RUN INSTRUCTIONS
------------------------
Full setup and environment instructions are in the root README.md of the
repository (comp0250_s26_labs/README.md).

Quick start:
  cd ~/comp0250_s26_labs
  colcon build --packages-select cw2_team_13
  source install/setup.bash
  ros2 run cw2_team_13 cw2_node

TRIGGERING TASKS
----------------
With the simulation running, call the world spawner service:

  ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 1}"
  ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 2}"
  ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 3}"

TIME SPENT PER TASK (hours, approximate)
-----------------------------------------
  Task 1 (Pick and Place):  ~10 hours
  Task 2 (Shape Detection):  TBD
  Task 3 (Planning):         TBD
