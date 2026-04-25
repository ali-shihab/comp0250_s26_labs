# COMP0250 Coursework 2 — Team 13

This repository is the Team 13 fork of `surgical-vision/comp0250_s26_labs`. It contains
our solution package `cw2_team_13` for **Task 1 (Pick and Place)** of CW2.

## Authors

| Author       | Email                        | Role                          |
| ------------ | ---------------------------- | ----------------------------- |
| Ali Shihab   | ali.shihab.25@ucl.ac.uk      | Completed Task 1              |

## License

MIT (matches the upstream `surgical-vision/comp0250_s26_labs` and the per-package
`license` field in `cw2_team_13/package.xml`).

## Environment

Tested on Ubuntu 22.04 / ROS 2 Humble running under an OrbStack x86 VM on a MacBook Pro
M1 Pro (Rosetta), with the display forwarded over XRDP. Gazebo Classic for simulation.

## Build

One-time dependency install (already covered by the upstream's
`scripts/install_ros2_deps.sh`, plus the MoveIt / PCL / gazebo-ros2-control packages):

```bash
./scripts/install_ros2_deps.sh
sudo apt-get install -y \
  ros-humble-moveit ros-humble-moveit-core ros-humble-moveit-ros-planning-interface \
  ros-humble-pcl-ros ros-humble-pcl-conversions \
  ros-humble-gazebo-ros2-control
```

Build:

```bash
cd ~/comp0250_s26_labs
source /opt/ros/humble/setup.bash
colcon build --packages-select cw2_team_13 --symlink-install
source install/setup.bash
```

A convenience script `rebuild_task1.sh` at the repo root kills any leftover Gazebo /
move_group processes, rebuilds `cw2_team_13`, and relaunches the solution.

## Run

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
ros2 launch cw2_team_13 run_solution.launch.py \
  use_gazebo_gui:=true use_rviz:=false
```

Trigger Task 1 from a second terminal once the simulation is up:

```bash
ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 1}"
```

The world spawner randomises the shape (`nought` / `cross`), its position, its orientation
(when `T1_ANY_ORIENTATION` is `True`) and the basket location each run.

---

## Task 1: Pick and Place — Implementation

### Overview

The solution lives in `src/courseworks/cw2_team_13/{src,include}/cw2_class.{cpp,h}`,
launched as a single ROS 2 node by `launch/run_solution.launch.py`. The node:

* Owns two `MoveGroupInterface`s (`panda_arm` for arm motion, `hand` retained but unused
  for actuation — see "Gripper control bypass" below).
* Subscribes to a `PointCloud2` topic (default `/r200/camera/depth_registered/points`)
  for shape pose detection.
* Subscribes to `/joint_states` directly to read finger positions for grasp verification.
* Publishes `JointTrajectory` directly to `/panda_hand_controller/joint_trajectory` to
  command the gripper, bypassing MoveIt for the hand group.
* Maintains its own collision objects in the planning scene: ground, an obstacle column
  over the target shape during the observation transit, four basket walls, an attached
  shape collision object during the held-shape transit, and a thin tile slab.
* Implements `t1_pickAndPlace(obj, goal, shape_type)` triggered from `t1_callback`.

### Gripper control bypass

`panda_arm_hand.srdf.xacro` declares `panda_finger_joint2` as a `<passive_joint>` in the
`hand` group. As a result, MoveIt's `setJointValueTarget(map{joint1, joint2})` on the
hand group silently produces a trajectory containing only `joint1`, which the
`panda_hand_controller` (a `JointTrajectoryController` requiring **both** finger joints
in every trajectory) then rejects without erroring. Symptom: fingers never moved.

Workaround: a `rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>` posts directly
to `/panda_hand_controller/joint_trajectory` with **both** finger joints listed. The
controller accepts the trajectory and the fingers actually close.

`openGripper()` and `closeGripper(width)` both use this path; `commandGripper(per_finger,
duration)` is the shared helper.

### Perception (`detectShapePose`)

Input: a `geometry_msgs::Point obj_xy` (the spawner-reported shape position, which is
the true geometric centre of the shape — verified against the SDF model origin) and the
shape type string.

Pipeline:

1. **Multi-frame cloud accumulation.** Take 5 consecutive cloud snapshots from the
   stationary observation pose, transform each into the planning frame via TF,
   ROI-filter (150 mm XY radius around `obj_xy`, world z ∈ [0.028, 0.110], drop
   green-dominant pixels to reject the underlying tile mesh), and accumulate all
   surviving points into one combined cluster. Reduces per-frame yaw variance by ~√5×.
2. **Mean centroid** is computed for diagnostic logging only — it is biased by the
   wrist-camera offset / gripper shadow which leaves more visible points on the far
   face.
3. **M4 yaw estimator anchored on `obj_xy`.** The 4th-order complex moment
   `M4 = Σ (z − pivot)^4` is computed with `pivot = obj_xy` (the true shape centre)
   instead of the biased mean. The recovered yaw is `(1/4) · arg(M4 · exp(−jφ))` with
   `φ = π` for nought and `0` for cross (the nought's corner-flip phase). Result is
   folded into `[−π/4, π/4]` because both shapes are 4-fold symmetric. Anchoring on
   `obj` removes the centroid-bias coupling that otherwise feeds a few degrees of yaw
   error per frame into the grasp.
4. **Size snap.** The maximum absolute extent in the rotated frame
   (`max(|rx|, |ry|)`) is divided by 2.5 and snapped to the nearest of
   `{0.020, 0.030, 0.040}`. Using `max_half` (with the biased mean as origin) is more
   robust than the OBB span here, because the depth sensor reliably strips ~10–15 mm
   off the outer edge of each face — the OBB span systematically undersizes (a 30 mm
   nought reads `obb_half ≈ 0.060` instead of `0.075`), but the mean-anchored max
   distance happens to compensate for that stripping with the bias offset.
5. **OBB midpoint** is also computed and returned as a `corr` diagnostic; it is not
   used for the grasp itself in this submission (we anchor the grasp on `obj` directly).

### Grasp / transit / place pipeline

```
openGripper                                 ; via direct JointTrajectory
addShapeCollision(obj)                      ; 0.20 × 0.20 × 0.08 m column over obj
addBasketCollision(goal)                    ; 4 wall boxes; per-wall apply + verify
moveArmToPose(obj.xy, SAFE_ALTITUDE=0.40, yaw=0)  ; observation
detectShapePose                              ; multi-frame, obj-anchored M4
moveArmCartesian to (grasp.xy, safe_z, link8_yaw)  ; constant-z, no fallback
removeShapeCollision
moveArmCartesian descent → pre_grasp_z → grasp_ee_z   ; no fallback
closeGripper(close_w = max(0.012, s − 0.004))
post-grasp finger-width verify              ; abort if width < close_w + 2 mm
attachShape(size_s, ox_local, oy_local)     ; held shape becomes part of robot
moveArmCartesian lift to safe_z at link8_yaw
moveArmCartesian rotate-in-place to transit_yaw = −π/4
addTileCollision                             ; 2 mm tile slab at z = [0.018, 0.020]
moveArmToPose(place.xy, safe_z, transit_yaw)  ; joint-space, RRT arcs around base
removeTileCollision
moveArmCartesian descent to place_ee_z
openGripper
detachShape
moveArmCartesian retreat to safe_z
removeBasketCollision
```

### Geometric constants (with derivations)

* `EE_TO_FINGER = 0.1122 m` — measured from
  `panda_description/meshes/collision/finger.stl` bounds (Z range [0.0001, 0.0538] in
  finger-link frame) plus the finger-joint origin at hand_z = 0.0584. The earlier value
  0.1034 was 8.8 mm short; that error pushed the released shape through the basket
  floor.
* `SHAPE_LINK_Z_OFFSET = 0.020 m` — from each shape SDF's `<pose>0 0 20e-3 1.5708 0 0`.
* `SHAPE_THICKNESS = 0.040 m` — 40 mm shape z-extent (from STL bounds and the "40H"
  suffix in mesh filenames).
* `BASKET_FLOOR_OFFSET = 0.015 m` — chosen to give ~10 mm of corner clearance above the
  basket interior floor (`goal.z + 0.0045` after settling) for shapes held with up to
  ~5° of in-grip tilt. `(0.075 m) · sin(5°) ≈ 6.5 mm` of corner descent + 5 mm physics
  margin → 15 mm.
* `SAFE_ALTITUDE = 0.40 m` — link8 z used for transits.
* `transit_yaw = −π/4` — chosen so the held shape's sides align with world axes
  (`shape +X axis world = link8_yaw + 5π/4`, must be `0 mod π/2` for a 4-fold symmetric
  shape, which folds to `link8_yaw = −π/4`). Without this, the held shape sat as a
  45°-rotated diamond in the basket and corners caught on basket walls.

### Place-position formula

The TCP at place must put the **shape centre** at `goal.xy`, not the gripper itself,
because the gripper holds an edge wall (nought) or arm midpoint (cross) of the shape,
not its centre. With the hand-to-shape rotation chain
`Rz(transit_yaw) · Rx(π) · Rz(−π/4)`, shape centre in hand frame is
`(−ox_local, +oy_local, EE_TO_FINGER)` (derived from hand_X = +shape_X,
hand_Y = −shape_Y at link8_yaw = −π/4). The world offset is computed explicitly in
`t1_pickAndPlace` and subtracted from `goal.xy` to give the TCP target.

### Robustness layers

* **Stale-state cleanup at task start.** Failed runs can leak the `held_shape`
  AttachedCollisionObject, the `t1_shape_obstacle` column, the `tile_top` slab, or
  stale basket walls into the planning scene. All are removed unconditionally at the
  top of `t1_pickAndPlace` (idempotent — no-op if absent) before the new task's
  collision objects are registered.
* **Cartesian path with `allow_fallback=false`** on every grasp/place descent. If
  Cartesian IK can't interpolate the path, the function returns false rather than
  silently dropping to joint-space planning that could sweep the arm through the shape.
* **Direct joint-space transit** for the long grasp→basket move (RRTConnect). No
  Cartesian midpoint waypoint, because the geometric midpoint of typical (grasp, place)
  pairs lies within ~10 cm of the Panda base axis, where IK goal sampling fails. Direct
  joint-space lets RRT arc around the base via joint-1 rotation.
* **Tile and basket collision objects** force the joint-space planner to keep the held
  shape above the tile and route around the basket walls. Basket walls are added with
  per-wall `applyCollisionObject` calls plus polling on `getKnownObjectNames` to verify
  they actually landed in the planning scene before the next motion plan runs.
* **Post-grasp verification** reads `/joint_states` 200 ms after the close trajectory
  and checks `total_finger_width >= close_w + 2 mm`. If the fingers reached the
  commanded close width (i.e. closed on air), the grip is recognised as failed, the
  gripper is reopened, basket walls cleaned up, and the function returns false —
  preventing a phantom-grip from continuing through to a no-op place.
* **Diagnostic logging.** Each task prints a single
  `T1[shape] s=… yaw=… obj=… c=… off=… grasp=… link8_yaw=… open_f1=… open_f2=… place=… close=…`
  line and a `grasp verify` line. These were the basis for most of the geometry /
  sign-error fixes during development.

### Known limitations

* No re-detection between observation pose and grasp pose; perception is a single shot
  (with 5-frame averaging) at the observation altitude. Slight drift between
  observation and descent is not corrected.
* No closed-loop pose correction after grasp — if the shape shifts in the gripper
  during close, the place position assumes the model-frame relationship is still exact.
* Place yaw is fixed at −π/4 (axis-aligning the held shape with the basket); shapes are
  not rotationally aligned to any specific basket orientation beyond that.
* The grasp descent contact velocity is not explicitly slowed; default MoveIt velocity
  scaling is used. Future work would lower velocity scaling on the final descent
  segment.

## Time per task & per-student contribution

| Task                                    | Approx. hours | Ali Shihab | (Member 2) | (Member 3) |
| --------------------------------------- | ------------- | ---------- | ---------- | ---------- |
| Task 1 — Pick and Place                 | ~30 hrs       | 100%       | 0%         | 0%         |
| Task 2 — Shape Detection                | 0 hrs         | n/a        | n/a        | n/a        |
| Task 3 — Planning and Execution         | 0 hrs         | n/a        | n/a        | n/a        |

## Repository layout

```
comp0250_s26_labs/
├── README.md                               # this file
├── rebuild_task1.sh                        # convenience: kill + rebuild + relaunch
├── src/
│   └── courseworks/
│       ├── cw2_world_spawner/              # provided by upstream — DO NOT modify when grading
│       ├── panda_description/              # provided by upstream
│       ├── panda_moveit_config/            # provided by upstream
│       ├── rpl_panda_with_rs/              # provided by upstream
│       └── cw2_team_13/                    # OUR SOLUTION — submission folder
│           ├── package.xml
│           ├── CMakeLists.txt
│           ├── README.txt                  # short per-package readme
│           ├── launch/
│           │   ├── run_solution.launch.py  # entrypoint launched by markers
│           │   └── run_solution.launch
│           ├── include/cw2_class.h
│           └── src/
│               ├── cw2_class.cpp           # all task logic
│               └── cw2_node.cpp            # main()
└── ...
```

## Reproducing development runs

`rebuild_task1.sh` is the script we used during development. It kills any leftover
Gazebo/move_group/world_spawner processes (they don't always shut down cleanly between
runs), rebuilds `cw2_team_13` with `--symlink-install`, and relaunches with both the
Gazebo GUI and RViz on. The cw2 solution node logs a `T1[...]` line at the start of
each Task 1 attempt and a `grasp verify` line after each close — those two lines are the
quickest way to see what the geometry-aware pipeline computed and whether the
post-grasp check passed.

## Acknowledgments

Built on the `surgical-vision/comp0250_s26_labs` upstream
(<https://github.com/surgical-vision/comp0250_s26_labs>). Panda model and
`cw2_world_spawner` infrastructure are from the upstream; `cw2_team_13` is our own.
