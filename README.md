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

The solution lives in `src/courseworks/cw2_team_13/{src,include}/cw2_class.{cpp,h}`,
launched as a single ROS 2 node by `launch/run_solution.launch.py`. `t1_callback`
handles `Task1Service` requests and delegates to `t1_pickAndPlace(obj, goal,
shape_type)`, which is the only mission code path described below.

### Node design

* One `MoveGroupInterface` for the `panda_arm` group; the hand is **not** driven through
  MoveIt (see "Gripper control" below).
* Three reentrant callback groups carried by a `MultiThreadedExecutor`: one for the
  `PointCloud2` subscription, one for `/joint_states`, one for the gripper action
  client. Without reentrant groups, `t1_callback` blocking on the gripper action would
  starve the cloud and joint-state callbacks, freezing perception and finger readback
  for the entire mission.
* All collision-scene objects (ground, basket walls, tile slab, attached held shape,
  shape obstacle column) are owned by this node and added/removed around the phases
  that need them.

### Gripper control

`panda_arm_hand.srdf.xacro` declares `panda_finger_joint2` as a `<passive_joint>` in the
`hand` group, so MoveIt's `setJointValueTarget` on the hand silently emits a trajectory
with only `joint1` — which the `panda_hand_controller` (a `JointTrajectoryController`
that requires both finger joints in every point) rejects without erroring. Symptom:
fingers never move.

We bypass MoveIt entirely for the hand: `commandGripper(per_finger, duration)` sends a
`control_msgs/action/FollowJointTrajectory` goal to
`/panda_hand_controller/follow_joint_trajectory` with both finger joints listed,
blocks on the action result, then polls `/joint_states` until the cached finger
positions stop moving (5 samples within 0.5 mm). `openGripper()` and
`closeGripper(width)` are thin wrappers.

### Perception (`detectShapePose`)

A 3-frame point-cloud accumulation is taken from a stationary observation pose
(`SAFE_ALTITUDE = 0.40 m` directly above the spawner-reported `obj.xy`). Each frame
waits up to 1500 ms for a fresh cloud sequence (`g_cloud_sequence_` increment),
TF-transforms it into the planning frame, and ROI-filters: 150 mm XY radius around
`obj.xy`, world `z ∈ [0.028, 0.110]`, drop green-dominant pixels. The surviving points
from all 3 frames feed one estimator.

**Yaw estimation differs by shape type, because the two shapes have different
mass distributions and the wrong tool fails on each:**

* **Nought** — minimum-area bounding rectangle (MABR). Coarse sweep over
  `θ ∈ [−π/4, π/4]` at 5° granularity, then a 1° fine sweep around the coarse min.
  The nought's hollow-ring outer rectangle has a sharp area minimum at the wall-aligned
  angle.
* **Cross** — 4th-order complex moment, `M4 = Σ (z − pivot)^4` with `pivot = obj.xy`.
  For a 4-fold symmetric mass distribution `arg(M4) = 4ψ`, so the recovered yaw is
  `arg(M4) / 4` folded into `[−π/4, π/4]`. M4 is preferred over MABR for crosses
  because the bbox of a `+` is approximately rotation-invariant — MABR alone gives
  noisy answers on cross point clouds.

The function also returns an `out_alt_yaw` second candidate. For the cross, this is
MABR-at-pivot (computed in the same pass for the cross-vs-cross-diagonal disambiguator
described next). For the nought it is set equal to the primary yaw (no ambiguity).

**Size snap.** Maximum absolute extent in the rotated frame divided by 2.5, snapped to
`{0.020, 0.030, 0.040}`. We use `max_half` rather than the OBB span because the depth
sensor strips ~10–15 mm off the outer edge of each face; the mean-anchored max distance
empirically compensates.

**Pose-correction `corr`.** OBB midpoint in the rotated frame, mapped back to world,
returned as `(out_cx, out_cy)` for downstream use as a finer-than-`obj.xy` centre when
the tiebreaker decides to adopt it.

### Cross yaw tiebreaker (second observation)

The M4 estimator for the cross can flip to the diagonal branch (`arg(M4) ≈ ±π`
instead of the expected ≈ 0) when the point cloud is small (~500 points) or
asymmetrically sampled. MABR has its own ~10% failure mode where it picks an angle
~π/4 off the true arms. Both estimators have failed in real runs and we cannot tell
which is right from a single viewpoint.

So when `|M4 − MABR| > 0.4 rad` (folded into `[−π/4, π/4]` for C4 symmetry),
`t1_pickAndPlace` aborts the first detection and triggers a tiebreaker:

1. Move to a second observation pose offset `(+60 mm, −40 mm)` in XY at
   `SAFE_ALTITUDE`. Asymmetric offset so any directional sampling bias in the first
   view shifts measurably for the second.
2. Re-run `detectShapePose` from there.
3. Pick whichever first-view candidate (M4 or MABR) is closer (folded) to the
   second-view's M4. The true yaw is consistent across viewpoints; sampling artifacts
   that flip M4 are not.
4. Adopt the second view's `(cx, cy)` as well, since the first view's sampling was
   biased enough to trigger the ambiguity.

Cost on the rare ambiguous-cross path: ~6–8 s. Cost on the common path (M4 and MABR
agree): zero. Validated end-to-end: in two consecutive runs the tiebreaker resolved
the ambiguity correctly and both grasps succeeded.

### Mission flow

```
cleanup stale planning-scene objects                  ; idempotent
addBasketCollision(goal)                              ; with retry, abort if fails
openGripper                                           ; via action client + JS settle
addShapeCollision(obj)                                ; column over obj for transit
moveArmToPose(obj.xy, SAFE_ALTITUDE, yaw=0)            ; observation
detectShapePose                                       ; 3-frame accumulation
  if cross && |M4 − MABR| > 0.4: tiebreaker           ; second observation, re-detect
  if false (cloud silent): sleep 2s, retry once       ; abort if still false
moveArmCartesian to (grasp.xy, safe_z, link8_yaw)     ; constant-z, no fallback
removeShapeCollision
moveArmCartesian descent → pre_grasp_z                ; eef_step 0.005
moveArmCartesianConstrained descent → grasp_ee_z      ; PositionConstraint, eef 0.002
closeGripper(close_w)
post-grasp finger-width verify                        ; abort if width out of band
attachShape(size_s, ox_local, oy_local)
moveArmCartesian lift to safe_z at link8_yaw
moveArmCartesian rotate-in-place to transit_yaw = −π/4
addTileCollision                                      ; 2 mm slab forces link8.z lift
moveArmToPose(place.xy, safe_z, transit_yaw)          ; joint-space, RRT around base
removeTileCollision
moveArmCartesian descent to place_ee_z
openGripper
sleep 350 ms                                          ; let fingers physically clear
detachShape
two-stage post-place retreat:                         ; see "Retreat" below
  stage 1: collision-disabled Cartesian lift +100 mm
  stage 2: joint plan to SRDF "ready" pose
removeBasketCollision
return
```

### Constrained final descent

The standard pre-grasp Cartesian descent (`safe_z → pre_grasp_z`, eef_step 0.005) is
followed by a tighter constrained descent for the last 5 mm to `grasp_ee_z`. We use
the `MoveGroupInterface::computeCartesianPath` overload that takes a `Constraints`
message and apply a `PositionConstraint` on `panda_link8`: a 10 mm × 10 mm × (descent
height + 20 mm) box centred on the grasp column, axis-aligned, weight 1.0. Each IK
checkpoint along the linear EE path is rejected unless link8's origin sits inside that
box, which forces the joint solver to find configurations whose EE stays in the
column. Without it, IK reseeding between 2 mm checkpoints could let the EE drift 5–10
mm laterally during descent and clip the outer finger against a basket wall on the
larger noughts.

If the constrained path fails (compute returns < 90% fraction), `moveArmCartesianConstrained`
falls back automatically to an unconstrained Cartesian descent so the grasp still
happens. The downstream verify catches any resulting bad grip.

### Two-stage post-place retreat

The original single Cartesian retreat from inside the open basket up to safe altitude
was failing at 0.0% on every run, leaving the arm sitting in the basket area. The
next mission's joint planner then started from that bad pose and could sweep the EE
through wherever the next basket happened to be, physically knocking it. Two-stage
fix:

* **Stage 1** — collision-disabled Cartesian lift by `APPROACH_DIST = 100 mm` straight
  up. Going straight up out of an open basket whose walls are <50 mm tall, there is
  nothing to collide with; disabling collision avoidance bypasses any stale start-state
  collision flag (the just-detached `held_shape` AttachedCollisionObject can leave
  residual collision state on the very next IK check).
* **Stage 2** — joint-space plan to the SRDF `"ready"` named target (Panda's standard
  folded posture). Returning to a canonical pose between every mission means the next
  observation move always starts from the same predictable joint state, eliminating
  the "full 360" inter-mission swing.

Both stages are warn-only — `removeBasketCollision` still runs at the end either way.

### Basket collision robustness

`addBasketCollision` adds 4 wall boxes via per-wall `applyCollisionObject` calls, then
polls `getKnownObjectNames` for up to 3 s waiting for all 4 to register. If they don't
appear, it resends them and polls again for another 3 s. If they still aren't there
it returns `false` and the caller aborts the mission cleanly — proceeding without
basket walls in the planning scene risks the arm ploughing through the physical
basket on the next motion.

### Place-position formula

The TCP at place must put the **shape centre** at `goal.xy`, not the gripper itself.
With the hand-to-shape rotation chain `Rz(transit_yaw) · Rx(π) · Rz(−π/4)`, the
shape centre in hand frame is `(−ox_local, +oy_local, EE_TO_FINGER)` (derived from
hand_X = +shape_X, hand_Y = −shape_Y at link8_yaw = −π/4). The world offset is
computed explicitly in `t1_pickAndPlace` and subtracted from `goal.xy` to get the TCP
target. `transit_yaw = −π/4` is chosen so the held shape's sides align with world
axes — without this, the held shape sat as a 45°-rotated diamond in the basket and
corners caught on basket walls.

### Geometric constants

| Constant | Value | Source |
| --- | --- | --- |
| `EE_TO_FINGER` | 0.1122 m | `finger.stl` Z range [0.0001, 0.0538] + hand_z = 0.0584 |
| `SHAPE_LINK_Z_OFFSET` | 0.020 m | shape SDF `<pose>0 0 20e-3 1.5708 0 0` |
| `SHAPE_THICKNESS` | 0.040 m | shape STL bounds, "40H" mesh suffix |
| `BASKET_FLOOR_OFFSET` | 0.015 m | ~10 mm corner clearance + 5 mm physics margin |
| `SAFE_ALTITUDE` | 0.40 m | link8 z used for transits |
| `APPROACH_DIST` | 0.10 m | Cartesian descent / lift segment |
| `transit_yaw` | −π/4 | aligns held shape sides to world axes (4-fold sym) |

### Other reliability layers

* **Cleanup at task start** — `held_shape` AttachedCollisionObject, `t1_shape_obstacle`
  column, `tile_top` slab, and any stale basket walls are removed unconditionally
  (idempotent) before the new mission's collision objects are registered. Failed
  prior runs can leak any of these.
* **Cartesian with `allow_fallback=false`** on every grasp/place segment. If
  Cartesian IK can't interpolate the path, we fail fast rather than dropping to
  joint-space planning that could sweep the arm through the shape.
* **Joint-space transit** for the long grasp→basket move, intentionally NOT Cartesian.
  The geometric midpoint of typical (grasp, place) pairs lies within ~10 cm of the
  Panda base axis, where IK goal sampling fails ("Unable to sample any valid states
  for goal tree"). Direct joint-space lets RRT arc around the base via joint-1
  rotation.
* **Tile collision object** during transit forces `link8.z > ~0.16 m` throughout,
  preventing the held shape from dragging across the table.
* **350 ms settle** between `openGripper` and `detachShape`/retreat. ODE physics has
  residual finger-pad contact even after the action reports success; without the
  pause the immediate Cartesian lift drags the just-released shape and tips it onto
  the basket wall.
* **Post-grasp verify** reads cached `/joint_states` finger widths and rejects three
  failure modes: closed-on-air (width below `close_w + 5 mm`), fingers-near-open
  (width above `size_s + 15 mm` — the upper margin allows for ODE contact
  penetration on firmly-gripped shapes), and asymmetric grip (`|f1 − f2| > size_s`).
  On any of these the gripper is reopened, the held_shape is detached, basket walls
  cleaned up, and the function returns false.
* **Vel/accel scaling 0.8** in the constructor — empirically the highest scaling that
  doesn't introduce overshoot on the small Panda joints.

### Diagnostic logging

Each mission emits a single
`T1[shape] s=… yaw=… obj=… c=… off=… grasp=… link8_yaw=… open_f1=… open_f2=… place=… close=…`
line plus a `grasp verify` line; for crosses, an additional
`detectShapePose[cross-M4]: pivot=… arg(M4)=… coh=… MABR_yaw=… branch_diff=…`
line shows the M4 estimator state and lets the tiebreaker decision be reconstructed
post-hoc. These three lines are the basis for almost every fix in the development
history.

### Known limitations

* No re-detection at the bottom of the pre-grasp descent; perception is one accumulated
  observation (with the second-observation tiebreaker for ambiguous crosses). Drift
  between observation and grasp is not corrected.
* Once grasped, the shape's relationship to the gripper is assumed exact — no
  closed-loop pose correction during transit.
* Place yaw is fixed at `−π/4`. Shapes are not rotationally aligned to any specific
  basket orientation beyond the world-axis alignment that `−π/4` produces.
* The MoveIt planning-scene monitor occasionally lags an `applyCollisionObject` by
  more than the verify timeout, in which case a planned path can use a stale scene
  snapshot. The basket-add retry catches the case where the verify itself times out,
  but cannot detect a stale-snapshot plan that completes nominally with bad data. In
  development this manifested as one basket knock per ~10 missions.

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
