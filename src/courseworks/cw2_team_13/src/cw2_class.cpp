/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <cmath>
#include <utility>

// ═══════════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════════

cw2::cw2(const rclcpp::Node::SharedPtr &node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this,
              std::placeholders::_1, std::placeholders::_2));
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this,
              std::placeholders::_1, std::placeholders::_2));
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this,
              std::placeholders::_1, std::placeholders::_2));

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pc_opts;
  pc_opts.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pc_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pc_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }
  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_, pc_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pc_opts);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    node_, "hand");

  arm_group_->setPlanningTime(3.0);
  arm_group_->setNumPlanningAttempts(3);
  arm_group_->setMaxVelocityScalingFactor(0.5);
  arm_group_->setMaxAccelerationScalingFactor(0.5);
  arm_group_->setGoalPositionTolerance(0.005);
  arm_group_->setGoalOrientationTolerance(0.01);

  RCLCPP_INFO(node_->get_logger(),
    "cw2_team_13 initialised. Pointcloud topic: '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}

// ═══════════════════════════════════════════════════════════════════════════
// Point-cloud callback
// ═══════════════════════════════════════════════════════════════════════════

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest);
  ++g_cloud_sequence_;
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 1 — pick-and-place
// ═══════════════════════════════════════════════════════════════════════════

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  RCLCPP_INFO(node_->get_logger(),
    "Task 1 start — shape='%s'  obj=(%.3f,%.3f,%.3f)  goal=(%.3f,%.3f,%.3f)",
    request->shape_type.c_str(),
    request->object_point.point.x,
    request->object_point.point.y,
    request->object_point.point.z,
    request->goal_point.point.x,
    request->goal_point.point.y,
    request->goal_point.point.z);

  addGroundCollision();

  bool ok = t1_pickAndPlace(
    request->object_point.point,
    request->goal_point.point,
    request->shape_type);

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "Task 1 completed successfully.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Task 1 failed — returning empty response.");
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Grasp geometry
// ─────────────────────────────────────────────────────────────────────────
//
//  Both shapes have 40 mm-wide arms and a 200 mm bounding square.
//  The Panda gripper opens to a maximum of 80 mm (40 mm each finger).
//  Descending at the shape centroid with fully-open fingers would land
//  the finger bodies on top of the arm surfaces, blocking descent.
//
//  Instead we grasp a single arm away from the centroid, with the open
//  fingers sitting in the adjacent void so the gripper can descend freely:
//
//  CROSS (yaw=0 in world frame, T1_ANY_ORIENTATION=False):
//    • Grasp the right horizontal arm.
//    • Arm centre:  (obj.x + 60 mm, obj.y)   [midpoint of x+20…x+100 mm]
//    • Fingers open ±40 mm in world-Y: land at (x+60, y±40)
//      → outside both the horizontal arm (|y| ≤ 20 mm) and the vertical
//        arm (|x–cx| ≤ 20 mm).  No contact during descent. ✓
//    • Fingers close to ±19 mm: land at y ±19 mm, inside ±20 mm arm. ✓
//    • Place compensation: move gripper to (goal.x + 60 mm, goal.y)
//      so the shape centroid (60 mm behind gripper in X) lands at goal.
//
//  NOUGHT (yaw=0 in world frame):
//    • Grasp the top arm of the ring.
//    • Arm centre:  (obj.x, obj.y + 80 mm)   [midpoint of y+60…y+100 mm]
//    • Fingers open ±40 mm in world-Y: land at y+40 mm (in void, < 60 mm
//      inner radius) and y+120 mm (outside ring, > 100 mm outer radius). ✓
//    • Fingers close to ±19 mm: land at y+61 mm and y+99 mm, inside
//      the arm band y+60…y+100 mm. ✓
//    • Place compensation: move gripper to (goal.x, goal.y + 80 mm).
//
//  Gripper orientation:
//    panda_hand is fixed to panda_link8 with a −45° rotation around z.
//    The fingers open/close in the panda_hand y-axis.  With panda_link8
//    yaw = π/4, the panda_hand y-axis aligns with world +Y, so setting
//    GRASP_YAW = π/4 always directs the fingers along the world Y axis.
// ─────────────────────────────────────────────────────────────────────────

bool cw2::t1_pickAndPlace(
  const geometry_msgs::msg::Point & obj,
  const geometry_msgs::msg::Point & goal,
  const std::string & shape_type)
{
  // ── Physical constants ────────────────────────────────────────────────
  // Shape: 40 mm tall, arm width 40 mm (for x = 40 mm size variant).
  // panda_link8 → fingertip distance along approach axis: 103.4 mm.
  // panda_hand_joint has xyz=(0,0,0), only a −45° z-rotation from link8.
  static constexpr double SHAPE_HEIGHT    = 0.040;   // m
  static constexpr double GRASP_WIDTH     = 0.038;   // finger gap at close (m)
  static constexpr double EE_TO_FINGER    = 0.1034;  // link8 → fingertip (m)
  static constexpr double APPROACH_DIST = 0.06;    // vertical pre-grasp clearance (m)
  // Basket: base plate centre = goal.z (spawn z).
  // Interior floor ≈ goal.z + 5 mm (top of 9 mm base plate).
  // We place the shape so its centre is at floor + SHAPE_HEIGHT/2:
  //   panda_link8_z = goal.z + 0.005 + SHAPE_HEIGHT/2 + EE_TO_FINGER
  static constexpr double BASKET_FLOOR_OFFSET = 0.005; // above goal.z (m)
  // world_spawner sends the Gazebo model-origin spawn height (≈0.025 m, NOT the shape centroid).
  // The cross_link/nought_link origin in the SDF is offset 20 mm above the model base frame,
  // so the shape centroid = obj.z + SHAPE_LINK_Z_OFFSET + SHAPE_HEIGHT/2.
  // This is valid both in our dev environment and the graders' environment, which use the
  // same world_spawner (get_model_state() returns spawn z, not centroid).
  static constexpr double SHAPE_LINK_Z_OFFSET  = 0.020; // SDF cross_link/nought_link z offset from model base
  // Gripper yaw for world-Y finger direction (π/4 compensates hand −45° joint)
  static constexpr double GRASP_YAW = M_PI / 4.0;

  // ── Shape-dependent grasp / place offsets ────────────────────────────
  double grasp_x = obj.x;
  double grasp_y = obj.y;
  double place_x = goal.x;
  double place_y = goal.y;

  if (shape_type == "cross") {
    // Grasp the right horizontal arm, centre at (obj.x + 60 mm, obj.y).
    // Compensate so shape centroid lands at goal during placement.
    const double CROSS_ARM_OFFSET = 0.060;
    grasp_x = obj.x + CROSS_ARM_OFFSET;
    place_x = goal.x + CROSS_ARM_OFFSET;
  } else {  // "nought"
    // Grasp the top arm, centre at (obj.x, obj.y + 80 mm).
    const double NOUGHT_ARM_OFFSET = 0.080;
    grasp_y = obj.y + NOUGHT_ARM_OFFSET;
    place_y = goal.y + NOUGHT_ARM_OFFSET;
  }

  // ── Z heights for panda_link8 ────────────────────────────────────────
  // obj.z is the model-origin spawn height (bottom face of shape, ≈ 0.025 m).
  // Fingertip target = shape centre = obj.z + SHAPE_HEIGHT/2.
  const double grasp_ee_z  = obj.z + SHAPE_LINK_Z_OFFSET + SHAPE_HEIGHT / 2.0 + EE_TO_FINGER;
  const double pre_grasp_z = grasp_ee_z + APPROACH_DIST;

  const double place_ee_z  = goal.z + BASKET_FLOOR_OFFSET
                             + SHAPE_HEIGHT / 2.0 + EE_TO_FINGER;
  const double pre_place_z = place_ee_z + APPROACH_DIST;

  RCLCPP_INFO(node_->get_logger(),
    "T1 grasp=(%.3f,%.3f,%.3f) place=(%.3f,%.3f) ee_z: grasp=%.3f place=%.3f",
    grasp_x, grasp_y, grasp_ee_z, place_x, place_y, grasp_ee_z, place_ee_z);

  // ── Step 1: open gripper ─────────────────────────────────────────────
  if (!openGripper()) return false;

  // ── Step 2: joint-space move to pre-grasp (safe approach, clear of shape)
  if (!moveArmToPose(
      makeTopDownPose(grasp_x, grasp_y, pre_grasp_z, GRASP_YAW),
      "pre-grasp")) return false;

  // ── Step 3: Cartesian descent — open fingers pass through adjacent void
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, grasp_ee_z, GRASP_YAW)}))
    return false;

  // ── Step 4: close gripper onto arm ───────────────────────────────────
  if (!closeGripper(GRASP_WIDTH)) return false;


  // ── Step 6: joint-space move to above basket ──────────────────────────
  if (!moveArmToPose(
      makeTopDownPose(place_x, place_y, pre_place_z, GRASP_YAW),
      "pre-place")) return false;

  // ── Step 7: Cartesian descent into basket ─────────────────────────────
  if (!moveArmCartesian(
      {makeTopDownPose(place_x, place_y, place_ee_z, GRASP_YAW)}))
    return false;

  // ── Step 8: open gripper — release shape ─────────────────────────────
  if (!openGripper()) return false;

  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gripper helpers
// ═══════════════════════════════════════════════════════════════════════════

bool cw2::openGripper()
{
  hand_group_->setNamedTarget("open");
  auto result = hand_group_->move();
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "openGripper failed (code %d)", static_cast<int>(result.val));
    return false;
  }
  return true;
}

bool cw2::closeGripper(double width)
{
  // Each prismatic finger joint value = half the total gap.
  const double finger_pos = width / 2.0;
  std::map<std::string, double> targets;
  targets["panda_finger_joint1"] = finger_pos;
  targets["panda_finger_joint2"] = finger_pos;
  hand_group_->setJointValueTarget(targets);

  auto result = hand_group_->move();
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    // Contact-stop: fingers hit the object before reaching the target joint
    // value.  This is the expected outcome when gripping a solid shape, so
    // we log a warning rather than propagating a failure.
    RCLCPP_WARN(node_->get_logger(),
      "closeGripper: code %d — likely contact-stop, continuing",
      static_cast<int>(result.val));
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Arm motion helpers
// ═══════════════════════════════════════════════════════════════════════════

bool cw2::moveArmToPose(
  const geometry_msgs::msg::Pose & target_pose,
  const std::string & description)
{
  arm_group_->setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto plan_rc = arm_group_->plan(plan);
  if (plan_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: planning failed (code %d)",
      description.c_str(), static_cast<int>(plan_rc.val));
    arm_group_->clearPoseTargets();
    return false;
  }

  auto exec_rc = arm_group_->execute(plan);
  arm_group_->clearPoseTargets();
  if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: execution failed (code %d)",
      description.c_str(), static_cast<int>(exec_rc.val));
    return false;
  }
  return true;
}

bool cw2::moveArmCartesian(
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  double eef_step,
  double jump_threshold)
{
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  if (fraction < 0.0) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesian: path computation error — falling back to joint-space");
    return moveArmToPose(waypoints.back(), "cartesian-fallback");
  }
  if (fraction < 0.9) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesian: only %.1f%% computed — falling back to joint-space",
      fraction * 100.0);
    return moveArmToPose(waypoints.back(), "cartesian-fallback");
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  auto exec_rc = arm_group_->execute(plan);
  if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmCartesian: execution failed (code %d)",
      static_cast<int>(exec_rc.val));
    return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pose factory
// ═══════════════════════════════════════════════════════════════════════════

geometry_msgs::msg::Pose cw2::makeTopDownPose(
  double x, double y, double z, double yaw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  // RPY = (π, 0, yaw)
  //   roll = π  → panda_link8 z-axis points downward (−world-z)
  //   yaw       → rotates the gripper about the world vertical axis
  // With yaw = π/4, the panda_hand y-axis (finger-open axis) aligns
  // with world +Y (compensating the panda_hand −45° fixed joint).
  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, yaw);
  q.normalize();
  pose.orientation = tf2::toMsg(q);
  return pose;
}

// ═══════════════════════════════════════════════════════════════════════════
// Collision scene
// ═══════════════════════════════════════════════════════════════════════════

void cw2::addGroundCollision()
{
  moveit_msgs::msg::CollisionObject ground;
  ground.header.frame_id = arm_group_->getPlanningFrame();
  ground.id = "ground_plane";

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions.resize(3);
  box.dimensions[0] = 3.0;    // x extent (m)
  box.dimensions[1] = 3.0;    // y extent (m)
  box.dimensions[2] = 0.020;  // thickness (m) = tile height

  geometry_msgs::msg::Pose pose;
  pose.position.z = -0.010;   // centre just below z = 0
  pose.orientation.w = 1.0;

  ground.primitives.push_back(box);
  ground.primitive_poses.push_back(pose);
  ground.operation = moveit_msgs::msg::CollisionObject::ADD;

  planning_scene_interface_.applyCollisionObjects({ground});
  RCLCPP_INFO(node_->get_logger(), "addGroundCollision: ground plane applied");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 2 stub
// ═══════════════════════════════════════════════════════════════════════════

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  (void)request;
  response->mystery_object_num = -1;
  RCLCPP_WARN(node_->get_logger(), "Task 2 not yet implemented.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 3 stub
// ═══════════════════════════════════════════════════════════════════════════

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();
  RCLCPP_WARN(node_->get_logger(), "Task 3 not yet implemented.");
}
