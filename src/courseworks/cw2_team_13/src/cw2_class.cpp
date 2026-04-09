/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

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
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2));
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2));
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2));

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  // Conservative planning parameters — increase success rate
  arm_group_->setPlanningTime(10.0);
  arm_group_->setNumPlanningAttempts(5);
  arm_group_->setMaxVelocityScalingFactor(0.3);
  arm_group_->setMaxAccelerationScalingFactor(0.3);
  arm_group_->setGoalPositionTolerance(0.005);    // 5 mm
  arm_group_->setGoalOrientationTolerance(0.01);  // ~0.6 deg

  RCLCPP_INFO(
    node_->get_logger(),
    "cw2_team_13 initialised with pointcloud topic '%s' (%s QoS)",
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

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest_cloud);
  ++g_cloud_sequence_;
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 1 — pick-and-place
// ═══════════════════════════════════════════════════════════════════════════

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  RCLCPP_INFO(node_->get_logger(),
    "Task 1 received: shape_type='%s'  obj=(%.3f, %.3f, %.3f)  goal=(%.3f, %.3f, %.3f)",
    request->shape_type.c_str(),
    request->object_point.point.x, request->object_point.point.y, request->object_point.point.z,
    request->goal_point.point.x,   request->goal_point.point.y,   request->goal_point.point.z);

  addGroundCollision();

  bool ok = t1_pickAndPlace(
    request->object_point.point,
    request->goal_point.point);

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "Task 1 completed successfully.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Task 1 pick-and-place failed.");
  }
}

// ─────────────────────────────────────────────────────────────────────────
// t1_pickAndPlace — full sequence
// ─────────────────────────────────────────────────────────────────────────

bool cw2::t1_pickAndPlace(
  const geometry_msgs::msg::Point & obj,
  const geometry_msgs::msg::Point & goal)
{
  // ── Geometric constants ────────────────────────────────────────────────
  // Shape: 40 mm arm width, 40 mm tall. Panda gripper opens to 80 mm max
  // (each finger travels 40 mm). We close to 38 mm gap to grip the 40 mm
  // arm with a small compliance margin.
  //
  // The world spawner sends model origin positions (panda_link0 frame).
  // For these SDF models the origin sits at the bottom face of the shape,
  // so the top surface is at obj.z + shape_height.
  //
  // panda_link8 to fingertip distance along the approach axis ≈ 0.1034 m.
  // We grasp with fingertips at (obj.z + shape_height/2), so:
  //   panda_link8_z = obj.z + shape_height/2 + 0.1034

  static constexpr double SHAPE_HEIGHT   = 0.040;   // m
  static constexpr double GRASP_WIDTH    = 0.038;   // finger gap (m)
  static constexpr double EE_TO_FINGER   = 0.1034;  // panda_link8 → fingertip (m)
  static constexpr double APPROACH_DIST  = 0.12;    // vertical pre-grasp clearance (m)
  static constexpr double BASKET_HEIGHT  = 0.050;   // basket wall height (m)
  static constexpr double PLACE_MARGIN   = 0.015;   // EE height above basket floor when releasing

  // Height of panda_link8 for a mid-shape grasp
  const double grasp_ee_z  = obj.z  + SHAPE_HEIGHT / 2.0 + EE_TO_FINGER;
  const double pre_grasp_z = grasp_ee_z + APPROACH_DIST;

  // Height of panda_link8 for placing inside basket
  // Basket floor ≈ goal.z + BASKET_HEIGHT; we want fingertips at floor + PLACE_MARGIN
  const double place_ee_z  = goal.z + BASKET_HEIGHT + PLACE_MARGIN + EE_TO_FINGER;
  const double pre_place_z = place_ee_z + APPROACH_DIST;

  RCLCPP_INFO(node_->get_logger(),
    "T1 grasp_ee_z=%.3f  place_ee_z=%.3f", grasp_ee_z, place_ee_z);

  // ── Step 1: open gripper ───────────────────────────────────────────────
  if (!openGripper()) return false;

  // ── Step 2: move to pre-grasp (joint-space plan, safe approach) ────────
  if (!moveArmToPose(makeTopDownPose(obj.x, obj.y, pre_grasp_z), "pre-grasp")) return false;

  // ── Step 3: linear descent to grasp height (Cartesian) ────────────────
  if (!moveArmCartesian({makeTopDownPose(obj.x, obj.y, grasp_ee_z)}, 0.005)) return false;

  // ── Step 4: close gripper ─────────────────────────────────────────────
  if (!closeGripper(GRASP_WIDTH)) return false;

  // ── Step 5: linear lift back to pre-grasp height ──────────────────────
  if (!moveArmCartesian({makeTopDownPose(obj.x, obj.y, pre_grasp_z)}, 0.005)) return false;

  // ── Step 6: joint-space move to above basket ──────────────────────────
  if (!moveArmToPose(makeTopDownPose(goal.x, goal.y, pre_place_z), "pre-place")) return false;

  // ── Step 7: linear descent into basket ────────────────────────────────
  if (!moveArmCartesian({makeTopDownPose(goal.x, goal.y, place_ee_z)}, 0.005)) return false;

  // ── Step 8: open gripper (release shape) ──────────────────────────────
  if (!openGripper()) return false;

  // ── Step 9: linear retreat upward ─────────────────────────────────────
  if (!moveArmCartesian({makeTopDownPose(goal.x, goal.y, pre_place_z)}, 0.005)) return false;

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
    RCLCPP_ERROR(node_->get_logger(), "openGripper: move failed (code %d)",
      static_cast<int>(result.val));
    return false;
  }
  return true;
}

bool cw2::closeGripper(double width)
{
  // Each Panda finger joint value = half the total gap
  const double finger_pos = width / 2.0;
  std::map<std::string, double> joint_targets;
  joint_targets["panda_finger_joint1"] = finger_pos;
  joint_targets["panda_finger_joint2"] = finger_pos;

  hand_group_->setJointValueTarget(joint_targets);
  auto result = hand_group_->move();
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(node_->get_logger(),
      "closeGripper: move returned code %d (may be contact-stopped — continuing)",
      static_cast<int>(result.val));
    // A non-SUCCESS result when closing can mean the fingers hit the object,
    // which is actually the desired behaviour. We do not fail here.
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
  auto plan_result = arm_group_->plan(plan);

  if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: planning failed (code %d)",
      description.c_str(), static_cast<int>(plan_result.val));
    arm_group_->clearPoseTargets();
    return false;
  }

  auto exec_result = arm_group_->execute(plan);
  arm_group_->clearPoseTargets();

  if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: execution failed (code %d)",
      description.c_str(), static_cast<int>(exec_result.val));
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

  // computeCartesianPath returns the fraction [0,1] of path achieved
  double fraction = arm_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  if (fraction < 0.0) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmCartesian: computeCartesianPath returned error (fraction=%.3f)", fraction);
    return moveArmToPose(waypoints.back(), "cartesian-fallback");
  }

  if (fraction < 0.9) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesian: only %.1f%% of path computed — falling back to joint-space",
      fraction * 100.0);
    return moveArmToPose(waypoints.back(), "cartesian-fallback");
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;

  auto exec_result = arm_group_->execute(plan);
  if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmCartesian: execution failed (code %d)",
      static_cast<int>(exec_result.val));
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

  // Top-down orientation: panda_link8 z-axis pointing downward (-world z).
  // RPY = (π, 0, yaw):
  //   • roll = π  flips the EE so the fingers point down
  //   • yaw   rotates the gripper about the world vertical axis, allowing
  //             orientation-matched grasping (Task 1b)
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
  // Add a flat box representing the table surface so MoveIt! prevents the
  // arm from planning through the ground plane.
  moveit_msgs::msg::CollisionObject ground;
  ground.header.frame_id = arm_group_->getPlanningFrame();
  ground.id = "ground_plane";

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions.resize(3);
  primitive.dimensions[0] = 3.0;   // x extent (m)
  primitive.dimensions[1] = 3.0;   // y extent (m)
  primitive.dimensions[2] = 0.02;  // thickness (m)

  geometry_msgs::msg::Pose pose;
  pose.position.x = 0.0;
  pose.position.y = 0.0;
  pose.position.z = -0.01;  // top face at z = 0
  pose.orientation.w = 1.0;

  ground.primitives.push_back(primitive);
  ground.primitive_poses.push_back(pose);
  ground.operation = moveit_msgs::msg::CollisionObject::ADD;

  planning_scene_interface_.applyCollisionObjects({ground});
  RCLCPP_INFO(node_->get_logger(), "addGroundCollision: ground plane added to planning scene");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 2 stub
// ═══════════════════════════════════════════════════════════════════════════

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
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
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();
  RCLCPP_WARN(node_->get_logger(), "Task 3 not yet implemented.");
}
