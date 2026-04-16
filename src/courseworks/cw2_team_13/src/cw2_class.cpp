/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <thread>
#include <utility>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>


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

  arm_group_->setPlanningTime(5.0);
  arm_group_->setNumPlanningAttempts(5);
  arm_group_->setMaxVelocityScalingFactor(0.5);
  arm_group_->setMaxAccelerationScalingFactor(0.5);
  arm_group_->setGoalPositionTolerance(0.005);
  arm_group_->setGoalOrientationTolerance(0.01);

  RCLCPP_INFO(node_->get_logger(),
    "cw2_team_13 initialised. Pointcloud topic: '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}


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


void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  RCLCPP_INFO(node_->get_logger(),
    "Task 1 start: shape='%s' obj=(%.3f,%.3f,%.3f) goal=(%.3f,%.3f,%.3f)",
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
    RCLCPP_INFO(node_->get_logger(), "Task 1 complete.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Task 1 failed.");
  }
}


// Grasp layout
//
// Both shapes come in three arm-widths s = {20, 30, 40} mm and sit in a
// 5s x 5s bounding box, 40 mm tall.
//   cross:   grasp the +x arm in shape-local frame, at (+1.5s, 0)
//   nought:  grasp the +y wall at (0, +2s)
// In both cases the fingers close along the shape-local y axis, so the
// hand's finger-open axis must lie along world yaw + pi/2.
//
// Yaw is recovered from the 4th-order complex moment of the filtered
// cloud, M4 = sum (x + j y)^4. For a C4-symmetric mass distribution
// arg(M4) = 4 theta + phi_shape (mod 2 pi). The cross has mass along
// its arms so phi_shape = 0; the square nought is dominated by its four
// outer corners which pull M4 onto the negative real axis, giving
// phi_shape = pi.

namespace {

constexpr double SHAPE_THICKNESS      = 0.040;   // z extent of the shape
constexpr double EE_TO_FINGER         = 0.1034;  // panda_link8 -> fingertip
constexpr double APPROACH_DIST        = 0.10;    // cartesian descent
constexpr double SAFE_ALTITUDE        = 0.40;    // transit height for panda_link8
constexpr double BASKET_FLOOR_OFFSET  = 0.005;   // clearance above goal.z
constexpr double SHAPE_LINK_Z_OFFSET  = 0.020;   // link-frame offset in the SDF

// ~2 mm of squeeze on the arm, with a sensible lower bound so we never
// close so far we crush a contact-stop.
inline double graspCloseWidth(double s)
{
  return std::max(0.012, s - 0.004);
}

// panda_hand is bolted on panda_link8 with a fixed -pi/4 yaw offset, so
// the world angle of the finger-open axis is link8_yaw - pi/4. Invert
// that here.
inline double link8YawForHandY(double phi)
{
  return phi + M_PI / 4.0;
}

}  // namespace


bool cw2::detectShapePose(
  const geometry_msgs::msg::Point & obj_xy,
  const std::string & shape_type,
  double & out_yaw,
  double & out_size)
{
  // Wait for a couple of fresh frames - the arm has just moved and we don't
  // want a snapshot taken mid-motion.
  std::uint64_t seq0;
  {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    seq0 = g_cloud_sequence_;
  }
  using namespace std::chrono;
  const auto deadline = steady_clock::now() + milliseconds(2500);
  while (steady_clock::now() < deadline) {
    std::this_thread::sleep_for(milliseconds(30));
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    if (g_cloud_sequence_ >= seq0 + 2 && !g_cloud_ptr->empty()) break;
  }

  PointCPtr cloud_in;
  std::string cloud_frame;
  {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    cloud_in = g_cloud_ptr;
    cloud_frame = g_input_pc_frame_id_;
  }
  if (!cloud_in || cloud_in->empty() || cloud_frame.empty()) {
    RCLCPP_ERROR(node_->get_logger(),
      "detectShapePose: no cloud (frame='%s', size=%zu)",
      cloud_frame.c_str(), cloud_in ? cloud_in->size() : 0);
    return false;
  }

  const std::string planning_frame = arm_group_->getPlanningFrame();
  geometry_msgs::msg::TransformStamped tfs;
  try {
    tfs = tf_buffer_.lookupTransform(
      planning_frame, cloud_frame, tf2::TimePointZero,
      tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException & e) {
    RCLCPP_ERROR(node_->get_logger(),
      "detectShapePose: tf %s -> %s failed: %s",
      cloud_frame.c_str(), planning_frame.c_str(), e.what());
    return false;
  }
  const Eigen::Quaterniond R(
    tfs.transform.rotation.w, tfs.transform.rotation.x,
    tfs.transform.rotation.y, tfs.transform.rotation.z);
  const Eigen::Vector3d T(
    tfs.transform.translation.x, tfs.transform.translation.y,
    tfs.transform.translation.z);

  // ROI around obj: a 0.15 m XY radius, z-band above the tiles, and drop
  // green-dominant pixels so the tile mesh doesn't leak in.
  std::vector<Eigen::Vector2d> xy_pts;
  xy_pts.reserve(cloud_in->size() / 4 + 64);
  const double z_lo = 0.028;
  const double z_hi = 0.110;
  const double roi_r2 = 0.15 * 0.15;
  for (const auto & p : *cloud_in) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      continue;
    const Eigen::Vector3d pw =
      R * Eigen::Vector3d(p.x, p.y, p.z) + T;
    if (pw.z() < z_lo || pw.z() > z_hi) continue;
    const double dx = pw.x() - obj_xy.x;
    const double dy = pw.y() - obj_xy.y;
    if (dx * dx + dy * dy > roi_r2) continue;
    const int r = p.r, g = p.g, b = p.b;
    if (g > r + 25 && g > b + 25) continue;
    xy_pts.emplace_back(pw.x(), pw.y());
  }
  if (xy_pts.size() < 30) {
    RCLCPP_ERROR(node_->get_logger(),
      "detectShapePose: only %zu points after filter", xy_pts.size());
    return false;
  }

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto & q : xy_pts) centroid += q;
  centroid /= static_cast<double>(xy_pts.size());

  // 4th-order complex moment. The nought's corners flip its sign, so
  // cancel that phase before dividing by 4.
  std::complex<double> M4(0.0, 0.0);
  for (const auto & q : xy_pts) {
    const std::complex<double> z(q.x() - centroid.x(), q.y() - centroid.y());
    const std::complex<double> z2 = z * z;
    M4 += z2 * z2;
  }
  const double phi_shape = (shape_type == "nought") ? M_PI : 0.0;
  double yaw = 0.25 * std::arg(M4 * std::polar(1.0, -phi_shape));
  while (yaw >  M_PI / 4.0) yaw -= M_PI / 2.0;
  while (yaw < -M_PI / 4.0) yaw += M_PI / 2.0;
  out_yaw = yaw;

  // Oriented bounding box half-extent = 2.5 * s for both shapes; snap to
  // the spawner's discrete set.
  const double cs = std::cos(-yaw), sn = std::sin(-yaw);
  double max_half = 0.0;
  for (const auto & q : xy_pts) {
    const double dx = q.x() - centroid.x(), dy = q.y() - centroid.y();
    const double rx = cs * dx - sn * dy;
    const double ry = sn * dx + cs * dy;
    max_half = std::max(max_half,
                        std::max(std::fabs(rx), std::fabs(ry)));
  }
  const double s_raw = max_half / 2.5;
  const std::array<double, 3> sizes = {0.020, 0.030, 0.040};
  double best_size = sizes[0];
  double best_err = std::fabs(s_raw - sizes[0]);
  for (size_t i = 1; i < sizes.size(); ++i) {
    const double e = std::fabs(s_raw - sizes[i]);
    if (e < best_err) { best_err = e; best_size = sizes[i]; }
  }
  if (s_raw > 0.050) best_size = 0.040;
  out_size = best_size;

  RCLCPP_INFO(node_->get_logger(),
    "detectShapePose[%s]: pts=%zu centroid=(%.3f,%.3f) yaw=%.3f max_half=%.3f raw_s=%.3f -> size=%.3f",
    shape_type.c_str(), xy_pts.size(), centroid.x(), centroid.y(),
    out_yaw, max_half, s_raw, out_size);

  return true;
}


bool cw2::t1_pickAndPlace(
  const geometry_msgs::msg::Point & obj,
  const geometry_msgs::msg::Point & goal,
  const std::string & shape_type)
{
  // Open first so the fingers don't occlude the wrist camera during
  // observation.
  if (!openGripper()) return false;

  // Overhead observation pose directly above obj.
  if (!moveArmToPose(
      makeTopDownPose(obj.x, obj.y, SAFE_ALTITUDE, 0.0),
      "safe-above-observation")) return false;

  double yaw = 0.0;
  double size_s = 0.040;
  if (!detectShapePose(obj, shape_type, yaw, size_s)) {
    RCLCPP_WARN(node_->get_logger(),
      "perception failed, falling back to yaw=0, size=40mm");
    yaw = 0.0;
    size_s = 0.040;
  }

  // Grasp offset in shape-local frame, then rotate into world.
  const bool is_cross = (shape_type == "cross");
  const double ox_local = is_cross ? (1.5 * size_s) : 0.0;
  const double oy_local = is_cross ? 0.0           : (2.0 * size_s);
  const double c_y = std::cos(yaw), s_y = std::sin(yaw);
  const double ox_world = c_y * ox_local - s_y * oy_local;
  const double oy_world = s_y * ox_local + c_y * oy_local;

  const double grasp_x = obj.x + ox_world;
  const double grasp_y = obj.y + oy_world;

  // Fingers open along the shape's local +y axis -> world angle yaw + pi/2.
  // Fold link8_yaw into [-pi/2, pi/2] (180 deg gripper symmetry) so MoveIt
  // picks the near IK solution.
  double link8_yaw = link8YawForHandY(yaw + M_PI / 2.0);
  while (link8_yaw >  M_PI) link8_yaw -= 2.0 * M_PI;
  while (link8_yaw < -M_PI) link8_yaw += 2.0 * M_PI;
  if (link8_yaw >  M_PI / 2.0) link8_yaw -= M_PI;
  if (link8_yaw < -M_PI / 2.0) link8_yaw += M_PI;

  // Same world offset at place so the shape's centre lands on goal.
  const double place_x = goal.x + ox_world;
  const double place_y = goal.y + oy_world;

  const double grasp_ee_z  = obj.z + SHAPE_LINK_Z_OFFSET
                              + SHAPE_THICKNESS / 2.0 + EE_TO_FINGER;
  const double pre_grasp_z = grasp_ee_z + APPROACH_DIST;
  const double place_ee_z  = goal.z + BASKET_FLOOR_OFFSET
                              + SHAPE_THICKNESS / 2.0 + EE_TO_FINGER;
  const double pre_place_z = place_ee_z + APPROACH_DIST;
  const double safe_z      = std::max({SAFE_ALTITUDE, pre_grasp_z + 0.05,
                                       pre_place_z + 0.05});

  const double close_w = graspCloseWidth(size_s);

  RCLCPP_INFO(node_->get_logger(),
    "T1[%s] s=%.3f yaw=%.3f grasp=(%.3f,%.3f,%.3f) place=(%.3f,%.3f,%.3f) link8_yaw=%.3f close=%.3f",
    shape_type.c_str(), size_s, yaw,
    grasp_x, grasp_y, grasp_ee_z,
    place_x, place_y, place_ee_z, link8_yaw, close_w);

  if (!moveArmToPose(
      makeTopDownPose(grasp_x, grasp_y, safe_z, link8_yaw),
      "safe-above-grasp")) return false;

  // Cartesian descent down to the shape. We disable the joint-space
  // fallback here because joint planning can sweep the arm through
  // the object.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, pre_grasp_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, grasp_ee_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  if (!closeGripper(close_w)) return false;

  // Straight up, across, straight down.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, safe_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  if (!moveArmToPose(
      makeTopDownPose(place_x, place_y, safe_z, link8_yaw),
      "safe-above-basket")) return false;

  if (!moveArmCartesian(
      {makeTopDownPose(place_x, place_y, place_ee_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  if (!openGripper()) return false;

  // Warn-only: if the retreat fails the shape is already dropped.
  if (!moveArmCartesian(
      {makeTopDownPose(place_x, place_y, safe_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) {
    RCLCPP_WARN(node_->get_logger(), "retreat failed after place");
  }

  return true;
}


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
  const double finger_pos = width / 2.0;
  std::map<std::string, double> targets;
  targets["panda_finger_joint1"] = finger_pos;
  targets["panda_finger_joint2"] = finger_pos;
  hand_group_->setJointValueTarget(targets);

  auto result = hand_group_->move();
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    // Contact stop is expected when the fingers hit the arm before
    // reaching the commanded width.
    RCLCPP_WARN(node_->get_logger(),
      "closeGripper: code %d (contact stop?), continuing",
      static_cast<int>(result.val));
  }
  return true;
}


bool cw2::moveArmToPose(
  const geometry_msgs::msg::Pose & target_pose,
  const std::string & description)
{
  arm_group_->setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto plan_rc = arm_group_->plan(plan);
  if (plan_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: plan failed (%d)",
      description.c_str(), static_cast<int>(plan_rc.val));
    arm_group_->clearPoseTargets();
    return false;
  }

  auto exec_rc = arm_group_->execute(plan);
  arm_group_->clearPoseTargets();
  if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmToPose [%s]: exec failed (%d)",
      description.c_str(), static_cast<int>(exec_rc.val));
    return false;
  }
  return true;
}

bool cw2::moveArmCartesian(
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  double eef_step,
  double jump_threshold,
  bool allow_fallback)
{
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  if (fraction < 0.0) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesian: compute error");
    if (allow_fallback) {
      return moveArmToPose(waypoints.back(), "cartesian-fallback");
    }
    return false;
  }
  if (fraction < 0.9) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesian: only %.1f%%", fraction * 100.0);
    if (allow_fallback) {
      return moveArmToPose(waypoints.back(), "cartesian-fallback");
    }
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  auto exec_rc = arm_group_->execute(plan);
  if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmCartesian: exec failed (%d)",
      static_cast<int>(exec_rc.val));
    return false;
  }
  return true;
}


geometry_msgs::msg::Pose cw2::makeTopDownPose(
  double x, double y, double z, double yaw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  // Roll = pi points panda_link8's z axis down; yaw rotates about world z.
  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, yaw);
  q.normalize();
  pose.orientation = tf2::toMsg(q);
  return pose;
}


void cw2::addGroundCollision()
{
  moveit_msgs::msg::CollisionObject ground;
  ground.header.frame_id = arm_group_->getPlanningFrame();
  ground.id = "ground_plane";

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions.resize(3);
  box.dimensions[0] = 3.0;
  box.dimensions[1] = 3.0;
  box.dimensions[2] = 0.020;

  geometry_msgs::msg::Pose pose;
  pose.position.z = -0.010;
  pose.orientation.w = 1.0;

  ground.primitives.push_back(box);
  ground.primitive_poses.push_back(pose);
  ground.operation = moveit_msgs::msg::CollisionObject::ADD;

  planning_scene_interface_.applyCollisionObjects({ground});
}


void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  (void)request;
  response->mystery_object_num = -1;
  RCLCPP_WARN(node_->get_logger(), "Task 2 not implemented.");
}


void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request>  request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();
  RCLCPP_WARN(node_->get_logger(), "Task 3 not implemented.");
}
