/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
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

  // Direct subscription to /joint_states so the post-grasp verify can
  // read the actual finger positions without going through MoveIt's
  // (unreliable, in this setup) cached state.
  // Reentrant callback group so jointStatesCallback can fire while
  // t1_callback is blocked in commandGripper or in the post-grasp
  // verify settle loop. Without this, finger position cache is frozen
  // at whatever it was when t1_callback started (default open width
  // 0.07 m) and the verify always reads stale data.
  joint_states_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions js_opts;
  js_opts.callback_group = joint_states_callback_group_;
  joint_states_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::SensorDataQoS(),
    std::bind(&cw2::jointStatesCallback, this, std::placeholders::_1),
    js_opts);

  // Action client for the panda_hand_controller, on a Reentrant
  // callback group. Without this, the action client's response
  // callbacks share the default MutuallyExclusiveCallbackGroup with
  // t1_callback - so when t1_callback blocks waiting on the action
  // result, the action's response callback can never run because it
  // can't preempt t1_callback. Symptom: hang forever after the first
  // grasp move (no respawn, no progress). Reentrant + multi-threaded
  // executor lets the response callback run on another thread.
  hand_action_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  hand_action_client_ = rclcpp_action::create_client<FjtAction>(
    node_,
    "/panda_hand_controller/follow_joint_trajectory",
    hand_action_callback_group_);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    node_, "hand");

  arm_group_->setPlanningTime(5.0);
  arm_group_->setNumPlanningAttempts(5);
  arm_group_->setMaxVelocityScalingFactor(0.8);
  arm_group_->setMaxAccelerationScalingFactor(0.8);
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

void cw2::jointStatesCallback(
  const sensor_msgs::msg::JointState::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(joint_states_mutex_);
  for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
    if (msg->name[i] == "panda_finger_joint1") finger1_pos_ = msg->position[i];
    else if (msg->name[i] == "panda_finger_joint2") finger2_pos_ = msg->position[i];
  }
  finger_state_seen_ = true;
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
constexpr double EE_TO_FINGER         = 0.1122;  // panda_link8 -> fingertip
// EVIDENCE: read directly from
// panda_description/meshes/collision/finger.stl, which has Z bounds
// [0.0001, 0.0538] in finger-link frame. Combined with finger_joint1
// origin at hand_z=0.0584 in hand frame, the actual fingertip sits
// at hand_z = 0.0584 + 0.0538 = 0.1122 m from link8. The previous
// 0.1034 placed grasp+place poses 8.8 mm too LOW; visible as the
// shape clipping through the basket floor on release (the user's
// 'placed it so deep into the box it broke the physics' symptom).
constexpr double APPROACH_DIST        = 0.10;    // cartesian descent
constexpr double SAFE_ALTITUDE        = 0.40;    // transit height for panda_link8
constexpr double BASKET_FLOOR_OFFSET  = 0.015;   // clearance above goal.z
// EVIDENCE for raising from 0.005 to 0.015: the held shape can tilt
// a few degrees during grasp/transit due to slight off-centre
// finger contact. A 5 deg tilt drops a 30 mm nought's corner by
// 0.075 * sin(5 deg) = 6.5 mm. With 5 mm of nominal clearance the
// corner ends up below the basket interior floor (goal.z + 0.0045
// after settling) and the physics engine resolves the
// interpenetration by ejecting the shape downward through the
// floor - the 'shoves the object deep below the bottom of the box'
// symptom the user reported. 15 mm of nominal clearance covers
// up to ~11 deg of tilt while still being a soft 0.54 m/s drop.
constexpr double SHAPE_LINK_Z_OFFSET  = 0.020;   // link-frame offset in the SDF

// AGGRESSIVE close target: command the gripper to close FAR past the
// shape thickness so the controller continues applying squeeze effort
// after contact. Without this the controller marks the goal "reached"
// (or ABORTED on goal-time tolerance) almost immediately and the
// fingers only lightly touch the shape - shape slips during transit.
// EVIDENCE for the change: a 40 mm nought run had width=0.040 (light
// grip = touching but no force), action ABORTED, and the shape fell
// out of the gripper during the transit to the basket.
//
// Per-finger commanded position = (close_w / 2). For an aggressive
// grip we want commanded position MUCH less than s/2 so that contact
// reaction is the limiting factor, not the trajectory target.
//
// Lower-bounded at 0.005 (= 1 cm width) so we don't try to drive the
// fingers into each other.
inline double graspCloseWidth(double s)
{
  return std::max(0.005, s - 0.020);
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
  double & out_size,
  double & out_cx,
  double & out_cy,
  double * out_alt_yaw)
{
  // MULTI-FRAME ACCUMULATION: take N consecutive cloud snapshots from
  // the static observation pose and accumulate ROI-filtered points
  // from all of them into one combined xy_pts. Per-frame point
  // jitter and asymmetric sampling biases largely average out, and
  // the M4 yaw estimate computed over the union has lower variance
  // than any single-frame estimate.
  using namespace std::chrono;
  constexpr int kNumFrames = 3;
  const std::string planning_frame = arm_group_->getPlanningFrame();

  std::vector<Eigen::Vector2d> xy_pts;
  xy_pts.reserve(8192);

  std::uint64_t last_seq = 0;
  {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    last_seq = g_cloud_sequence_;
  }

  std::string cloud_frame;
  for (int frame_idx = 0; frame_idx < kNumFrames; ++frame_idx) {
    // Wait for a NEW frame (sequence advances).
    const auto deadline = steady_clock::now() + milliseconds(1500);
    PointCPtr cloud_in;
    while (steady_clock::now() < deadline) {
      std::this_thread::sleep_for(milliseconds(30));
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (g_cloud_sequence_ > last_seq && !g_cloud_ptr->empty()) {
        cloud_in = g_cloud_ptr;
        cloud_frame = g_input_pc_frame_id_;
        last_seq = g_cloud_sequence_;
        break;
      }
    }
    if (!cloud_in || cloud_in->empty() || cloud_frame.empty()) {
      RCLCPP_WARN(node_->get_logger(),
        "detectShapePose: no fresh cloud for frame %d, skipping",
        frame_idx);
      continue;
    }

    geometry_msgs::msg::TransformStamped tfs;
    try {
      tfs = tf_buffer_.lookupTransform(
        planning_frame, cloud_frame, tf2::TimePointZero,
        tf2::durationFromSec(1.0));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(node_->get_logger(),
        "detectShapePose: tf %s -> %s failed: %s, skipping frame",
        cloud_frame.c_str(), planning_frame.c_str(), e.what());
      continue;
    }
    const Eigen::Quaterniond Rq(
      tfs.transform.rotation.w, tfs.transform.rotation.x,
      tfs.transform.rotation.y, tfs.transform.rotation.z);
    const Eigen::Vector3d Tt(
      tfs.transform.translation.x, tfs.transform.translation.y,
      tfs.transform.translation.z);

    const double z_lo = 0.028;
    const double z_hi = 0.110;
    const double roi_r2 = 0.15 * 0.15;
    for (const auto & p : *cloud_in) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      const Eigen::Vector3d pw =
        Rq * Eigen::Vector3d(p.x, p.y, p.z) + Tt;
      if (pw.z() < z_lo || pw.z() > z_hi) continue;
      const double dx = pw.x() - obj_xy.x;
      const double dy = pw.y() - obj_xy.y;
      if (dx * dx + dy * dy > roi_r2) continue;
      const int r = p.r, g = p.g, b = p.b;
      if (g > r + 25 && g > b + 25) continue;
      xy_pts.emplace_back(pw.x(), pw.y());
    }
  }
  if (xy_pts.size() < 30) {
    RCLCPP_ERROR(node_->get_logger(),
      "detectShapePose: only %zu points after %d-frame accumulation",
      xy_pts.size(), kNumFrames);
    return false;
  }

  // Cluster mean - kept for diagnostics only. The mean is biased
  // toward whichever side of the shape has more visible points (the
  // wrist camera is offset, so this can be 1-2 cm away from the
  // true centre). We do NOT use it as the M4 pivot anymore.
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto & q : xy_pts) centroid += q;
  centroid /= static_cast<double>(xy_pts.size());

  // YAW: SHAPE-AWARE estimator anchored on obj_xy.
  //
  //   For NOUGHT (square ring): use minimum-area bounding rectangle
  //   (MABR). The nought's outer envelope is a filled square, so
  //   bbox area is minimised when the shape's sides align with the
  //   test frame axes - giving the correct yaw directly.
  //
  //   For CROSS (+ shape): use the 4th-order complex moment M4.
  //   MABR is WRONG for cross because the bbox of a + at any
  //   rotation depends on arm-tip projections plus arm thickness,
  //   and the minimum-area rotation is at psi - pi/4 (NOT psi).
  //   For an axis-aligned cross, MABR returns +-pi/4. A grasp aimed
  //   at +X arm at world angle = MABR-yaw lands in the empty space
  //   between two arms - exactly the user's reported "fingers
  //   close on air" failure for crosses (e.g. detected -0.676,
  //   true cross arms at +0.109 etc, no arm at -0.676 -> miss).
  //   M4 is the right tool: M4 = sum (z - pivot)^4 has phase 4*psi
  //   for the cross's 4-fold symmetric mass distribution, so
  //   recovered yaw = arg(M4) / 4 is the actual arm direction.
  const Eigen::Vector2d pivot(obj_xy.x, obj_xy.y);
  double best_yaw = 0.0;

  if (shape_type == "nought") {
    // MABR helper: bbox area after rotating cluster by -theta.
    auto bbox_area = [&](double theta) -> double {
      const double cs = std::cos(-theta), sn = std::sin(-theta);
      double xmin =  std::numeric_limits<double>::infinity();
      double xmax = -std::numeric_limits<double>::infinity();
      double ymin =  std::numeric_limits<double>::infinity();
      double ymax = -std::numeric_limits<double>::infinity();
      for (const auto & q : xy_pts) {
        const double dx = q.x() - pivot.x();
        const double dy = q.y() - pivot.y();
        const double rx = cs * dx - sn * dy;
        const double ry = sn * dx + cs * dy;
        if (rx < xmin) xmin = rx;
        if (rx > xmax) xmax = rx;
        if (ry < ymin) ymin = ry;
        if (ry > ymax) ymax = ry;
      }
      return (xmax - xmin) * (ymax - ymin);
    };

    constexpr int kCoarseSteps = 90;
    const double half_range = M_PI / 4.0;
    double best_area = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= kCoarseSteps; ++i) {
      const double theta =
        -half_range + 2.0 * half_range * i / kCoarseSteps;
      const double a = bbox_area(theta);
      if (a < best_area) { best_area = a; best_yaw = theta; }
    }
    constexpr int kFineSteps = 100;
    const double fine_range = M_PI / 90.0;
    const double coarse_min = best_yaw;
    for (int i = -kFineSteps; i <= kFineSteps; ++i) {
      const double theta = coarse_min + fine_range * i / kFineSteps;
      if (theta < -half_range || theta > half_range) continue;
      const double a = bbox_area(theta);
      if (a < best_area) { best_area = a; best_yaw = theta; }
    }
  } else {
    // CROSS: 4th-order complex moment around obj_xy. The shape's
    // 4-fold symmetric mass distribution gives M4 phase = 4 * psi
    // (cross has phi_shape=0 in our earlier derivation: arms
    // contribute z^4 = +t^4 along axes, no sign flip).
    std::complex<double> M4(0.0, 0.0);
    double r4_sum = 0.0;  // sum |z|^4, for normalised |M4| diagnostic
    for (const auto & q : xy_pts) {
      const std::complex<double> z(q.x() - pivot.x(),
                                    q.y() - pivot.y());
      const std::complex<double> z2 = z * z;
      const std::complex<double> z4 = z2 * z2;
      M4 += z4;
      r4_sum += std::abs(z4);
    }
    const double m4_arg = std::arg(M4);  // in (-pi, pi]
    const double m4_mag = std::abs(M4);
    // Coherence: 1.0 means all z^4 vectors point the same way (perfect
    // C4 alignment). <0.3 means M4 is dominated by noise / asymmetric
    // sampling and the argument has no meaning.
    const double m4_coh = (r4_sum > 1e-12) ? (m4_mag / r4_sum) : 0.0;
    double yaw_raw = 0.25 * m4_arg;
    const double yaw_pre_fold = yaw_raw;
    while (yaw_raw >  M_PI / 4.0) yaw_raw -= M_PI / 2.0;
    while (yaw_raw < -M_PI / 4.0) yaw_raw += M_PI / 2.0;
    best_yaw = yaw_raw;

    // For comparison against MABR-at-this-pivot. If M4 is wrong (low
    // coherence or fingers-clip-corner failures), we'd expect the
    // MABR-best yaw at this pivot to disagree by ~pi/4 (cross arms
    // vs cross diagonal). Compare raw OBB area at coarse steps;
    // duplicates the MABR cost to avoid coupling logic to control flow.
    const int kDiagSteps = 18;  // 5-deg granularity over [-pi/4,pi/4]
    double mabr_best_yaw = 0.0, mabr_best_area = std::numeric_limits<double>::infinity();
    for (int i = -kDiagSteps; i <= kDiagSteps; ++i) {
      const double theta = (M_PI / 4.0) * i / kDiagSteps;
      const double cs_t = std::cos(-theta), sn_t = std::sin(-theta);
      double xlo =  std::numeric_limits<double>::infinity();
      double xhi = -std::numeric_limits<double>::infinity();
      double ylo =  std::numeric_limits<double>::infinity();
      double yhi = -std::numeric_limits<double>::infinity();
      for (const auto & q : xy_pts) {
        const double dx = q.x() - pivot.x(), dy = q.y() - pivot.y();
        const double rx = cs_t * dx - sn_t * dy;
        const double ry = sn_t * dx + cs_t * dy;
        if (rx < xlo) xlo = rx;
        if (rx > xhi) xhi = rx;
        if (ry < ylo) ylo = ry;
        if (ry > yhi) yhi = ry;
      }
      const double area = (xhi - xlo) * (yhi - ylo);
      if (area < mabr_best_area) { mabr_best_area = area; mabr_best_yaw = theta; }
    }
    // Branch difference: how far apart M4 and MABR are after C4 folding.
    double dyaw = best_yaw - mabr_best_yaw;
    while (dyaw >  M_PI / 4.0) dyaw -= M_PI / 2.0;
    while (dyaw < -M_PI / 4.0) dyaw += M_PI / 2.0;

    // ROLLED BACK: previously when |dyaw|>0.4 we overrode best_yaw
    // with mabr_best_yaw on the assumption MABR was always right
    // when the two disagreed. EVIDENCE that this was wrong: log run
    // had M4=0.066, MABR=-0.698 (diff=+0.764), the override picked
    // MABR, and the gripper closed on air at width=0.0100 - MABR
    // was off by pi/4 in that case. Both estimators can fail with
    // ~10% probability; we cannot pick the right one from the
    // first-view data alone. Now we always return M4 as primary
    // and expose MABR as the alternative; the caller is expected
    // to break the tie via a second observation when they disagree.
    if (out_alt_yaw) {
      *out_alt_yaw = mabr_best_yaw;
    }

    RCLCPP_INFO(node_->get_logger(),
      "detectShapePose[cross-M4]: pivot=(%.3f,%.3f) N=%zu "
      "arg(M4)=%.3f -> /4=%.3f -> folded=%.3f  "
      "|M4|=%.3e r4_sum=%.3e coh=%.3f  "
      "MABR_yaw=%.3f branch_diff=%.3f",
      pivot.x(), pivot.y(), xy_pts.size(),
      m4_arg, yaw_pre_fold, best_yaw,
      m4_mag, r4_sum, m4_coh,
      mabr_best_yaw, dyaw);
  }

  // For nought, no ambiguity exists; set alt = primary so caller's
  // diff check is a no-op. (For cross, the else-branch above has
  // already populated *out_alt_yaw.)
  if (out_alt_yaw && shape_type == "nought") {
    *out_alt_yaw = best_yaw;
  }

  out_yaw = best_yaw;
  double yaw = best_yaw;  // alias for the existing OBB / size code below

  // Oriented bounding box in the (mean-centroid, detected-yaw) frame.
  // The mean centroid is biased toward the far-from-camera side because
  // the wrist camera sits +X +Y of link8 at the observation pose, and
  // the open gripper shadows the near-camera face of the shape, leaving
  // more cluster points on the far face. The OBB midpoint, by contrast,
  // is the average of the EXTREME points along each axis; provided the
  // far-edge points along all four sides are visible (which they are
  // for this top-down + small camera offset) the midpoint sits at the
  // true geometric centre and we can correct the bias one-shot.
  const double cs = std::cos(-yaw), sn = std::sin(-yaw);
  double x_lo =  std::numeric_limits<double>::infinity();
  double x_hi = -std::numeric_limits<double>::infinity();
  double y_lo =  std::numeric_limits<double>::infinity();
  double y_hi = -std::numeric_limits<double>::infinity();
  double max_half = 0.0;
  for (const auto & q : xy_pts) {
    const double dx = q.x() - centroid.x(), dy = q.y() - centroid.y();
    const double rx = cs * dx - sn * dy;
    const double ry = sn * dx + cs * dy;
    if (rx < x_lo) x_lo = rx;
    if (rx > x_hi) x_hi = rx;
    if (ry < y_lo) y_lo = ry;
    if (ry > y_hi) y_hi = ry;
    max_half = std::max(max_half,
                        std::max(std::fabs(rx), std::fabs(ry)));
  }

  // Span half-extent (for diagnostics / centroid only). We do NOT use
  // this for the size snap because the depth sensor strips ~10-15 mm
  // off each outer edge of the shape, so the OBB extent reads
  // systematically short (a 30 mm nought lands at obb_half ~ 0.060
  // when the true outer half is 0.075). max_half below is more
  // robust because it captures the far extreme from the biased mean,
  // which roughly cancels the edge stripping for our setup.
  const double obb_half = std::max(x_hi - x_lo, y_hi - y_lo) / 2.0;
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

  // OBB midpoint (in rotated frame) is the bias offset from mean
  // centroid to true centre. Rotate it back into world coords and
  // anchor on the mean centroid to recover the true world centre.
  const double mid_local_x = 0.5 * (x_lo + x_hi);
  const double mid_local_y = 0.5 * (y_lo + y_hi);
  const double cs_b = std::cos(yaw), sn_b = std::sin(yaw);
  out_cx = centroid.x() + cs_b * mid_local_x - sn_b * mid_local_y;
  out_cy = centroid.y() + sn_b * mid_local_x + cs_b * mid_local_y;

  RCLCPP_INFO(node_->get_logger(),
    "detectShapePose[%s]: pts=%zu mean=(%.3f,%.3f) yaw=%.3f "
    "obb_xlo/hi=(%.3f,%.3f) obb_ylo/hi=(%.3f,%.3f) obb_half=%.3f "
    "max_half=%.3f raw_s=%.3f -> size=%.3f corr=(%.3f,%.3f)",
    shape_type.c_str(), xy_pts.size(), centroid.x(), centroid.y(),
    out_yaw,
    x_lo, x_hi, y_lo, y_hi,
    obb_half, max_half, s_raw, out_size,
    out_cx, out_cy);

  return true;
}


bool cw2::t1_pickAndPlace(
  const geometry_msgs::msg::Point & obj,
  const geometry_msgs::msg::Point & goal,
  const std::string & shape_type)
{
  // EVIDENCE-DRIVEN PROLOGUE: previous failed runs can leak the
  // held_shape AttachedCollisionObject and the t1_shape_obstacle into
  // the planning scene. The next task then sees them at start state
  // and the very first plan fails with 'Computed path is not valid.
  // Invalid states at index locations: [0 1 2]' and
  //   't1_shape_obstacle' ... 'panda_hand' ... constitutes a collision
  //   'ground_plane' ... 'held_shape' (Robot attached) ... constitutes a collision
  // (see /tmp/cw2.log timestamps 79819.46 and 79832.21 of the prior
  // run.) Clearing both unconditionally at task start gives every
  // task a clean planning scene regardless of how the previous one
  // ended. Both calls are idempotent - they no-op when the object
  // doesn't exist.
  detachShape();
  removeShapeCollision();
  removeTileCollision();  // also clear any stale tile slab from a
                          // failed transit that didn't reach removeTileCollision
  removeBasketCollision();  // and any stale basket walls from a prior task

  // Add the basket as four wall collision objects so the joint-space
  // observation transit refuses paths that sweep the arm through
  // where the basket physically sits. The user observed the arm
  // dipping low and shoving the basket out of place during the
  // observation move - cause: nothing in the planning scene was
  // modelling the basket. EVIDENCE that this still happens
  // intermittently: cw2.log run had a frame where addBasketCollision
  // verify timed out at 1 s ("basket walls did not appear within 1 s")
  // and the next observation move physically knocked the basket.
  // Now we return false on verify timeout and the caller aborts.
  if (!addBasketCollision(goal)) {
    RCLCPP_ERROR(node_->get_logger(),
      "addBasketCollision failed (planning scene did not register "
      "walls); aborting task to avoid knocking the basket.");
    removeBasketCollision();
    return false;
  }

  // Open first so the fingers don't occlude the wrist camera during
  // observation.
  if (!openGripper()) return false;

  // Register the shape as a collision obstacle so the joint-space
  // planner used by moveArmToPose() routes the elbow around it
  // instead of swinging through the column above obj.
  addShapeCollision(obj, shape_type);

  // Overhead observation pose directly above obj.
  if (!moveArmToPose(
      makeTopDownPose(obj.x, obj.y, SAFE_ALTITUDE, 0.0),
      "safe-above-observation")) {
    removeShapeCollision();
    return false;
  }

  double yaw = 0.0;
  double size_s = 0.040;
  double cx = obj.x, cy = obj.y;
  double alt_yaw = 0.0;
  if (!detectShapePose(obj, shape_type, yaw, size_s, cx, cy, &alt_yaw)) {
    // Cloud went silent for 4+ seconds (Gazebo depth-camera plugin
    // transient - we have evidence of this between missions).
    // Sleep, then retry the full 3-frame accumulation once before
    // aborting. The cloud subscriber is already on a Reentrant
    // callback group so the issue is not at our end.
    RCLCPP_WARN(node_->get_logger(),
      "perception failed first attempt - sleeping 2s and retrying");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    if (!detectShapePose(obj, shape_type, yaw, size_s, cx, cy, &alt_yaw)) {
      RCLCPP_ERROR(node_->get_logger(),
        "perception failed AFTER retry - aborting task. Falling back "
        "to yaw=0/size=40mm produces wrong grasps on tilted shapes; "
        "better to abort cleanly than mis-grasp.");
      removeShapeCollision();
      removeBasketCollision();
      return false;
    }
    RCLCPP_INFO(node_->get_logger(),
      "perception recovered on retry");
  }

  // CROSS YAW TIEBREAKER. EVIDENCE this is needed: cw2.log has runs
  // where M4 was wrong (M4 picked diagonal branch, MABR correct) AND
  // separate runs where MABR was wrong (M4 picked arm-aligned arms,
  // MABR off by pi/4). Both estimators have ~10% failure rate on
  // small / partially-occluded crosses, and we cannot tell which is
  // wrong from a single viewpoint. So when they disagree by more
  // than 0.4 rad we move to a SECOND observation pose offset
  // laterally (different sampling/occlusion pattern) and re-run
  // detectShapePose. We then pick whichever first-view candidate is
  // closer to the second-view's M4 estimate. The TRUE yaw should be
  // consistent across viewpoints; sampling artifacts that flip M4
  // are not.
  if (shape_type == "cross") {
    double diff_first = yaw - alt_yaw;
    while (diff_first >  M_PI / 4.0) diff_first -= M_PI / 2.0;
    while (diff_first < -M_PI / 4.0) diff_first += M_PI / 2.0;
    if (std::fabs(diff_first) > 0.4) {
      RCLCPP_WARN(node_->get_logger(),
        "T1[cross]: yaw ambiguous (M4=%.3f, MABR=%.3f, diff=%.3f) - "
        "doing second observation 60mm offset",
        yaw, alt_yaw, diff_first);

      // Move to a second observation pose offset 60 mm in +X and
      // 40 mm in -Y (asymmetric so any directional sampling bias
      // shifts measurably). Same SAFE_ALTITUDE so transit is
      // short and obstacle-free.
      const double obs2_x = obj.x + 0.060;
      const double obs2_y = obj.y - 0.040;
      if (!moveArmToPose(
          makeTopDownPose(obs2_x, obs2_y, SAFE_ALTITUDE, 0.0),
          "second-observation-tiebreaker")) {
        RCLCPP_WARN(node_->get_logger(),
          "T1[cross]: second-obs move failed - keeping first-view "
          "M4 as best guess");
      } else {
        double yaw2 = 0.0, size2 = 0.0, cx2 = 0.0, cy2 = 0.0;
        double alt_yaw2 = 0.0;
        if (!detectShapePose(obj, shape_type, yaw2, size2, cx2, cy2,
                             &alt_yaw2)) {
          RCLCPP_WARN(node_->get_logger(),
            "T1[cross]: second-obs detect failed - keeping first-view "
            "M4 as best guess");
        } else {
          // Compare second-view M4 to both first-view candidates,
          // folded into [-pi/4, pi/4] so the comparison is C4-aware.
          auto fold = [](double a) {
            while (a >  M_PI / 4.0) a -= M_PI / 2.0;
            while (a < -M_PI / 4.0) a += M_PI / 2.0;
            return a;
          };
          const double d_to_M4   = std::fabs(fold(yaw2 - yaw));
          const double d_to_MABR = std::fabs(fold(yaw2 - alt_yaw));
          if (d_to_MABR < d_to_M4) {
            RCLCPP_INFO(node_->get_logger(),
              "T1[cross]: tiebreaker chose MABR (yaw2=%.3f, "
              "d(M4)=%.3f, d(MABR)=%.3f)",
              yaw2, d_to_M4, d_to_MABR);
            yaw = alt_yaw;
          } else {
            RCLCPP_INFO(node_->get_logger(),
              "T1[cross]: tiebreaker chose M4 (yaw2=%.3f, "
              "d(M4)=%.3f, d(MABR)=%.3f)",
              yaw2, d_to_M4, d_to_MABR);
            // yaw stays at first-view M4
          }
          // Prefer the second view's centre estimate too, since the
          // ambiguity-triggering first view had biased sampling.
          cx = cx2;
          cy = cy2;
        }
      }
    }
  }

  // Grasp offset in shape-local frame, then rotate into world.
  const bool is_cross = (shape_type == "cross");
  const double ox_local = is_cross ? (1.5 * size_s) : 0.0;
  const double oy_local = is_cross ? 0.0           : (2.0 * size_s);
  const double c_y = std::cos(yaw), s_y = std::sin(yaw);
  const double ox_world = c_y * ox_local - s_y * oy_local;
  const double oy_world = s_y * ox_local + c_y * oy_local;

  // Anchor the grasp on the SPAWNER-REPORTED obj rather than the
  // perception centroid. In this Gazebo setup the shapes do not
  // measurably drift during settling (high-friction, 5 mm drop),
  // so obj IS the true geometric centre. The cluster centroid we
  // compute from the depth cloud is biased - even after the
  // OBB-midpoint correction, the cross's vertical arm gets
  // asymmetric edge stripping that residually shifts the midpoint
  // by ~10 mm. obj has none of that error.
  // (cx/cy is still computed and logged as a sanity-check signal -
  // if obj and cx/cy ever diverge by > ~25 mm, perception is broken
  // and the discrepancy will surface in the T1 log line.)
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

  // Neutral yaw used for the held-shape transit and for the place
  // descent. Set to -pi/4 (NOT 0) because the hand-to-world rotation
  // chain Rz(link8_yaw) * Rx(pi) * Rz(-pi/4) gives the shape's +X
  // world axis at angle (link8_yaw + 5*pi/4). For the held shape's
  // sides to align with world X/Y (mod pi/2 because both shapes are
  // 4-fold symmetric), link8_yaw must satisfy
  //   link8_yaw + 5*pi/4 == 0 (mod pi/2)
  // which folds to link8_yaw = -pi/4.
  // EVIDENCE for why this matters: at transit_yaw = 0 the held shape
  // sat at world yaw pi/4 (a 45 deg-rotated diamond), with corners
  // sticking 0.106 m from TCP instead of 0.075 m for a 30 mm nought.
  // Those corners caught on basket walls during the place descent
  // (Cartesian fractioned at 80.4%) and ended up rotated relative to
  // the basket cavity when released, which is what the user observed
  // visually. With transit_yaw = -pi/4 the shape's bbox in world is
  // tight (+/- 2.5*s in each axis) and aligned with the basket walls.
  const double transit_yaw = -M_PI / 4.0;

  // Place TCP must put the SHAPE CENTER at goal, given that the
  // gripper will be at link8_yaw = transit_yaw (= -pi/4) when
  // placing, not at grasp_yaw. The shape center sits in hand frame at
  // (ox_local, -oy_local, EE_TO_FINGER); we rotate that hand-frame
  // offset into world via the link8 RPY(pi, 0, transit_yaw)
  // composition, then negate so that link8 (= TCP) is offset from
  // goal by minus that vector.
  // Derivation: world = Rz(transit_yaw) * Rx(pi) * Rz(-pi/4) * hand
  //   where Rz(-pi/4) is the fixed link8->hand rotation,
  //         Rx(pi)    is the link8 RPY(pi,0,_) roll,
  //         Rz(transit_yaw) is the link8 yaw.
  // CORRECTED SIGNS: at link8_yaw = -pi/4, hand_X_world = (+1, 0, 0)
  // (= shape +X axis direction in world, since shape +X aligns with
  // world +X for an axis-aligned shape) and hand_Y_world = (0, -1, 0)
  // (= NEGATIVE shape +Y axis direction). So in hand frame, shape +X
  // axis = +hand_X and shape +Y axis = -hand_Y. Therefore the shape
  // CENTER (which sits at shape-frame (-ox_local, -oy_local) relative
  // to TCP) maps to hand-frame (-ox_local, +oy_local, EE_TO_FINGER).
  // The previous code had the signs flipped on both, which placed the
  // cross 60 mm in the wrong direction (logged place=(-0.470,-0.360)
  // for goal=(-0.410,-0.360) put the cross's west arm tip into the
  // west wall - exactly the symptom the user reported).
  const double sc_hx = -ox_local;
  const double sc_hy =  oy_local;
  // Rz(-pi/4) * (sc_hx, sc_hy):
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  const double sc_lx = (sc_hx + sc_hy) * inv_sqrt2;
  const double sc_ly = (-sc_hx + sc_hy) * inv_sqrt2;
  // Rx(pi) flips y-sign (also z-sign, but z drops out of XY):
  const double sc_px = sc_lx;
  const double sc_py = -sc_ly;
  // Rz(transit_yaw):
  const double cs_t = std::cos(transit_yaw);
  const double sn_t = std::sin(transit_yaw);
  const double sc_wx = sc_px * cs_t - sc_py * sn_t;
  const double sc_wy = sc_px * sn_t + sc_py * cs_t;
  // For shape_center_world = goal: TCP_world = goal - shape_center_offset.
  const double place_x = goal.x - sc_wx;
  const double place_y = goal.y - sc_wy;

  const double grasp_ee_z  = obj.z + SHAPE_LINK_Z_OFFSET
                              + SHAPE_THICKNESS / 2.0 + EE_TO_FINGER;
  const double pre_grasp_z = grasp_ee_z + APPROACH_DIST;
  const double place_ee_z  = goal.z + BASKET_FLOOR_OFFSET
                              + SHAPE_THICKNESS / 2.0 + EE_TO_FINGER;
  const double pre_place_z = place_ee_z + APPROACH_DIST;
  const double safe_z      = std::max({SAFE_ALTITUDE, pre_grasp_z + 0.05,
                                       pre_place_z + 0.05});

  const double close_w = graspCloseWidth(size_s);

  // Open-finger world positions, for debugging east/west errors.
  // hand_Y_world_angle = link8_yaw - pi/4 (derived from RPY(pi,0,link8_yaw)
  // applied first, then the hand's fixed -pi/4 yaw on link8). Fingers
  // open along hand_Y axis to +/- 0.035 m when the gripper is at "open".
  const double hand_y_ang = link8_yaw - M_PI / 4.0;
  const double hyx = std::cos(hand_y_ang);
  const double hyy = std::sin(hand_y_ang);
  const double f1x = grasp_x + 0.035 * hyx;
  const double f1y = grasp_y + 0.035 * hyy;
  const double f2x = grasp_x - 0.035 * hyx;
  const double f2y = grasp_y - 0.035 * hyy;

  RCLCPP_INFO(node_->get_logger(),
    "T1[%s] s=%.3f yaw=%.3f obj=(%.3f,%.3f,%.3f) c=(%.3f,%.3f) "
    "off=(%.3f,%.3f) grasp=(%.3f,%.3f,%.3f) link8_yaw=%.3f "
    "open_f1=(%.3f,%.3f) open_f2=(%.3f,%.3f) place=(%.3f,%.3f,%.3f) close=%.3f",
    shape_type.c_str(), size_s, yaw,
    obj.x, obj.y, obj.z, cx, cy,
    ox_world, oy_world,
    grasp_x, grasp_y, grasp_ee_z, link8_yaw,
    f1x, f1y, f2x, f2y,
    place_x, place_y, place_ee_z, close_w);

  // Constant-Z translate + yaw rotate at safe_z. Cartesian here
  // forces the EE (and elbow) to stay at safe altitude throughout
  // the rotation; joint-space planning is free to swing the elbow
  // low through the obstacle box, which we want to avoid.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, safe_z, link8_yaw)},
      0.01, 0.0, /*allow_fallback=*/false)) {
    removeShapeCollision();
    return false;
  }

  // We're now directly above the grasp at safe altitude. Take the
  // shape obstacle out so the descent isn't blocked by it.
  removeShapeCollision();

  // Cartesian descent down to the shape. We disable the joint-space
  // fallback here because joint planning can sweep the arm through
  // the object.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, pre_grasp_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;
  // Final descent to grasp pose: tighter eef_step than the
  // pre_grasp descent. computeCartesianPath calls IK at every
  // eef_step boundary; smaller step means more checkpoints, less
  // joint reconfiguration between checkpoints, less lateral EE
  // drift on tilted-arm configurations. Targets the user's
  // outer-finger-clips-wall-during-descent failure mode for
  // mid-large noughts where the open-finger margin to the wall
  // is only ~20 mm.
  //
  // Additionally, enforce a PositionConstraint on panda_link8
  // throughout this descent: a thin XY box (+/-5 mm) centered on
  // the descent column. This forces the IK solver at every
  // checkpoint to find joint solutions whose link8 origin (and
  // therefore the TCP, which is just below it on a top-down grasp)
  // stays inside the column. Eliminates the 5-10 mm lateral drift
  // that lets the outer finger clip the basket wall on mid-large
  // noughts. If the constraint makes IK infeasible we fall back to
  // the unconstrained descent rather than abort the whole grasp.
  {
    moveit_msgs::msg::Constraints descent_constraints;
    moveit_msgs::msg::PositionConstraint pc;
    pc.header.frame_id = arm_group_->getPlanningFrame();
    pc.link_name = "panda_link8";
    pc.target_point_offset.x = 0.0;
    pc.target_point_offset.y = 0.0;
    pc.target_point_offset.z = 0.0;
    pc.weight = 1.0;
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions.resize(3);
    // half_xy = 0.005 -> full width 0.010. Z spans the descent
    // (pre_grasp_z down to grasp_ee_z) plus a small margin so the
    // checkpoint at either endpoint isn't on the box face.
    box.dimensions[0] = 0.010;
    box.dimensions[1] = 0.010;
    box.dimensions[2] = (pre_grasp_z - grasp_ee_z) + 0.020;
    pc.constraint_region.primitives.push_back(box);
    geometry_msgs::msg::Pose region_pose;
    region_pose.position.x = grasp_x;
    region_pose.position.y = grasp_y;
    region_pose.position.z = (pre_grasp_z + grasp_ee_z) / 2.0;
    region_pose.orientation.w = 1.0;
    pc.constraint_region.primitive_poses.push_back(region_pose);
    descent_constraints.position_constraints.push_back(pc);

    if (!moveArmCartesianConstrained(
        {makeTopDownPose(grasp_x, grasp_y, grasp_ee_z, link8_yaw)},
        descent_constraints,
        0.002, 0.0, /*retry_unconstrained=*/true)) return false;
  }

  if (!closeGripper(close_w)) return false;

  // POST-GRASP VERIFY. The hand is driven by panda_hand_controller via
  // a JointTrajectory topic publish. Empirical evidence: ALL 8 runs in
  // the prior log (7 successful, 1 failed) showed width=0.0699 at the
  // verify time -> we were reading BEFORE the close trajectory had
  // even started executing (controller takes ~2 s; our previous 200
  // ms sleep was way too short). The successful runs only worked
  // because the close happened ASYNCHRONOUSLY during the lift/transit.
  // The failed run was the case where the close ALSO didn't happen
  // during transit (or the shape was knocked away).
  //
  // Two fixes:
  //   1) Wait long enough for the close to actually complete. Poll
  //      /joint_states until either both fingers stop changing more
  //      than 0.5 mm between samples, or a hard 3 s deadline.
  //   2) Check BOTH bounds. If width < close_w + 2 mm, fingers closed
  //      on air. If width > size_s + 8 mm, fingers never closed
  //      (still at the open ~70 mm). Either way: abort.
  // Action-client commandGripper already blocked on the controller's
  // result, so fingers are at their final position. With joint_states
  // on a reentrant callback group the cache is always fresh; we just
  // sleep one publish-period (~40 ms at 25 Hz) to make sure we see
  // the post-trajectory state, not the last pre-close one.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    double f1 = 0.0, f2 = 0.0;
    bool seen = false;
    {
      std::lock_guard<std::mutex> lk(joint_states_mutex_);
      f1 = finger1_pos_;
      f2 = finger2_pos_;
      seen = finger_state_seen_;
    }

    const double total_width = f1 + f2;
    const double lower = close_w + 0.005;
    // Upper bound = size_s + 0.015. The +0.015 absorbs ODE
    // min_depth=5e-3 contact penetration (per finger) plus a few
    // mm of slack. Without this, a successful grip with deep
    // contact penetration (e.g. width=0.039 on s=0.030 wall = both
    // fingers 4.5 mm INTO the wall material due to controller push)
    // would be rejected as "fingers near open width" even though
    // physically the wall is firmly clamped. The fully-open case
    // (width = 2 x 0.040 = 0.080) is still well above 0.055 (=
    // 0.040 + 0.015 for the largest shape) so genuine open-fingers
    // failures are still caught.
    const double upper = size_s + 0.015;
    // Per-finger asymmetry check: a successful grip on a centred
    // wall stalls both fingers at ~s/2 each, so |f1 - f2| should
    // be < 5 mm. The user's failed run had f1=0.0379 (= URDF max
    // open joint limit, finger pushed back by collision) and
    // f2=-0.0003 (= fully closed, no shape on its side). The total
    // width 0.0376 happened to sit inside [0.015, 0.038] for an
    // s=0.030 shape, so the width-only check passed wrongly.
    // Adding |f1 - f2| < 0.005 catches the collision-pushed-open
    // pattern bilaterally.
    const double asymmetry = std::abs(f1 - f2);
    // Asymmetry threshold = size_s. Geometric reasoning: with TCP
    // offset d from wall centre, both fingers still contact the wall
    // iff d < s/2 (wall straddles TCP), and asymmetry = 2d. So
    // |f1-f2| < s iff bilateral contact. Above s, the wall is
    // entirely on one finger's side - only that one contacts and
    // the grip is unilateral / weak. EVIDENCE for the change:
    // a 20 mm nought run had f1=0.0049, f2=0.0151, width=0.020 (=s,
    // valid grip), asym=0.010 (TCP 5 mm off wall centre, both
    // fingers still on the wall). The fixed 5 mm threshold rejected
    // this real grip; size_s (=0.020) accepts it while still
    // catching the collision-pushed-open case (previous failure
    // had asym=0.038 with s=0.030, > 0.030 threshold -> caught).
    const double max_asymmetry = size_s;
    RCLCPP_INFO(node_->get_logger(),
      "grasp verify: seen=%d f1=%.4f f2=%.4f width=%.4f asym=%.4f "
      "expected=[%.4f, %.4f], asym<=%.4f (close_w=%.4f, size_s=%.4f)",
      seen, f1, f2, total_width, asymmetry,
      lower, upper, max_asymmetry, close_w, size_s);
    if (seen && total_width < lower) {
      RCLCPP_ERROR(node_->get_logger(),
        "grasp verify FAILED: width=%.4f below %.4f - fingers "
        "closed on air (no shape between them); aborting before lift",
        total_width, lower);
      openGripper();
      removeBasketCollision();
      return false;
    }
    if (seen && total_width > upper) {
      RCLCPP_ERROR(node_->get_logger(),
        "grasp verify FAILED: width=%.4f above %.4f - fingers never "
        "closed (still near open width); aborting before lift",
        total_width, upper);
      openGripper();
      removeBasketCollision();
      return false;
    }
    if (seen && asymmetry > max_asymmetry) {
      RCLCPP_ERROR(node_->get_logger(),
        "grasp verify FAILED: f1-f2 asymmetry %.4f > %.4f - one "
        "finger collided with the shape during descent and was "
        "pushed open while the other closed unobstructed; "
        "aborting before lift",
        asymmetry, max_asymmetry);
      openGripper();
      removeBasketCollision();
      return false;
    }
  }

  // Attach the held shape to the gripper as a collision object, so
  // joint-space planning during the long transit refuses paths that
  // would dip the shape into the tile. Must happen BEFORE the lift.
  attachShape(size_s, ox_local, oy_local);

  // Straight up at the grasp yaw.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, safe_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  // Rotate to a neutral link8 yaw (0.0) at safe_z, with constant XY.
  // EVIDENCE: in the previous failure the joint-space fallback for the
  // grasp->basket transit logged
  //   "Unable to sample any valid states for goal tree" x4
  // and timed out, because the EE-yaw was locked to the grasp yaw
  // (~-0.78 rad) all the way to (place_x, place_y). At that yaw, no
  // IK solution at the place pose was both reachable and collision-
  // free. Rotating to yaw=0 here gives the IK solver its full
  // redundancy back for the long XY transit. The shape's orientation
  // in the basket isn't graded so changing link8_yaw mid-flight is
  // free.
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, safe_z, transit_yaw)},
      0.05, 0.0, /*allow_fallback=*/true)) return false;

  // Long transit from grasp to over-basket via joint-space planning.
  // EVIDENCE for the previous attempt failing: in /tmp/cw2.log timestamp
  // 80929.66 the planner produced a trajectory with link8 at z = 0.19
  // through 0.30 m, all of which the post-processor flagged as
  //   "Position constraint violated on link 'panda_link8'. Desired:
  //    0.000000, 0.000000, 0.550000, current: 0.193, -0.201, 0.265"
  // RRTConnect doesn't natively respect Cartesian path constraints
  // - it ignored the box and the post-validator rejected the plan.
  //
  // What RRT DOES respect is the planning-scene collision world.
  // Adding a thin tile slab at z = [0.018, 0.020] during transit
  // forces the planner to keep the held_shape (attached at hand_z =
  // 0.1122 +/- 0.020 m) above it, i.e. link8.z > ~0.172 m, which
  // matches what the constraint was trying to enforce. The tile
  // slab is OFF during grasp (so it doesn't interfere with the
  // grasp descent's finger clearance) and OFF after detach.
  //
  // We do NOT use a Cartesian midpoint waypoint: the geometric
  // midpoint of typical (grasp, place) pairs lies within ~10 cm of
  // the Panda base axis (wrist-over-base singular region) and RRT
  // goal sampling fails there with
  //     "Unable to sample any valid states for goal tree"
  // (see /tmp/cw2.log timestamps 403.581-408.685 of an earlier run).
  addTileCollision();
  const bool transit_ok = moveArmToPose(
    makeTopDownPose(place_x, place_y, safe_z, transit_yaw),
    "grasp-to-basket-transit");
  removeTileCollision();
  if (!transit_ok) return false;

  // Descend at neutral yaw to drop the shape into the basket.
  if (!moveArmCartesian(
      {makeTopDownPose(place_x, place_y, place_ee_z, transit_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  if (!openGripper()) return false;

  // Settle pause: openGripper returns when the action client + a
  // short joint-state stability check report success, but the
  // physics simulator can still have residual contact between
  // finger pads and the just-released shape. Without this pause
  // the immediate Cartesian lift drags the shape on a finger pad
  // and tips it onto the basket wall (user-observed once: a cross
  // got dragged out of the basket and ended up on the wall).
  std::this_thread::sleep_for(std::chrono::milliseconds(350));

  // Shape is now in the basket - detach it from the gripper so the
  // retreat (and any subsequent task) doesn't see a phantom shape
  // attached to the hand.
  detachShape();

  // Retreat after place. Multi-stage strategy because the prior
  // single-call Cartesian retreat was failing at 0.0% on EVERY run
  // (see /tmp/cw2.log: repeated "moveArmCartesian: only 0.0%
  // retreat failed after place"). The arm was being left at the
  // basket location, and the next run's joint-space planner then
  // swept through wherever the next basket happened to be.
  //
  // Hypothesised cause of the 0% failure: the planning scene at
  // this instant has the held_shape AttachedCollisionObject just
  // detached + the basket_walls + the open fingers AND maybe a
  // stale physics state where the just-released shape is still
  // sitting between the wall corners and the open fingers. The
  // first Cartesian IK checkpoint inherits that "in-collision"
  // start state and rejects everything.
  //
  // Stage 1: Cartesian lift by APPROACH_DIST (10 cm) with
  // avoid_collisions=false. We're going straight up from inside an
  // open basket; the only thing we could hit is the basket walls
  // we own, and they're <50 mm tall so a 100 mm lift clears them.
  // Disabling collision avoidance bypasses any stale start-state
  // collision flag.
  // Stage 2: Joint-space plan to safe_z. Now well above the basket,
  // the planner has full freedom to find a path; it can choose any
  // elbow configuration.
  RCLCPP_INFO(node_->get_logger(),
      "post-place retreat from place=(%.3f,%.3f,%.3f) yaw=%.3f -> "
      "lift then go-home(ready)",
      place_x, place_y, place_ee_z, transit_yaw);

  // Stage 1: collision-disabled Cartesian lift.
  {
    moveit_msgs::msg::RobotTrajectory traj;
    const double lift_z = std::min(safe_z,
                                   place_ee_z + APPROACH_DIST);
    std::vector<geometry_msgs::msg::Pose> wps = {
      makeTopDownPose(place_x, place_y, lift_z, transit_yaw)};
    double frac = arm_group_->computeCartesianPath(
        wps, 0.005, 0.0, traj, /*avoid_collisions=*/false);
    if (frac >= 0.9) {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_ = traj;
      auto rc = arm_group_->execute(plan);
      if (rc != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(node_->get_logger(),
            "post-place retreat stage1 exec failed (%d)",
            static_cast<int>(rc.val));
      }
    } else {
      RCLCPP_WARN(node_->get_logger(),
          "post-place retreat stage1 cartesian only %.1f%% (collision-off)",
          frac * 100.0);
    }
  }

  // Stage 2: joint-space plan back to the SRDF named "ready" pose.
  // EVIDENCE for going home rather than lingering above the basket:
  // the prior retreat targeted (place_x, place_y, safe_z) which is
  // directly over the just-released basket, in a wrist orientation
  // (transit_yaw=-pi/4) that is not the natural arm posture. Each
  // subsequent run then had to plan a long swing across the entire
  // workspace from that awkward pose to the next observation pose,
  // and this manifested visually as the arm doing a "full 360"
  // between runs. Returning to a canonical home posture instead
  // makes every run start from the same predictable joint state,
  // shortens the next observation move, and pulls the arm
  // physically clear of the goal area.
  arm_group_->setNamedTarget("ready");
  moveit::planning_interface::MoveGroupInterface::Plan home_plan;
  auto home_rc = arm_group_->plan(home_plan);
  if (home_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(node_->get_logger(),
        "post-place retreat stage2 (plan to 'ready') failed (%d)",
        static_cast<int>(home_rc.val));
  } else {
    auto exec_rc = arm_group_->execute(home_plan);
    if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(node_->get_logger(),
          "post-place retreat stage2 (exec 'ready') failed (%d)",
          static_cast<int>(exec_rc.val));
    }
  }
  arm_group_->clearPoseTargets();

  removeBasketCollision();
  return true;
}


bool cw2::commandGripper(double per_finger_target, double duration_s)
{
  const double tgt = std::max(0.0, std::min(0.04, per_finger_target));

  // Wait for the action server (the controller) to be available.
  if (!hand_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
    RCLCPP_ERROR(node_->get_logger(),
      "commandGripper: panda_hand_controller follow_joint_trajectory "
      "action server not available");
    return false;
  }

  FjtAction::Goal goal_msg;
  goal_msg.trajectory.header.stamp = node_->now();
  goal_msg.trajectory.joint_names = {
    "panda_finger_joint1", "panda_finger_joint2"
  };

  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions = {tgt, tgt};
  pt.velocities = {0.0, 0.0};
  const auto ns =
    static_cast<int64_t>(std::max(0.05, duration_s) * 1e9);
  pt.time_from_start.sec = static_cast<int32_t>(ns / 1000000000);
  pt.time_from_start.nanosec = static_cast<uint32_t>(ns % 1000000000);
  goal_msg.trajectory.points.push_back(pt);

  // Tolerances: relaxed because shape contact stops the fingers short
  // of the commanded position by a few mm. We don't want the controller
  // to mark the goal as failed in that case.
  goal_msg.goal_time_tolerance.sec = 2;

  // Send goal. With the action client on a Reentrant callback group
  // and the multi-threaded executor running, we can simply wait on
  // the future from this thread - the action client's response
  // callbacks run on a different executor thread and unblock the
  // future for us. We do NOT use spin_until_future_complete, which
  // would deadlock because t1_callback is itself running on the
  // executor.
  auto goal_future = hand_action_client_->async_send_goal(goal_msg);
  if (goal_future.wait_for(std::chrono::seconds(2)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_->get_logger(),
      "commandGripper: timed out waiting for goal acceptance");
    return false;
  }

  auto goal_handle = goal_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(node_->get_logger(),
      "commandGripper: goal was REJECTED by panda_hand_controller");
    return false;
  }

  // Wait for the controller's result. ABORTED (code 4) is expected on
  // contact stop; we just log and let the post-grasp verify decide.
  // If the action result times out, fall through to a polling wait
  // on /joint_states - the controller may still be executing, or it
  // may have reached the goal and the result message just got lost.
  auto result_future = hand_action_client_->async_get_result(goal_handle);
  const auto status = result_future.wait_for(
    std::chrono::duration<double>(duration_s + 4.0));
  if (status == std::future_status::ready) {
    const auto wrapped = result_future.get();
    // GoalStatus enum: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED.
    RCLCPP_INFO(node_->get_logger(),
      "commandGripper: result code=%d (4=SUCCEEDED, 5=CANCELED, 6=ABORTED)",
      static_cast<int>(wrapped.code));
  } else {
    RCLCPP_WARN(node_->get_logger(),
      "commandGripper: action result timed out - falling through to "
      "joint_states polling to confirm finger motion");
  }

  // GROUND-TRUTH SETTLE: poll /joint_states until fingers stop
  // moving. This is the authoritative check - the action client
  // result can be lost, delayed, or report ABORTED for a successful
  // grasp, but fingers actually being at a stable position is what
  // matters for the verify downstream. Wait up to 2 s for stability.
  using namespace std::chrono;
  const auto settle_deadline = steady_clock::now() + milliseconds(2000);
  double prev_f1 = -1.0, prev_f2 = -1.0;
  int stable_count = 0;
  while (steady_clock::now() < settle_deadline) {
    double f1, f2;
    bool seen;
    {
      std::lock_guard<std::mutex> lk(joint_states_mutex_);
      f1 = finger1_pos_;
      f2 = finger2_pos_;
      seen = finger_state_seen_;
    }
    if (seen && std::abs(f1 - prev_f1) < 0.0005 &&
                std::abs(f2 - prev_f2) < 0.0005) {
      if (++stable_count >= 5) break;
    } else {
      stable_count = 0;
    }
    prev_f1 = f1;
    prev_f2 = f2;
    std::this_thread::sleep_for(milliseconds(40));
  }

  return true;
}

bool cw2::openGripper()
{
  // 0.040 m is the URDF upper joint limit (hand.xacro: upper="0.04").
  // EVIDENCE for using the maximum: during the descent to grasp, the
  // OUTSIDE finger sits at TCP + per_finger from the wall centerline.
  // Margin to the wall outer face = per_finger - s/2:
  //   s=0.020 -> 0.030 m at per_finger=0.035, 0.030 m at 0.040
  //   s=0.030 -> 0.020 m at per_finger=0.035, 0.025 m at 0.040
  //   s=0.040 -> 0.015 m at per_finger=0.035, 0.020 m at 0.040
  // The user observed consistent collision of the outside finger with
  // the wall TOP during descent on mid-large noughts (s>=0.030),
  // exactly the cases with smallest margin. Opening to the URDF
  // maximum gives 5 mm of extra clearance on every size and
  // increases the 40 mm-nought margin by 33%.
  const bool ok = commandGripper(0.040, 0.6);
  if (!ok) {
    RCLCPP_ERROR(node_->get_logger(), "openGripper: action call failed");
    return false;
  }
  return true;
}

bool cw2::closeGripper(double width)
{
  const double per_finger = width / 2.0;
  // commandGripper now blocks on the action result, so fingers are at
  // their final position when this returns. No additional sleep needed.
  const bool ok = commandGripper(per_finger, 1.0);
  if (!ok) {
    RCLCPP_WARN(node_->get_logger(),
      "closeGripper: action call failed, continuing to verify");
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


bool cw2::moveArmCartesianConstrained(
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  const moveit_msgs::msg::Constraints & path_constraints,
  double eef_step,
  double jump_threshold,
  bool retry_unconstrained)
{
  // computeCartesianPath in MoveIt 2 Humble has an overload that takes
  // a Constraints message. Each IK checkpoint along the linear EE path
  // is rejected unless it satisfies the constraints. This pins lateral
  // EE drift during the final descent (the failure mode where the
  // outer finger clips the basket wall on mid-large noughts because IK
  // chooses joint solutions whose EE drifts ~5-10 mm sideways from the
  // planned column).
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory,
    path_constraints, /*avoid_collisions=*/true);

  bool ok = (fraction >= 0.9);
  if (!ok) {
    RCLCPP_WARN(node_->get_logger(),
      "moveArmCartesianConstrained: constrained compute only %.1f%%, "
      "%s.",
      fraction * 100.0,
      retry_unconstrained ? "retrying without constraint"
                          : "no retry requested");
    if (!retry_unconstrained) return false;
    // Retry unconstrained so the descent still happens. The verify
    // step downstream of the descent catches any resulting bad grip.
    return moveArmCartesian(waypoints, eef_step, jump_threshold,
                            /*allow_fallback=*/false);
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  auto exec_rc = arm_group_->execute(plan);
  if (exec_rc != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(),
      "moveArmCartesianConstrained: exec failed (%d)",
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
  // Subfloor only (z = -0.020 to 0). Earlier we tried extending this
  // upward to z = +0.020 (covering the tile) so the joint-space
  // transit couldn't drag the held shape across the tile - but
  // EVIDENCE from /tmp/cw2.log timestamp 80255.802:
  //   'ground_plane' ... 'panda_rightfinger' ... constitutes a collision
  // shows the start state of closeGripper failed because MoveIt's
  // default robot+object collision padding (~10 mm each, total ~20
  // mm) brings the expanded finger volume (true finger min z =
  // link8.z - 0.1122 = 0.031 at grasp pose) into contact with the
  // expanded ground top. The held-shape-on-tile problem is now
  // handled separately by the path constraint applied during
  // transit, NOT by extending the ground.
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


void cw2::addShapeCollision(
  const geometry_msgs::msg::Point & obj,
  const std::string & shape_type)
{
  // Bounding box that closely tracks the shape itself. Worst-case
  // footprint is 5s x 5s at s=0.040 -> 0.20 x 0.20 m. The shape sits
  // on the tile spanning z = [0.020, 0.060]; we wrap that with a 20
  // mm margin -> z = [0.000, 0.080], 80 mm tall.
  //
  // EVIDENCE for shrinking from 0.25 m to 0.08 m: in the previous
  // failure log the joint-space planner returned a path whose
  // panda_rightfinger collided with t1_shape_obstacle at trajectory
  // indices 25 and 38 of 49, after post-smoothing. That happened
  // because the column extended to z=0.25 m, which is well into the
  // natural elbow/finger sweep envelope at 0.10-0.30 m. Capping the
  // column at z=0.08 means it only blocks links that actually intrude
  // on the shape's z-range, so smoothed paths can pass freely above.
  (void)shape_type;  // worst-case sizing covers both shapes

  moveit_msgs::msg::CollisionObject obstacle;
  obstacle.header.frame_id = arm_group_->getPlanningFrame();
  obstacle.id = "t1_shape_obstacle";

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions.resize(3);
  box.dimensions[0] = 0.20;
  box.dimensions[1] = 0.20;
  box.dimensions[2] = 0.08;

  geometry_msgs::msg::Pose pose;
  pose.position.x = obj.x;
  pose.position.y = obj.y;
  // Centre at z=0.040 = mid-shape; column spans [0, 0.080] in world.
  pose.position.z = 0.040;
  pose.orientation.w = 1.0;

  obstacle.primitives.push_back(box);
  obstacle.primitive_poses.push_back(pose);
  obstacle.operation = moveit_msgs::msg::CollisionObject::ADD;

  planning_scene_interface_.applyCollisionObjects({obstacle});
}


void cw2::removeShapeCollision()
{
  planning_scene_interface_.removeCollisionObjects({"t1_shape_obstacle"});
}


void cw2::attachShape(double size_s, double ox_local, double oy_local)
{
  // Held shape after closeGripper, modelled in panda_hand frame.
  // At grasp time the gripper is oriented so hand_Y aligns with
  // shape +Y (in world) and hand_X aligns with -shape +X (in world);
  // therefore in hand frame:
  //   shape_center = (ox_local, -oy_local, EE_TO_FINGER)
  // i.e. the SHAPE CENTER is offset from the gripper TCP by the
  // grasp offset, with hand_X having a sign flip relative to
  // shape_X. Modelling the box at this offset means the planning
  // scene sees the held shape where it actually is.
  moveit_msgs::msg::AttachedCollisionObject aco;
  aco.link_name = "panda_hand";
  aco.object.id = "held_shape";
  aco.object.header.frame_id = "panda_hand";

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions.resize(3);
  // 5*size_s in each lateral axis (worst-case bounding for either
  // shape with a few mm of margin), 40 mm thick.
  box.dimensions[0] = 5.0 * size_s;
  box.dimensions[1] = 5.0 * size_s;
  box.dimensions[2] = SHAPE_THICKNESS;

  geometry_msgs::msg::Pose box_pose;
  // Shape center in hand frame: at link8_yaw = -pi/4 (the place yaw
  // we use), hand_X = +shape_X and hand_Y = -shape_Y. Shape center is
  // at shape-frame (-ox_local, -oy_local) relative to the TCP, which
  // maps to hand-frame (-ox_local, +oy_local, EE_TO_FINGER).
  box_pose.position.x = -ox_local;
  box_pose.position.y =  oy_local;
  box_pose.position.z = EE_TO_FINGER;
  box_pose.orientation.w = 1.0;
  aco.object.primitives.push_back(box);
  aco.object.primitive_poses.push_back(box_pose);
  aco.object.operation = moveit_msgs::msg::CollisionObject::ADD;

  aco.touch_links = {
    "panda_hand", "panda_leftfinger", "panda_rightfinger"
  };

  planning_scene_interface_.applyAttachedCollisionObject(aco);
}


void cw2::detachShape()
{
  // Detach AND remove. Detach alone leaves the object floating at the
  // last hand pose as a non-attached collision object, which would
  // then block subsequent moves.
  moveit_msgs::msg::AttachedCollisionObject aco;
  aco.link_name = "panda_hand";
  aco.object.id = "held_shape";
  aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  planning_scene_interface_.applyAttachedCollisionObject(aco);
  planning_scene_interface_.removeCollisionObjects({"held_shape"});
}


void cw2::addTileCollision()
{
  moveit_msgs::msg::CollisionObject tile;
  tile.header.frame_id = arm_group_->getPlanningFrame();
  tile.id = "tile_top";

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions.resize(3);
  // 3 m x 3 m thin slab, 2 mm thick. Span z = [0.018, 0.020] - just
  // the very top of the tile. With held_shape attached at hand_z =
  // [0.0922, 0.1322] and MoveIt's default ~10 mm robot+10 mm object
  // padding, this forces link8.z > ~0.16 m throughout the transit,
  // which is enough margin to keep the held_shape from dragging.
  box.dimensions[0] = 3.0;
  box.dimensions[1] = 3.0;
  box.dimensions[2] = 0.002;

  geometry_msgs::msg::Pose pose;
  pose.position.z = 0.019;
  pose.orientation.w = 1.0;

  tile.primitives.push_back(box);
  tile.primitive_poses.push_back(pose);
  tile.operation = moveit_msgs::msg::CollisionObject::ADD;

  planning_scene_interface_.applyCollisionObjects({tile});
}


void cw2::removeTileCollision()
{
  planning_scene_interface_.removeCollisionObjects({"tile_top"});
}


bool cw2::addBasketCollision(const geometry_msgs::msg::Point & goal)
{
  // Basket geometry from cw2_world_spawner/models/basket/model.sdf:
  //   - base 350 x 350 x 9 mm at model origin
  //   - 4 walls 9 x 350 x 50 mm (or 350 x 9 x 50 mm) at +/- 0.17050 m
  //     in X or Y, centred at z = 0.025 in model frame (wall span
  //     z = [0, 0.050] in model frame)
  // The model is spawned at goal.xyz, so the basket walls in WORLD
  // span z = [goal.z, goal.z + 0.050]. We model the four walls as
  // collision boxes; we do NOT model the base because the place
  // descent has to land *on* the base, and the base is already
  // covered by ground_plane padding.
  const std::string frame = arm_group_->getPlanningFrame();
  const double inset = 0.17050;     // wall centre offset from basket centre
  const double wall_t = 0.009;      // wall thickness
  const double wall_h = 0.050;      // wall height
  const double wall_l = 0.350;      // wall length
  const double wall_z = goal.z + 0.025;  // wall centre z

  // EVIDENCE for using a different API: the prior run's log contained
  //   "Tried to remove world object 'basket_wall_xpos', but it does
  //    not exist in this scene"
  // at task end - meaning the previous applyCollisionObjects(vector)
  // call silently failed to add any walls. With no walls in the
  // scene the joint-space planner can swing through where the basket
  // physically sits, and on the third place attempt the user observed
  // the basket physically shifted. The fix is to add each wall with
  // its own applyCollisionObject call, then yield briefly to give
  // the planning scene monitor time to propagate the change before
  // the next motion-planning request.
  auto add_wall =
    [&](const std::string & id, double cx, double cy,
        double sx, double sy)
    {
      moveit_msgs::msg::CollisionObject w;
      w.header.frame_id = frame;
      w.id = id;
      shape_msgs::msg::SolidPrimitive box;
      box.type = shape_msgs::msg::SolidPrimitive::BOX;
      box.dimensions.resize(3);
      box.dimensions[0] = sx;
      box.dimensions[1] = sy;
      box.dimensions[2] = wall_h;
      geometry_msgs::msg::Pose p;
      p.position.x = cx;
      p.position.y = cy;
      p.position.z = wall_z;
      p.orientation.w = 1.0;
      w.primitives.push_back(box);
      w.primitive_poses.push_back(p);
      w.operation = moveit_msgs::msg::CollisionObject::ADD;
      planning_scene_interface_.applyCollisionObject(w);
    };

  add_wall("basket_wall_xpos", goal.x + inset, goal.y, wall_t, wall_l);
  add_wall("basket_wall_xneg", goal.x - inset, goal.y, wall_t, wall_l);
  add_wall("basket_wall_ypos", goal.x, goal.y + inset, wall_l, wall_t);
  add_wall("basket_wall_yneg", goal.x, goal.y - inset, wall_l, wall_t);

  // Yield + verify. Lambda factored out so we can retry the add
  // and re-verify on first failure.
  using namespace std::chrono;
  auto verify_walls = [&](milliseconds timeout) -> bool {
    const auto deadline = steady_clock::now() + timeout;
    while (steady_clock::now() < deadline) {
      const auto known = planning_scene_interface_.getKnownObjectNames();
      int found = 0;
      for (const auto & n : known) {
        if (n == "basket_wall_xpos" || n == "basket_wall_xneg" ||
            n == "basket_wall_ypos" || n == "basket_wall_yneg") {
          ++found;
        }
      }
      if (found == 4) return true;
      std::this_thread::sleep_for(milliseconds(50));
    }
    return false;
  };

  std::this_thread::sleep_for(milliseconds(200));
  if (verify_walls(milliseconds(3000))) {
    RCLCPP_INFO(node_->get_logger(),
      "addBasketCollision: all 4 basket walls present in planning scene");
    return true;
  }

  // First attempt timed out. Re-send the walls (the messages may
  // have been dropped or the planning_scene_monitor was busy) and
  // give it more time.
  RCLCPP_WARN(node_->get_logger(),
    "addBasketCollision: walls not in scene after 3 s; resending");
  add_wall("basket_wall_xpos", goal.x + inset, goal.y, wall_t, wall_l);
  add_wall("basket_wall_xneg", goal.x - inset, goal.y, wall_t, wall_l);
  add_wall("basket_wall_ypos", goal.x, goal.y + inset, wall_l, wall_t);
  add_wall("basket_wall_yneg", goal.x, goal.y - inset, wall_l, wall_t);
  std::this_thread::sleep_for(milliseconds(200));
  if (verify_walls(milliseconds(3000))) {
    RCLCPP_INFO(node_->get_logger(),
      "addBasketCollision: walls present after resend retry");
    return true;
  }

  RCLCPP_ERROR(node_->get_logger(),
    "addBasketCollision: walls STILL missing after resend; "
    "returning failure so caller can abort.");
  return false;
}


void cw2::removeBasketCollision()
{
  planning_scene_interface_.removeCollisionObjects({
    "basket_wall_xpos", "basket_wall_xneg",
    "basket_wall_ypos", "basket_wall_yneg"
  });
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
