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
  joint_states_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::SensorDataQoS(),
    std::bind(&cw2::jointStatesCallback, this, std::placeholders::_1));

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
  double & out_size,
  double & out_cx,
  double & out_cy)
{
  // MULTI-FRAME ACCUMULATION: take N consecutive cloud snapshots from
  // the static observation pose and accumulate ROI-filtered points
  // from all of them into one combined xy_pts. Per-frame point
  // jitter and asymmetric sampling biases largely average out, and
  // the M4 yaw estimate computed over the union has lower variance
  // than any single-frame estimate.
  using namespace std::chrono;
  constexpr int kNumFrames = 5;
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
    const auto deadline = steady_clock::now() + milliseconds(1000);
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

  // M4 PIVOT: the spawner-reported obj_xy is the true geometric
  // centre of the shape (the SDF model origin coincides with the
  // shape centre after the link transform). Computing the 4th-order
  // moment around obj_xy instead of the biased cluster mean removes
  // the centroid-bias coupling from the yaw estimate and makes the
  // recovered yaw an unbiased measurement of orientation alone.
  // Earlier behaviour: 5 deg of yaw bias from cluster mean offset
  // produced ~5 mm of finger-position error at the wall - enough to
  // straddle one finger past the wall edge. With obj as pivot, only
  // the per-frame measurement noise remains.
  const Eigen::Vector2d pivot(obj_xy.x, obj_xy.y);

  // 4th-order complex moment. The nought's corners flip its sign, so
  // cancel that phase before dividing by 4.
  std::complex<double> M4(0.0, 0.0);
  for (const auto & q : xy_pts) {
    const std::complex<double> z(q.x() - pivot.x(), q.y() - pivot.y());
    const std::complex<double> z2 = z * z;
    M4 += z2 * z2;
  }
  const double phi_shape = (shape_type == "nought") ? M_PI : 0.0;
  double yaw = 0.25 * std::arg(M4 * std::polar(1.0, -phi_shape));
  while (yaw >  M_PI / 4.0) yaw -= M_PI / 2.0;
  while (yaw < -M_PI / 4.0) yaw += M_PI / 2.0;
  out_yaw = yaw;

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
  // modelling the basket.
  addBasketCollision(goal);

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
  if (!detectShapePose(obj, shape_type, yaw, size_s, cx, cy)) {
    RCLCPP_WARN(node_->get_logger(),
      "perception failed, falling back to yaw=0, size=40mm and obj xy");
    yaw = 0.0;
    size_s = 0.040;
    cx = obj.x;
    cy = obj.y;
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
  if (!moveArmCartesian(
      {makeTopDownPose(grasp_x, grasp_y, grasp_ee_z, link8_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) return false;

  if (!closeGripper(close_w)) return false;

  // POST-GRASP VERIFY: if the fingers reached the commanded close
  // width (or below), they're closed on air - the shape was either
  // missed entirely or knocked away by the descent. We MUST abort
  // before lift+transit+place; otherwise the rest of the pipeline
  // proceeds as if the shape were gripped (the user's exact symptom:
  // 'gripper slammed the nought, made it bounce, then proceeded to
  // the basket as if it had it'). When the shape IS gripped, the
  // fingers stall on its wall/arm thickness s, so total finger width
  // ~= s; commanded close_w = max(0.012, s-0.004), so a successful
  // grip leaves total width >= close_w + 0.004 in the steady state.
  // We use a 2 mm tolerance against measurement noise.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
    const double threshold = close_w + 0.002;
    RCLCPP_INFO(node_->get_logger(),
      "grasp verify: seen=%d f1=%.4f f2=%.4f width=%.4f close_w=%.4f thr=%.4f",
      seen, f1, f2, total_width, close_w, threshold);
    if (seen && total_width < threshold) {
      RCLCPP_ERROR(node_->get_logger(),
        "grasp verify FAILED: fingers fully closed (width=%.4f < %.4f) - "
        "no shape gripped, aborting before lift",
        total_width, threshold);
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

  // Shape is now in the basket - detach it from the gripper so the
  // retreat (and any subsequent task) doesn't see a phantom shape
  // attached to the hand.
  detachShape();

  // Warn-only: if the retreat fails the shape is already dropped.
  // Use transit_yaw because the descent was at that yaw; reverting
  // to link8_yaw here would force an unnecessary yaw rotation right
  // after release and could re-trigger the same goal-IK failure
  // we just avoided.
  if (!moveArmCartesian(
      {makeTopDownPose(place_x, place_y, safe_z, transit_yaw)},
      0.005, 0.0, /*allow_fallback=*/false)) {
    RCLCPP_WARN(node_->get_logger(), "retreat failed after place");
  }

  removeBasketCollision();
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


void cw2::addBasketCollision(const geometry_msgs::msg::Point & goal)
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

  // Yield + verify. Earlier evidence: the prior code used
  // applyCollisionObjects(vector) and 100 ms wasn't enough - the
  // walls weren't actually in the scene by the time the next plan()
  // request ran (warning at task end:
  //   "Tried to remove world object 'basket_wall_xpos', but it does
  //    not exist in this scene")
  // So we wait longer and then poll getKnownObjectNames() until the
  // walls show up (or 1 s timeout).
  using namespace std::chrono;
  std::this_thread::sleep_for(milliseconds(200));
  const auto deadline = steady_clock::now() + milliseconds(1000);
  while (steady_clock::now() < deadline) {
    const auto known = planning_scene_interface_.getKnownObjectNames();
    int found = 0;
    for (const auto & n : known) {
      if (n == "basket_wall_xpos" || n == "basket_wall_xneg" ||
          n == "basket_wall_ypos" || n == "basket_wall_yneg") {
        ++found;
      }
    }
    if (found == 4) {
      RCLCPP_INFO(node_->get_logger(),
        "addBasketCollision: all 4 basket walls present in planning scene");
      return;
    }
    std::this_thread::sleep_for(milliseconds(50));
  }
  RCLCPP_WARN(node_->get_logger(),
    "addBasketCollision: basket walls did not appear within 1 s; "
    "subsequent plans may not avoid the basket");
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
