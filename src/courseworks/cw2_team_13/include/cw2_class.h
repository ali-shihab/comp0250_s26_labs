/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#ifndef CW2_CLASS_H_
#define CW2_CLASS_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "cw2_world_spawner/srv/task1_service.hpp"
#include "cw2_world_spawner/srv/task2_service.hpp"
#include "cw2_world_spawner/srv/task3_service.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

class cw2
{
public:
  explicit cw2(const rclcpp::Node::SharedPtr &node);

  // Service callbacks.
  void t1_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response);

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  // Task 1.
  bool t1_pickAndPlace(const geometry_msgs::msg::Point & obj,
                       const geometry_msgs::msg::Point & goal,
                       const std::string & shape_type);
  bool openGripper();
  bool closeGripper(double width);
  bool moveArmToPose(const geometry_msgs::msg::Pose & target_pose,
                     const std::string & description = "");
  bool moveArmCartesian(const std::vector<geometry_msgs::msg::Pose> & waypoints,
                        double eef_step = 0.005,
                        double jump_threshold = 0.0,
                        bool allow_fallback = true);
  // Same as moveArmCartesian but enforces path_constraints at every IK
  // checkpoint. If the constrained Cartesian compute fails (fraction <
  // 0.9 or compute error), and retry_unconstrained is true, falls back
  // to an unconstrained Cartesian path so the descent still happens.
  // Returns false only if BOTH attempts fail.
  bool moveArmCartesianConstrained(
      const std::vector<geometry_msgs::msg::Pose> & waypoints,
      const moveit_msgs::msg::Constraints & path_constraints,
      double eef_step = 0.005,
      double jump_threshold = 0.0,
      bool retry_unconstrained = true);
  geometry_msgs::msg::Pose makeTopDownPose(double x, double y, double z,
                                           double yaw = 0.0);
  void addGroundCollision();

  // Estimates the shape's yaw (folded into [-pi/4, pi/4] since both shapes
  // are C4-symmetric) and arm-width (snapped to {0.020, 0.030, 0.040} m)
  // from the latest point cloud, filtering around obj_xy.
  // For cross: out_yaw is the M4 estimate (4th-order complex moment).
  // out_alt_yaw is the MABR (minimum-area bounding rectangle) estimate
  // when the two disagree by more than ~0.4 rad - the caller then has
  // both candidates and can disambiguate via a second observation.
  // For nought: out_alt_yaw is set to the same as out_yaw (no
  // ambiguity to resolve).
  bool detectShapePose(const geometry_msgs::msg::Point & obj_xy,
                       const std::string & shape_type,
                       double & out_yaw,
                       double & out_size,
                       double & out_cx,
                       double & out_cy,
                       double * out_alt_yaw = nullptr);

  // Adds a tall conservative collision box at the spawner-reported obj
  // so MoveIt's joint-space planner routes around the shape during
  // long approach moves. removeShapeCollision() takes it back out
  // before the final descent so the gripper can actually reach it.
  void addShapeCollision(const geometry_msgs::msg::Point & obj,
                         const std::string & shape_type);
  void removeShapeCollision();

  // Attaches a worst-case 5*size_s x 5*size_s x 40 mm box to the
  // panda_hand link at the SHAPE CENTER (= hand frame
  // (ox_local, -oy_local, EE_TO_FINGER), derived from hand_X = -shape_X
  // and hand_Y = +shape_Y at grasp time), so the planning scene
  // matches the physical shape position rather than placing the box
  // at the fingertip plane.
  void attachShape(double size_s, double ox_local, double oy_local);
  // Detaches and removes the held_shape attached collision object.
  void detachShape();

  // Adds/removes a thin tile slab (z = [0.018, 0.020]) at the centre
  // of the workspace. Used during the held-shape transit so the
  // planner refuses paths that drag the held_shape into the tile.
  // Off during grasp/place descents so it doesn't impose padding-
  // induced finger collisions.
  void addTileCollision();
  void removeTileCollision();

  // Models the basket as four thin wall collision objects + a base
  // slab so the planner refuses paths that sweep the arm through the
  // basket during the observation transit. Removed at task end so
  // it doesn't pollute subsequent tasks (the spawner re-randomises
  // the basket location every task).
  // Returns false if the planning scene does not register all 4 walls
  // within the (extended) timeout, even after a resend retry. Caller
  // should abort the mission rather than risk knocking the basket.
  bool addBasketCollision(const geometry_msgs::msg::Point & goal);
  void removeBasketCollision();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw2_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task3Service>::SharedPtr t3_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr color_cloud_sub_;
  rclcpp::CallbackGroup::SharedPtr pointcloud_callback_group_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex cloud_mutex_;
  PointCPtr g_cloud_ptr;
  std::uint64_t g_cloud_sequence_ = 0;
  std::string g_input_pc_frame_id_;

  std::string pointcloud_topic_;
  bool pointcloud_qos_reliable_ = false;

  // /joint_states subscription, used by the post-grasp finger-width
  // verify in t1_pickAndPlace. We keep our own copy of the latest
  // finger positions because reading them through MoveIt right after
  // closeGripper is unreliable in this Gazebo+MoveIt setup.
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    joint_states_sub_;
  rclcpp::CallbackGroup::SharedPtr joint_states_callback_group_;

  // Action client for the panda_hand_controller. We use the
  // FollowJointTrajectory action interface (not the raw trajectory
  // topic) because the controller silently dropped some topic-published
  // trajectories in this setup - empirical evidence: fingers never moved
  // and the post-grasp verify caught width=0.0700 (= open) repeatedly.
  // The action interface acknowledges the goal and reports execution
  // status, so we know whether the controller accepted and ran the
  // trajectory.
  using FjtAction = control_msgs::action::FollowJointTrajectory;
  using FjtGoalHandle = rclcpp_action::ClientGoalHandle<FjtAction>;
  rclcpp_action::Client<FjtAction>::SharedPtr hand_action_client_;
  rclcpp::CallbackGroup::SharedPtr hand_action_callback_group_;

  // Drive both fingers to per_finger_target. Returns true on publish
  // success; the post-grasp verify in t1_pickAndPlace decides if the
  // grasp actually worked.
  bool commandGripper(double per_finger_target, double duration_s);
  std::mutex joint_states_mutex_;
  double finger1_pos_ = 0.04;
  double finger2_pos_ = 0.04;
  bool   finger_state_seen_ = false;

  void jointStatesCallback(
    const sensor_msgs::msg::JointState::ConstSharedPtr msg);
};

#endif  // CW2_CLASS_H_
