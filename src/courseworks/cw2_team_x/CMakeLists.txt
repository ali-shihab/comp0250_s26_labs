cmake_minimum_required(VERSION 3.8)
project(cw2_team_x)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(cw2_world_spawner REQUIRED)

set(srv_files
  "srv/Example.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_cw2_class_lib
  src/cw2_class.cpp
)

ament_target_dependencies(${PROJECT_NAME}_cw2_class_lib
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  octomap
  octomap_msgs
  shape_msgs
  cw2_world_spawner
)

target_link_libraries(${PROJECT_NAME}_cw2_class_lib
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(cw2_solution_node src/cw2_node.cpp)

ament_target_dependencies(cw2_solution_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  octomap
  octomap_msgs
  shape_msgs
  cw2_world_spawner
)

target_link_libraries(cw2_solution_node
  ${PROJECT_NAME}_cw2_class_lib
)

rosidl_get_typesupport_target(cw2_team_x_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_cw2_class_lib
  ${cw2_team_x_typesupport_target})
target_link_libraries(cw2_solution_node
  ${cw2_team_x_typesupport_target})

install(TARGETS
  ${PROJECT_NAME}_cw2_class_lib
  cw2_solution_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.launch" EXCLUDE
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
