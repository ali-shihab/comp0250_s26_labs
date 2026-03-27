Perception Pipeline Tutorial (Humble, Rolling-Style Adaptation)
================================================================

Prerequisites (Install First)
-----------------------------
Install required Humble packages first:

.. code-block:: bash

   sudo apt-get update && sudo apt-get install -y \
     ros-humble-rosbag2-storage-mcap \
     ros-humble-moveit-resources-panda-moveit-config \
     ros-humble-moveit-resources-panda-description

Install and initialize Git LFS:

.. code-block:: bash

   sudo apt-get install -y git-lfs
   git lfs install

If you still need to clone the benchmark resource repository somewhere
(not needed on the CS servers, it is part of /cs/student/msc/rai/comp0250/ws_moveit2) :

.. code-block:: bash

   gh repo clone moveit/moveit_benchmark_resources

Before each procedure:

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash

Introduction
------------
MoveIt allows seamless integration of 3D sensors through
`Octomap <http://octomap.github.io/>`_ and the occupancy map updater pipeline.

This tutorial follows the ROS 2 ``main`` (rolling-style) structure while
adapting commands and defaults for ROS 2 Humble with the benchmark bag used in
this workspace.

The occupancy updater plugins used in MoveIt are:

* PointCloud occupancy map updater
  (input type: ``sensor_msgs/msg/PointCloud2``)
* DepthImage occupancy map updater
  (input type: ``sensor_msgs/msg/Image``)

Upstream references:

* Main tutorial page:
  https://moveit.picknik.ai/main/doc/examples/perception_pipeline/perception_pipeline_tutorial.html
* Main tutorial source:
  https://github.com/moveit/moveit2_tutorials/tree/main/doc/examples/perception_pipeline

Adaptation repository:

* https://github.com/Dkaka/ros2_humble_perception_pipeline

Getting Started
---------------
If you have not run through the MoveIt Getting Started tutorial yet, it is
still possible to run this perception demo directly. However, completing
Getting Started first is recommended for background context.

Workspace paths used in this adaptation:

* Demo package: ``comp0250_s26_labs/src/labs/perception_pipeline_humble_demo``

Connecting to the Storage Backend
---------------------------------
Benchmark bag source:

* https://github.com/moveit/moveit_benchmark_resources/tree/main/moveit_benchmark_resources/bag_files/depth_camera_bag

Verify the bag is readable:

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   ros2 bag info \
     ..path..to../moveit_benchmark_resources/moveit_benchmark_resources/bag_files/depth_camera_bag/depth_camera_datas.mcap

On the CS servers:

.. code-block:: bash

   ros2 bag info \
      /cs/student/msc/rai/comp0250/ws_moveit2/src/moveit_benchmark_resources/moveit_benchmark_resources/bag_files/depth_camera_bag/depth_camera_datas.mcap


How to Create 3D Pointcloud Data for Octomap Creation (Optional)
-----------------------------------------------------------------
You can use the benchmark bag above, or record your own bag.

Shell 1:

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   cd comp0250_s26_labs
   source install/setup.bash
   ros2 launch perception_pipeline_humble_demo depth_camera_environment.launch.py

Shell 2:

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   ros2 bag record \
     /camera_1/points /camera_1/depth/image_raw /camera_1/depth/camera_info \
     /camera_2/points /camera_2/depth/image_raw /camera_2/depth/camera_info \
     /tf /tf_static

The depth-camera environment includes two depth cameras and static transforms so
MoveIt can transform incoming sensor data into the planning frame.

To visualize camera point clouds during recording:

.. code-block:: bash

   rviz2 -d src/labs/perception_pipeline_humble_demo/rviz2/depth_camera_environment.rviz

Configuration
-------------
MoveIt uses an octree-based representation of the world. The key octomap
parameters are:

* ``octomap_frame``: frame where the occupancy map is maintained
* ``octomap_resolution``: voxel size in meters
* ``max_range``: maximum sensor range used for map updates

In this workspace, the benchmark bag provides pointcloud, depth image, and TF
topics. The default stable path on Humble uses pointcloud updaters.

YAML Configuration file (Point Cloud)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Default/recommended (retimed topic) config:
``ros2_pp_ws/src/perception_pipeline_humble_demo/config/sensors_3d.yaml``

Raw-topic fallback config:
``ros2_pp_ws/src/perception_pipeline_humble_demo/config/sensors_3d_raw.yaml``

``sensors_3d.yaml``:

.. code-block:: yaml

   sensors:
     - camera_1_pointcloud
     - camera_2_pointcloud
   camera_1_pointcloud:
       sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
       point_cloud_topic: /camera_1/points_retimed
       max_range: 5.0
       point_subsample: 1
       padding_offset: 0.1
       padding_scale: 1.0
       max_update_rate: 1.0
       filtered_cloud_topic: /camera_1/filtered_points
   camera_2_pointcloud:
       sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
       point_cloud_topic: /camera_2/points_retimed
       max_range: 5.0
       point_subsample: 1
       padding_offset: 0.1
       padding_scale: 1.0
       max_update_rate: 1.0
       filtered_cloud_topic: /camera_2/filtered_points

Configurations for Point Cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General parameters:

* ``sensor_plugin``: plugin implementation name
* ``max_update_rate``: upper bound on occupancy update frequency

Pointcloud-specific parameters:

* ``point_cloud_topic``: incoming cloud topic
* ``max_range``: ignore points farther than this distance
* ``point_subsample``: use one out of every N points
* ``padding_offset``: self-filtering padding in meters
* ``padding_scale``: self-filtering mesh scale factor
* ``filtered_cloud_topic``: debug output for self-filtered cloud

YAML Configuration file (Depth Map)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Rolling-style depth-image updater configuration is supported by MoveIt, but in
this Humble adaptation it is not the default path for the short benchmark bag.
The benchmark bag under looped playback can produce synchronization and
self-filter noise with depth-image mode.

Example depth-image YAML entry (for parity with rolling docs):

.. code-block:: yaml

   sensors:
     - camera_2_depth_image
   camera_2_depth_image:
       sensor_plugin: occupancy_map_monitor/DepthImageOctomapUpdater
       image_topic: /camera_2/depth/image_raw
       queue_size: 5
       near_clipping_plane_distance: 0.3
       far_clipping_plane_distance: 5.0
       shadow_threshold: 0.2
       padding_scale: 1.0
       max_update_rate: 1.0
       filtered_cloud_topic: /camera_2/filtered_points

Configurations for Depth Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General parameters:

* ``sensor_plugin``: plugin implementation name
* ``max_update_rate``: upper bound on occupancy update frequency

Depth-image-specific parameters:

* ``image_topic``: depth image topic
* ``queue_size``: queued frame count for processing
* ``near_clipping_plane_distance``: minimum accepted depth
* ``far_clipping_plane_distance``: maximum accepted depth
* ``shadow_threshold``: shadow sensitivity in depth processing
* ``padding_scale``: self-filtering mesh scale factor
* ``filtered_cloud_topic``: debug filtered cloud output

Update the Launch File
^^^^^^^^^^^^^^^^^^^^^^
Add the YAML file to the launch script
""""""""""""""""""""""""""""""""""""""
This repository already wires sensor YAML into the MoveIt config builder at:

* ``ros2_pp_ws/src/perception_pipeline_humble_demo/launch/perception_pipeline_demo.launch.py``

The launch selects sensor config by ``use_retimed_pointclouds``:

* ``true``  -> ``config/sensors_3d.yaml`` (default)
* ``false`` -> ``config/sensors_3d_raw.yaml``

Launch wiring snippet:

.. code-block:: python

   sensors_file = "sensors_3d.yaml" if use_retimed else "sensors_3d_raw.yaml"
   ...
   .sensors_3d(
       file_path=os.path.join(
           get_package_share_directory("perception_pipeline_humble_demo"),
           "config",
           sensors_file,
       )
   )

Octomap Configuration
"""""""""""""""""""""
The same launch file configures ``move_group`` and the planning scene monitor.
With the sensor YAML above, octomap updates are driven by:

* frame from incoming sensor messages (plus available TF to ``world``)
* per-updater ``max_range`` from YAML
* map resolution from MoveIt config defaults (resource package)

For runtime visibility in RViz:

* ``MotionPlanning -> Scene Geometry -> Show Scene Geometry`` toggles
  occupancy rendering.

Running Demo
------------
Build once:

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   cd /home/ziwen/ros2_humble_perception_pipeline/ros2_pp_ws
   colcon build --symlink-install

Shell 1 (MoveIt perception demo):

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   cd comp0250_s26_labs
   source install/setup.bash
   ros2 launch perception_pipeline_humble_demo perception_pipeline_demo.launch.py \
     use_retimed_pointclouds:=true

Shell 2 (play benchmark bag):

.. code-block:: bash

   conda deactivate || true
   source /opt/ros/humble/setup.bash
   source /home/ziwen/ros2_humble_perception_pipeline/ros2_pp_ws/install/setup.bash
   ros2 bag play -r 1 \
      /cs/student/msc/rai/comp0250/ws_moveit2/src/moveit_benchmark_resources/moveit_benchmark_resources/bag_files/depth_camera_bag/depth_camera_datas.mcap \
     --loop

Headless option:

.. code-block:: bash

   ros2 launch perception_pipeline_humble_demo perception_pipeline_demo.launch.py use_rviz:=false


RViz Display Controls
---------------------
In ``rviz2/perception_pipeline.rviz``:

* planning scene geometry (including octomap voxels):
  ``MotionPlanning -> Scene Geometry -> Show Scene Geometry``
* floor grid:
  ``Grid -> Enabled``
* camera point clouds:
  * ``Camera1Points -> Enabled`` (topic ``/camera_1/points``)
  * ``Camera2Points -> Enabled`` (topic ``/camera_2/points``)

Depth-image displays are not enabled by default in this profile.
To add them:

#. Click ``Add`` -> ``By display type`` -> ``rviz_default_plugins/Image``.
#. Set ``Topic`` to ``/camera_1/depth/image_raw`` (repeat for camera 2).
#. Toggle each image display with its ``Enabled`` checkbox.

Validation Results (2026-02-14)
-------------------------------
Automated validation runs were executed on Saturday, February 14, 2026:

* Benchmark bag:
  ``moveit_benchmark_resources/.../depth_camera_bag/depth_camera_datas.mcap``
* Duration per run: 3 minutes
* Playback mode: ``ros2 bag play -r 1 --loop``
* Launch mode: headless (``use_rviz:=false``)

Run A (recommended/default):

* Launch args: ``use_retimed_pointclouds:=true``
* Runtime checks:
  * ``/pointcloud_retimer`` node present
  * ``/camera_1/points_retimed`` receives data
* Warning/error counts in launch log:
  * ``Invalid frame ID``: ``0``
  * ``Missing transform for shape mesh``: ``0``
  * ``Mesh filter handle ... not found``: ``0``

Run B (raw fallback):

* Launch args: ``use_retimed_pointclouds:=false``
* Runtime checks:
  * no ``/pointcloud_retimer`` node
  * ``/camera_1/points`` receives data
  * ``/camera_1/points_retimed`` not published
* Warning/error counts in launch log:
  * ``Invalid frame ID``: ``0``
  * ``Missing transform for shape mesh``: ``2980``
  * ``Mesh filter handle ... not found``: ``0``

These results confirm that retimed mode is the stable default for the short
benchmark bag under looped playback.
