#!/usr/bin/python3

"""
This code contains the world spawner. This file accomplishes two main goals:

  1. Create and manage the objects in the gazebo world using the World() class
  2. Spawn and monitor the completion of coursework tasks using the Task() class

The coursework contains three tasks, and each of them are defined here. There
are three classes derived from the Task() base class, Task1(), Task2(), and
Task3().
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from cw2_world_spawner.srv import TaskSetup, Task1Service, Task2Service, Task3Service
from cw2_world_spawner_lib.coursework_world_spawner import (
  Model,
  WorldSpawner,
  call_service_sync,
  random_position_in_area,
  random_orientation,
)

# ----- key coursework task parameters ----- #

# task 1 parameters
T1_SHAPE_X_LIMS = [0.40, 0.55]           # xrange a shape can spawn
T1_SHAPE_Y_LIMS = [-0.40, 0.40]          # yrange a shape can spawn
T1_ANY_ORIENTATION = False               # do we allow any rotation of a shape
T1_GROUND_PLANE_NOISE = 0e-3             # do we add noise on the z height of the green tiles
T1_USE_MULTIPLE_SIZES = False            # do we spawn objects with varying sizes

# task 2 parameters
T2_SHAPE_X_LIMS = [0.40, 0.55]           # xrange a shape can spawn
T2_SHAPE_Y_LIMS = [-0.40, 0.40]          # yrange a shape can spawn
T2_N_REF_SHAPES = 2                      # number of baskets to spawn
T2_OBJECT_REF_POINTS = [(-0.43, -0.4),
                        (-0.43,  0.4)]
T2_ANY_ORIENTATION = False               # do we allow any rotation of a shape
T2_GROUND_PLANE_NOISE = 0e-3             # do we add noise on the z height of the green tiles
T2_USE_MULTIPLE_SIZES = False            # do we spawn objects with varying sizes

# task 3 parameters
T3_MAX_SHAPES = 7                        # maximum number of spawned shapes
T3_SHAPE_X_LIMS = [-0.6, 0.7]            # xrange a shape can spawn
T3_SHAPE_Y_LIMS = [-0.55, 0.55]          # yrange a shape can spawn
T3_N_OBSTACLES = 2
T3_ANY_ORIENTATION = False               # do we allow any rotation of a shape
T3_GROUND_PLANE_NOISE = 0e-3             # do we add noise on the z height of the green tiles
T3_USE_MULTIPLE_SIZES = False            # do we spawn objects with varying sizes

# possible goal basket locations (x, y)
BASKET_LOCATIONS = [(-0.41, -0.36),
                    (-0.41,  0.36)]

# define the variety of spawned objects, these values are defined in .sdf files
POSSIBLE_SHAPES = ["nought", "cross"]
POSSIBLE_SIZES = ["40", "30", "20"]
POSSIBLE_COLOURS = {'purple': [0.8, 0.1, 0.8],
                    'red':    [0.8, 0.1, 0.1],
                    'blue':   [0.1, 0.1, 0.8]}
POSSIBLE_OBSTACLES = [
  ["obstacle_1", 80e-3],
  ["obstacle_2", 50e-3],
  ["obstacle_3", 60e-3],
  ["obstacle_4", 100e-3],
]

GLOBAL_NODE = None
GLOBAL_WORLD_SPAWNER = None
GLOBAL_WORLD = None


def set_context(node, world_spawner, world):
  global GLOBAL_NODE, GLOBAL_WORLD_SPAWNER, GLOBAL_WORLD
  GLOBAL_NODE = node
  GLOBAL_WORLD_SPAWNER = world_spawner
  GLOBAL_WORLD = world


class World(object):

  # lengths in metres for world objects, resultant from sdf model files, don't change
  tile_side_length = 100e-3
  tile_thickness = 20e-3
  basket_side_length = 350e-3
  basket_height = 50e-3
  robot_safety_radius = 250e-3

  def __init__(self, node, world_spawner):
    self.node = node
    self.world_spawner = world_spawner
    self.world_spawner.despawn_all(exceptions="object-golf-tile")
    self.spawn_tiles()

  def spawn_tiles(self, tile_height=0):
    model = Model(model_name="all_tiles",
                  instance_name='object-all-golf-tiles',
                  model_type='sdf',
                  position=[0, 0, tile_height],
                  spawner=self.world_spawner)
    self.world_spawner.spawn(model)

  def reset(self):
    self.world_spawner.despawn_all()


class Task(object):

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    self.node = node or GLOBAL_NODE
    self.world_spawner = world_spawner or GLOBAL_WORLD_SPAWNER
    self.world = world or GLOBAL_WORLD
    if self.node is None or self.world_spawner is None or self.world is None:
      raise RuntimeError("Task context not initialized")

    self.models = []
    self.ground_level = self.world.tile_thickness

    if mode == 'coursework':
      self.spawn_task_objects()
      self.begin_task()
    else:
      self.spawn_test_objects(validation_scenario)
      self.begin_test(validation_scenario)

  def spawn_model(self, name, point=None, spawn_height=None,
                  xlims=None, ylims=None,
                  rotationlims=None):
    if xlims is None:
      xlims = [0.3, 0.5]
    if ylims is None:
      ylims = [-0.12, 0.12]
    if rotationlims is None:
      rotationlims = [0, 0]

    model_name = name
    instance_name = name + f"_object_{len(self.models) + 1}"

    if spawn_height is None:
      zlims = [self.ground_level + 5e-3, self.ground_level + 5e-3]
    else:
      zlims = [spawn_height, spawn_height]

    if point is None:
      position = random_position_in_area(xlims, ylims, zlims)
    else:
      position = [point[0], point[1], zlims[0]]

    model = Model(model_name=model_name,
                  instance_name=instance_name,
                  model_type='sdf',
                  position=position,
                  orientation=random_orientation(rotationlims, [0, 0], [0, 0]),
                  spawner=self.world_spawner)
    self.world_spawner.spawn(model)

    self.models.append(model)
    return model

  def prepare_for_task_request(self, srv_type_or_name, service_name=None, timeout=60.0):
    if service_name is None:
      service_name = srv_type_or_name
      srv_type = getattr(self, "service_type", None)
      if srv_type is None:
        raise RuntimeError("service_type not set for task request")
    else:
      srv_type = srv_type_or_name

    self.node.get_logger().debug(f"Attempting to connect to {service_name} Service...")
    client_node = self.world_spawner.client_node
    client = client_node.create_client(srv_type, service_name)
    if not client.wait_for_service(timeout_sec=timeout):
      self.node.get_logger().warn(f"{service_name} Request failed - not advertised")
      return None
    return client

  def reset_task(self, respawn_tiles=False, tile_height=0):
    self.world_spawner.despawn_all(keyword='object', exceptions="tile")
    if respawn_tiles:
      self.world_spawner.despawn_all(keyword="tile")
      self.world.spawn_tiles(tile_height=tile_height)
      self.ground_level = self.world.tile_thickness + tile_height
    self.models = []

  def get_position_from_point(self, pt):
    return np.asarray([pt.x, pt.y, pt.z])

  def get_position_from_point_stamped(self, ptst):
    pt = ptst.point
    return np.asarray([pt.x, pt.y, pt.z])

  def get_position_from_pose(self, pose):
    pos_np = self.get_position_from_point(pose.position)
    return pos_np

  def get_euclidean_distance(self, a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))

  def spawn_task_objects(self):
    raise NotImplementedError

  def begin_task(self):
    raise NotImplementedError

  def spawn_test_objects(self, validation_scenario):
    raise NotImplementedError

  def begin_test(self, validation_scenario):
    raise NotImplementedError


class Task1(Task):

  service_to_request = "/task1_start"
  service_type = Task1Service

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    self.node = node or GLOBAL_NODE
    self.node.get_logger().info('================Starting Task1==============')
    super().__init__(mode, validation_scenario, node=node, world_spawner=world_spawner, world=world)

  def spawn_task_objects(self):
    if T1_GROUND_PLANE_NOISE < 1e-3:
      self.reset_task()
    else:
      tile_height = np.random.random() * (T1_GROUND_PLANE_NOISE)
      self.reset_task(respawn_tiles=True, tile_height=tile_height)

    colour_options = list(POSSIBLE_COLOURS.keys())
    rand_colour = np.random.choice(colour_options)
    rand_object = np.random.choice(POSSIBLE_SHAPES)
    if T1_USE_MULTIPLE_SIZES:
      rand_size = np.random.choice(POSSIBLE_SIZES)
    else:
      rand_size = POSSIBLE_SIZES[0]
    random_object = rand_object + "_" + rand_colour + "_" + rand_size + "mm"
    rot_lims = [0, 2 * math.pi] if T1_ANY_ORIENTATION else [0, 0]
    self.spawn_model(name=random_object, rotationlims=rot_lims, xlims=T1_SHAPE_X_LIMS,
                     ylims=T1_SHAPE_Y_LIMS)

    self.object_type = rand_object

    random_goal = BASKET_LOCATIONS[np.random.randint(0, len(BASKET_LOCATIONS))]
    self.spawn_model(name="basket", point=random_goal)

  def send_task1_request(self, client, object_point, goal_point):
    self.node.get_logger().debug("Task1 Service connected. Sending request...")
    client_node = self.world_spawner.client_node
    request = Task1Service.Request()
    obj_pt = PointStamped()
    obj_pt.point = object_point
    obj_pt.header.frame_id = "panda_link0"
    obj_pt.header.stamp = self.node.get_clock().now().to_msg()
    goal_pt = PointStamped()
    goal_pt.point = goal_point
    goal_pt.header.frame_id = "panda_link0"
    goal_pt.header.stamp = self.node.get_clock().now().to_msg()
    request.object_point = obj_pt
    request.goal_point = goal_pt
    request.shape_type = self.object_type
    return call_service_sync(client_node, client, request, timeout_sec=600.0)

  def begin_task(self):
    client = self.prepare_for_task_request(Task1Service, self.service_to_request)
    time.sleep(1)

    object_point = self.models[0].get_model_state().pose.position
    goal_point = self.models[1].get_model_state().pose.position
    if client is not None:
      self.send_task1_request(client, object_point, goal_point)
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


class Task2(Task):

  service_to_request = "/task2_start"
  service_type = Task2Service

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    self.node = node or GLOBAL_NODE
    self.node.get_logger().info('================Starting Task2==============')
    super().__init__(mode, validation_scenario, node=node, world_spawner=world_spawner, world=world)

  def spawn_task_objects(self):
    if T2_GROUND_PLANE_NOISE < 1e-3:
      self.reset_task()
    else:
      tile_height = np.random.random() * (T2_GROUND_PLANE_NOISE)
      self.reset_task(respawn_tiles=True, tile_height=tile_height)

    colour_options = list(POSSIBLE_COLOURS.keys())
    shape_options = POSSIBLE_SHAPES
    n_ref_shapes = T2_N_REF_SHAPES
    rot_lims = [0, 2 * math.pi] if T2_ANY_ORIENTATION else [0, 0]

    rand_shapes = np.random.permutation(shape_options)

    if n_ref_shapes > len(shape_options):
      self.node.get_logger().warn("T2_N_REF_SHAPES is greater than number of possible shapes")
      n_ref_shapes = len(shape_options)
    if n_ref_shapes > len(T2_OBJECT_REF_POINTS):
      self.node.get_logger().warn("T2_N_REF_SHAPES is greater than number of reference positions")
      n_ref_shapes = len(T2_OBJECT_REF_POINTS)

    rand_shapes = rand_shapes[:n_ref_shapes]

    for i in range(n_ref_shapes):
      rand_colour = np.random.choice(colour_options)
      rand_object = rand_shapes[i]
      if T2_USE_MULTIPLE_SIZES:
        rand_size = np.random.choice(POSSIBLE_SIZES)
      else:
        rand_size = POSSIBLE_SIZES[0]
      random_ref = rand_object + "_" + rand_colour + "_" + rand_size + "mm"
      self.spawn_model(name=random_ref, point=T2_OBJECT_REF_POINTS[i], rotationlims=rot_lims)

    rand_colour = np.random.choice(colour_options)
    rand_object = np.random.choice(rand_shapes)
    if T2_USE_MULTIPLE_SIZES:
      rand_size = np.random.choice(POSSIBLE_SIZES)
    else:
      rand_size = POSSIBLE_SIZES[0]
    random_object = rand_object + "_" + rand_colour + "_" + rand_size + "mm"
    self.spawn_model(name=random_object, rotationlims=rot_lims, xlims=T2_SHAPE_X_LIMS,
                     ylims=T2_SHAPE_Y_LIMS)

  def send_task2_request(self, client, ref_points, mystery_point):
    self.node.get_logger().debug("Task2 Service connected. Sending request...")
    client_node = self.world_spawner.client_node
    request = Task2Service.Request()
    request.ref_object_points = ref_points
    request.mystery_object_point = mystery_point
    return call_service_sync(client_node, client, request, timeout_sec=600.0)

  def begin_task(self):
    client = self.prepare_for_task_request(Task2Service, self.service_to_request)

    ref_points = []
    for i in range(len(self.models) - 1):
      point_st = PointStamped()
      point_st.point = self.models[i].get_model_state().pose.position
      point_st.header.frame_id = "panda_link0"
      point_st.header.stamp = self.node.get_clock().now().to_msg()
      ref_points.append(point_st)

    mystery_point = PointStamped()
    mystery_point.point = self.models[-1].get_model_state().pose.position
    mystery_point.header.frame_id = "panda_link0"
    mystery_point.header.stamp = self.node.get_clock().now().to_msg()

    if client is not None:
      self.send_task2_request(client, ref_points, mystery_point)
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


class Task3(Task):

  service_to_request = "/task3_start"
  service_type = Task3Service

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    self.node = node or GLOBAL_NODE
    self.node.get_logger().info('================Starting Task3==============')
    super().__init__(mode, validation_scenario, node=node, world_spawner=world_spawner, world=world)

  def find_empty_point(self, mysize, xlims, ylims, point_size_pairs):
    grid_index = 0

    xlims = xlims[:]
    ylims = ylims[:]

    radius = mysize * (1.0 / math.sqrt(2))
    xlims[0] += radius
    xlims[1] -= radius
    ylims[0] += radius
    ylims[1] -= radius

    x = np.arange(xlims[0], xlims[1], 2e-3)
    y = np.arange(ylims[0], ylims[1], 2e-3)
    X_grid, Y_grid = np.meshgrid(x, y)
    points_grid = np.dstack((X_grid, Y_grid))
    points_line = np.reshape(points_grid, (-1, 2))
    randomised_points = np.random.permutation(points_line)

    while grid_index < len(randomised_points):
      good_point = True

      rand_x = randomised_points[grid_index][0]
      rand_y = randomised_points[grid_index][1]
      grid_index += 1

      for ((x, y), prev_size) in point_size_pairs:
        radius = (prev_size + mysize * 1.1) / math.sqrt(2)

        this_xlims = [x - radius, x + radius]
        this_ylims = [y - radius, y + radius]

        if not (rand_x > this_xlims[0] and rand_x < this_xlims[1] and
                rand_y > this_ylims[0] and rand_y < this_ylims[1]):
          good_point = True
        else:
          good_point = False
          break

      if good_point:
        return (rand_x, rand_y)

    return None

  def spawn_task_objects(self):
    if T3_GROUND_PLANE_NOISE < 1e-3:
      self.reset_task()
    else:
      tile_height = np.random.random() * (T3_GROUND_PLANE_NOISE)
      self.reset_task(respawn_tiles=True, tile_height=tile_height)

    colour_options = list(POSSIBLE_COLOURS.keys())
    rot_lims = [0, 2 * math.pi] if T3_ANY_ORIENTATION else [0, 0]
    n_objects = np.random.randint(T3_MAX_SHAPES // 2, T3_MAX_SHAPES + 1)

    random_goal = BASKET_LOCATIONS[np.random.randint(0, len(BASKET_LOCATIONS))]
    self.spawn_model(name="basket", point=random_goal)

    panda = [(0, 0), self.world.robot_safety_radius * math.sqrt(2)]
    basket = [
      (self.models[0].get_model_state().pose.position.x,
       self.models[0].get_model_state().pose.position.y),
      0.5 * self.world.basket_side_length * math.sqrt(2)
    ]
    self.spawned_points = [panda, basket]

    for i in range(T3_N_OBSTACLES):
      random_obstacle = POSSIBLE_OBSTACLES[np.random.randint(0, len(POSSIBLE_OBSTACLES))]
      obstacle_size = 200e-3
      obstacle_name = random_obstacle[0]
      obstacle_spawn_height = random_obstacle[1] + self.ground_level + 10e-3
      spawn_point = self.find_empty_point(obstacle_size, T3_SHAPE_X_LIMS,
                                          T3_SHAPE_Y_LIMS, self.spawned_points)
      if spawn_point is not None:
        self.spawn_model(name=obstacle_name, point=spawn_point, rotationlims=rot_lims,
                         spawn_height=obstacle_spawn_height)
        self.spawned_points.append([spawn_point, obstacle_size])
      else:
        break

    for i in range(n_objects):
      rand_colour = np.random.choice(colour_options)
      rand_object = np.random.choice(POSSIBLE_SHAPES)
      if T3_USE_MULTIPLE_SIZES:
        rand_size = np.random.choice(POSSIBLE_SIZES)
      else:
        rand_size = POSSIBLE_SIZES[0]
      random_object = rand_object + "_" + rand_colour + "_" + rand_size + "mm"
      random_size = int(rand_size) * 5 * 1e-3
      spawn_point = self.find_empty_point(random_size, T3_SHAPE_X_LIMS,
                                          T3_SHAPE_Y_LIMS, self.spawned_points)
      if spawn_point is not None:
        self.spawn_model(name=random_object, point=spawn_point, rotationlims=rot_lims)
        self.spawned_points.append([spawn_point, random_size])
      else:
        break

  def send_task3_request(self, client):
    self.node.get_logger().debug("Task3 Service connected. Sending request...")
    client_node = self.world_spawner.client_node
    request = Task3Service.Request()
    return call_service_sync(client_node, client, request, timeout_sec=600.0)

  def begin_task(self):
    client = self.prepare_for_task_request(Task3Service, self.service_to_request)
    if client is not None:
      self.send_task3_request(client)
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


def handle_task_request(request, response):
  if request.task_index == 1:
    Task1(mode="coursework")
  elif request.task_index == 2:
    Task2(mode="coursework")
  elif request.task_index == 3:
    Task3(mode="coursework")
  else:
    GLOBAL_NODE.get_logger().warn("Unrecognized task requested")
  return response


def main():
  rclpy.init()
  node = Node('coursework3_wrapper')

  world_spawner = WorldSpawner(node)
  world = World(node, world_spawner)
  set_context(node, world_spawner, world)

  node.create_service(TaskSetup, '/task', handle_task_request)
  node.get_logger().info("Ready to initiate task.")
  node.get_logger().info("Use ros2 service call /task cw2_world_spawner/srv/TaskSetup \"{task_index: <INDEX>}\" to start a task")

  from ament_index_python.packages import get_package_share_directory
  pkg_models_path = get_package_share_directory("cw2_world_spawner") + "/models"
  export_cmd = f"export GAZEBO_MODEL_PATH={pkg_models_path}"
  node.get_logger().warn(
    "If Gazebo freezes when you call a task you need to run in your terminal the following: "
    + export_cmd
  )

  rclpy.spin(node)
  if hasattr(world_spawner, "destroy"):
    world_spawner.destroy()
  node.destroy_node()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
