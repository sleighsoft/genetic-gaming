import pygame
import sys
import random
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import os
import math
import argparse
import msgpackrpc
import time
import copy
import pprint
from pygame.locals import *
from PIL import Image
import numpy as np


class MapGenerator(object):
  def __init__(self, min_width, max_width, max_angle, min_length, max_length, game_height, game_width, start_point=None, start_angle=45):
    self._min_width = min_width
    self._max_width = max_width
    self._max_angle = max_angle
    self._min_length = min_length
    self._max_length = max_length
    self._game_height = game_height
    self._game_width = game_width
    self._start_point = start_point
    self._start_angle = start_angle

  def get_next_endings(self, left_start, right_start, last_angle):
    center = Vec2d((left_start.x + right_start.x)/2, (left_start.y+right_start.y)/2)
    length = random.uniform(self._min_length, self._max_length)
    angle = random.uniform(last_angle-self._max_angle, last_angle+self._max_angle)
    width = random.uniform(self._min_width, self._max_width)
    target_center = Vec2d.unit()
    target_center.angle = angle
    target_center.length = length
    target_center = target_center + center

    left_end = copy.copy(target_center)
    left_end.rotate_degrees(-90)
    left_end.length = width / 2

    right_end = copy.copy(target_center)
    right_end.rotate_degrees(90)
    right_end.length = width / 2

    left_end = target_center + left_end
    right_end = target_center + right_end
    return left_end, right_end, target_center.angle_degrees, target_center

  def is_valid(self, point):
    return 0 < point.x < self._game_height and 0 < point.y < self._game_width

  def get_start_points(self):
    if self._start_point is None:
      self._start_point = Vec2d(30, 30)

    width = random.uniform(self._min_width, self._max_width)

    left_end = Vec2d.unit()
    left_end.angle_degrees= self._start_angle-90
    left_end.length = width / 2

    right_end = Vec2d.unit()
    right_end.angle_degrees = self._start_angle+90
    right_end.length = width / 2

    return self._start_point+left_end, self._start_point+right_end

  def get_wall(self, start_point, end_point):
    return {
      'start': start_point,
      'end': end_point
    }

  def generate(self):
    last_left, last_right = self.get_start_points()
    last_angle = self._start_angle
    tries_left = 5

    found = []
    centers = [self._start_point]
    while tries_left > 0:
      next_left, next_right, angle, center = self.get_next_endings(last_left, last_right, last_angle)

      if self.is_valid(next_left) and self.is_valid(next_right):
        found.append(self.get_wall(last_left, next_left))
        found.append(self.get_wall(last_right, next_right))
        centers.append(center)
        tries_left = 5
        last_left = next_left
        last_right = next_right
        last_angle = angle
      else:
        tries_left -= 1

    return found, centers


class Car(object):

  def __init__(self, shape, position, rotation, rotation_speed, base_velocity,
               acceleration, deceleration, acceleration_time,
               max_velocity, color, sensor_range, num_sensors,
               sensor_color=(0, 0, 0)):
    # Static
    self._shape = shape
    self._position = position
    self._rotation = rotation
    self._rotation_speed = rotation_speed
    self._base_velocity = base_velocity
    self._acceleration = acceleration
    self._deceleration = deceleration
    self._acceleration_time = acceleration_time
    self._max_velocity = max_velocity
    self._color = color
    self._sensor_range = sensor_range
    self._num_sensors = num_sensors
    self._sensor_color = sensor_color

    inertia = pymunk.moment_for_box(1, self._shape)
    self.car_body = pymunk.Body(1, inertia)
    self.car_shape = pymunk.Poly.create_box(self.car_body, self._shape)

    # Dynamic
    self.reset()

  def add_to_space(self, space):
    space.add(self.car_body, self.car_shape)

  def get_sensors(self):
    sensors = []
    # for i in range(self.num_sensors):
    start = s_x, s_y = self.car_body.position
    e_x, e_y = start + (self._sensor_range, 0)
    rotation = self.car_body.angle
    rotated_end = Car.get_rotated_point(s_x, s_y, e_x, e_y, rotation)

    sensors.append((start, rotated_end))

    return sensors

  def show_sensors(self, screen, points_of_impact):
    for i, sensor in enumerate(self.get_sensors()):
      end = sensor[1] if points_of_impact[i] is None else points_of_impact[i]
      pygame.draw.line(screen, self._sensor_color, sensor[0], end)

  def get_sensor_distances(self, walls, screen=None):
    distances = []
    points_of_impact = []
    sensors = self.get_sensors()
    for sensor in sensors:
      # Determine points of impact of sensor rays
      for wall in walls:
        query = wall.segment_query(sensor[0], sensor[1])
        if query.shape is not None:
          point_of_impact = query.point
          points_of_impact.append(point_of_impact)
        else:
          points_of_impact.append(None)

      # Calculate distance until sensor collides with an object
      start = sensor[0]
      end = sensor[1]
      if points_of_impact[-1] is not None:
        end = points_of_impact[-1]
      distances.append(start.get_distance(end))

    if screen:
      self.show_sensors(screen, points_of_impact)

    return distances

  def trigger_rotate_left(self):
    self.rotation -= self._rotation_speed

  def trigger_rotate_right(self):
    self.rotation += self._rotation_speed

  def trigger_acceleration(self):
    if self.current_acceleration_time == 0:
      self.velocity = self._base_velocity
    else:
      self.velocity = min(self._max_velocity,
                          self.velocity * self._acceleration)
    self.current_acceleration_time = self._acceleration_time

  def move(self):
    """Perform all triggered movements."""
    if self.current_acceleration_time > 0:
      self.current_acceleration_time -= 1
    else:
      self.velocity = max(0, self.velocity * self._deceleration)
    driving_direction = Vec2d(1, 0).rotated(self.rotation)
    self.car_body.angle = self.rotation
    self.car_body.velocity = self.velocity * driving_direction

  def reset(self):
    """Reset car to initial settings."""
    # Pymunk
    self.car_body.position = self._position
    self.car_shape.color = self._color
    self.car_shape.elasticity = 1.0
    self.car_body.angle = self._rotation
    self.car_body.velocity = Vec2d(0, 0)
    driving_direction = Vec2d(0, 0).rotated(self.car_body.angle)
    self.car_body.apply_impulse_at_world_point(driving_direction)

    # Dynamic
    self.velocity = 0
    self.rotation = self._rotation
    self.current_acceleration_time = 0
    self.is_dead = False

  @staticmethod
  def get_rotated_point(x_1, y_1, x_2, y_2, radians):
    # Rotate x_2, y_2 around x_1, y_1 by angle.
    x_change = (x_2 - x_1) * math.cos(radians) + \
        (y_2 - y_1) * math.sin(radians)
    y_change = (y_1 - y_2) * math.cos(radians) - \
        (x_1 - x_2) * math.sin(radians)
    new_x = x_change + x_1
    new_y = y_change + y_1
    return int(new_x), int(new_y)


class Game(object):
  def __init__(self, args, simulator=None):
    # EvolutionServer
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_CARS = args['num_networks']
    self.SEND_PIXELS = args['send_pixels']
    self.SIMULATOR = simulator

    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

    # Game
    self.STEPPING = args['stepping']
    self.SCREEN_RESIZE_SHAPE = None
    if 'screen_resize_shape' in args:
      self.SCREEN_RESIZE_SHAPE = args['screen_resize_shape']
    if 'assets' not in args:
      args['assets'] = 'assets'
    self.SCREEN_WIDTH = 640
    self.SCREEN_HEIGHT = 480
    self.GAP = 130
    self.WALLX = 400
    self.BIRD_X = 70
    self.BIRD_HEIGHT = 50
    self.BIRD_WIDTH = 50
    self.GRAVITY = 5
    self.GRAVITY_ACCELERATION = 0.2
    self.JUMP_TIME = 17
    self.JUMP_SPEED = 10
    self.JUMP_SPEED_DECLINE = 1

    # Pygame
    self.screen = pygame.display.set_mode(
        (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    asset_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(asset_path, args['assets'])

    # Pymunk
    pymunk.pygame_util.positive_y_is_up = False
    self.space = pymunk.Space()
    self.space.gravity = pymunk.Vec2d(0., 0.)
    self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    RANDOM_RANGE = 40
    start_position = (100, np.random.randint(-RANDOM_RANGE, RANDOM_RANGE) + self.SCREEN_HEIGHT // 2)
    self.init_cars(start_position)
    self.init_walls(start_position)

    # Dynamic
    self.round = 0
    # If -stepping
    self.step = 0

  def init_cars(self, start_position):
    self.cars = []
    for _ in range(self.NUM_CARS):
      car = Car(shape=(15, 10),
                position=start_position,
                rotation=0.0,
                rotation_speed=0.05,
                base_velocity=5.0,
                acceleration=1.1,
                deceleration=0.8,
                acceleration_time=20,
                max_velocity=100,
                color=(0, 0, 0),
                sensor_range=100,
                num_sensors=2)
      self.cars.append(car)
      car.add_to_space(self.space)

  def init_walls(self, start_position):
    self.walls = []

    gen = MapGenerator(
      10, 20, 0.5, 20, 100, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, start_position, 0
    )
    wall_layout, centers = gen.generate()

    def get_wall(start, end):
      body = pymunk.Body(body_type=pymunk.Body.STATIC)
      segment = pymunk.Segment(body, start, end, 0)
      return segment

    for layout in wall_layout:
      wall = get_wall(**layout)
      self.walls.append(wall)

    self.space.add(self.walls)

  def reset(self):
    """Reset game state."""
    for car in self.cars:
      car.reset()

  def update_track(self):
    # TODO This may be required if our track is larger than the actual screen
    # and we would have to move the camera.
    pass

  def update_cars(self):
    # TODO Request car action from simulator see Flappybirds for an example
    for car in self.cars:
      car.move()
      self.check_for_collision(car)

  def check_for_collision(self, car):
    distances = car.get_sensor_distances(self.walls, self.screen)
    for distance in distances:
        # TODO Find suitable collision threshold
        # If you run into a border "distance" will only be zero when the car is
        # already stuck in the center of the wall
        if distance < 10.0:
          print('{} dead at {}'.format(car, distance))
          car.is_dead = True

  def run(self):
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 50)

    while True:
      clock.tick(60)

      # TODO Allow either manual single player or genetic algorithm
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_RIGHT:
            self.cars[0].trigger_rotate_right()
          if event.key == pygame.K_LEFT:
            self.cars[0].trigger_rotate_left()
          if event.key == pygame.K_UP:
            self.cars[0].trigger_acceleration()

      if sum(pygame.key.get_pressed()):
        pressed_key = pygame.key.get_pressed()
        if pressed_key[pygame.K_RIGHT]:
          self.cars[0].trigger_rotate_right()
        if pressed_key[pygame.K_LEFT]:
          self.cars[0].trigger_rotate_left()
        if pressed_key[pygame.K_UP]:
          self.cars[0].trigger_acceleration()

      self.screen.fill((255, 255, 255))

      # Track
      self.update_track()

      # Cars
      self.update_cars()

      # Reset check
      if all([car.is_dead for car in self.cars]):
        self.reset()

      # Pymunk & Pygame
      self.space.debug_draw(self.draw_options)
      pygame.display.update()
      fps = 60
      dt = 1. / fps
      self.space.step(dt)

  def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
    # Rotate x_2, y_2 around x_1, y_1 by angle.
    x_change = (x_2 - x_1) * math.cos(radians) + \
        (y_2 - y_1) * math.sin(radians)
    y_change = (y_1 - y_2) * math.cos(radians) - \
        (x_1 - x_2) * math.sin(radians)
    new_x = x_change + x_1
    new_y = y_change + y_1
    return int(new_x), int(new_y)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-num_networks',
      help='Number of birds to spawn and number of networks to use for them.',
      type=int,
      default=None,
      required=True
  )
  parser.add_argument(
      '--timeout',
      help='Initial sleep time to allow the ML agent to start.',
      type=int,
      default=None
  )
  parser.add_argument(
      '-host',
      help='Machine learning agent host.',
      type=str,
      default='localhost'
  )
  parser.add_argument(
      '-port',
      help='Machine learning agent port.',
      type=int,
      default=4004
  )
  parser.add_argument(
      '-assets',
      help='Asset directory name.',
      type=str,
      default='assets'
  )
  parser.add_argument(
      '-send_pixels',
      help='If set, send the whole screen as pixels instead of special '
      'features.',
      action='store_true'
  )
  parser.add_argument(
      '-stepping',
      help='If set, run one bird after the other until all birds died once. '
      'Then evolve.',
      action='store_true'
  )
  args = parser.parse_args()
  if args.timeout:
    time.sleep(args.timeout)
  args = args.__dict__
  Game(args).run()
