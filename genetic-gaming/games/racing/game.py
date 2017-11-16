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
import pprint
from pygame.locals import *
from PIL import Image
import numpy as np


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
    start = s_x, s_y = self.car_body.position

    # TODO Automatically create sensors based on self._num_sensors
    # Sensors should have same distance
    direction_offset = self._sensor_range / math.sqrt(2)
    sensor_directions = [start + (0, self._sensor_range),               # Left
                         # Half Left
                         start + (direction_offset, direction_offset),
                         start + (self._sensor_range, 0),               # Ahead
                         start + (direction_offset, - \
                                  direction_offset),  # Half Right
                         start + (0, -self._sensor_range)]              # Right

    for sensor_direction in sensor_directions:
      rotation = self.car_body.angle
      rotated_end = Car.get_rotated_point(
          s_x, s_y, sensor_direction[0], sensor_direction[1], rotation)
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
      impacts = []
      for wall in walls:
        query = wall.segment_query(sensor[0], sensor[1])
        if query.shape is not None:
          point_of_impact = query.point
          impacts.append(point_of_impact)

      # Calculate distance until sensor collides with an object
      start = sensor[0]
      end = sensor[1]
      min_distance = start.get_distance(end)
      for impact in impacts:
        distance = start.get_distance(impact)
        if min_distance is None or distance < min_distance:
          min_distance = distance
          end = impact
      distances.append(min_distance)
      points_of_impact.append(end)

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
    self.fitness = 0

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
    # Manual Control
    self.manual = args['manual'] if 'manual' in args else False
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

    self.init_cars()
    self.init_walls()

    # Dynamic
    self.round = 0
    # If -stepping
    self.step = 0

  def init_cars(self):
    self.cars = []
    Y_RANDOM_RANGE = 20  # 45 - 85 is valid for this map
    X_START = 50
    Y_START_MEAN = 65
    for _ in range(self.NUM_CARS):
      start_x = X_START
      start_y = (np.random.randint(-Y_RANDOM_RANGE,
                                   Y_RANDOM_RANGE) + Y_START_MEAN)
      car = Car(shape=(15, 10),
                position=(start_x, start_y),
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

  def init_walls(self):

    def get_wall(x, y, width, height, color=(104, 114, 117)):
      brick_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
      brick_body.position = x, y
      brick_shape = pymunk.Poly.create_box(brick_body, (width, height))
      brick_shape.color = color
      return brick_shape

    self.walls = []
    level = [
        "WWWWWWWWWWWWWWWWWWWW",
        "W                  W",
        "W                  W",
        "WWWWWWWWWWWWWWW    W",
        "WEEEW         W    W",
        "W   W         W    W",
        "W   W     WWWWW    W",
        "W   W     W        W",
        "W   WWW   W        W",
        "W     W   W        W",
        "W     W   WWWWW    W",
        "W     W  WW        W",
        "W     WWWW         W",
        "W                  W",
        "WWWWWWWWWWWWWWWWWWWW",
    ]
    # Parse the level string above. W = wall, E = exit
    x = y = 0
    x_step = self.SCREEN_WIDTH / len(level[0])
    y_step = self.SCREEN_HEIGHT / len(level)
    # to center the map correctly in the window (WTF I know ...)
    X_OFFSET, Y_OFFSET = 13, 17
    for row in level:
      for col in row:
        if col == "W":
          self.walls.append(
              get_wall(x + X_OFFSET, y + Y_OFFSET, x_step, y_step))
        if col == "E":
          self.walls.append(get_wall(x + X_OFFSET, y + Y_OFFSET,
                                     x_step, y_step, color=(255, 0, 0)))
        x += x_step
      y += y_step
      x = 0

    self.space.add(self.walls)

  def reset(self):
    """Reset game state."""
    for car in self.cars:
      car.reset()

  def calculate_current_fitness(self, car):
    # TODO
    return 0.0

  def build_features(self):
    features = []
    for car in self.cars:
      features.append([car.get_sensor_distances(self.walls)])
    return features

  def predict(self):
    """Predict movements of all cars using simulator."""
    features = self.build_features()
    if self.SIMULATOR:
      movements = self.SIMULATOR.predict(features)
      if movements is False:
        print('[Error] Prediction failed!')
        sys.exit()
    return movements

  def update_track(self):
    # TODO This may be required if our track is larger than the actual screen
    # and we would have to move the camera.
    pass

  def trigger_movements(self):
    """Triggers movements for all cars and allows manual control if
    `self.manual` is set."""
    # Get driving predictions
    if not self.manual:
      movements = self.predict()
      for movement, car in zip(movements, self.cars):
        if movement[0] > 0.5:
          car.trigger_rotate_right()
        if movement[1] > 0.5:
          car.trigger_rotate_left()
        if movement[2] > 0.5:
          car.trigger_acceleration()
    else:
      self.manual_controls()

  def update_cars(self):
    """Updates the position of all cars with the triggered movements and
    checks for collisions."""
    # Move the cars on screen
    for car in self.cars:
      if not car.is_dead:
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

  def manual_controls(self):
    """Allow manual controls of the first car."""
    for event in pygame.event.get():
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

      self.screen.fill((255, 255, 255))

      # Track
      self.update_track()

      # Cars
      self.trigger_movements()
      self.update_cars()

      # Reset check
      if all([car.is_dead for car in self.cars]):
        fitnesses = [car.fitness for car in self.cars]
        self.reset()
        pprint.pprint(fitnesses)
        # Evolution
        if self.SIMULATOR:
          print('Evolving')
          self.SIMULATOR.evolve(fitnesses)

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
      help='Number of cars to spawn and number of networks to use for them.',
      type=int,
      default=None,
      required=True
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
  parser.add_argument(
      '--timeout',
      help='Initial sleep time to allow the ML agent to start.',
      type=int,
      default=None
  )
  parser.add_argument(
      '--manual',
      help='If set, allow the first car to be controlled manually.'
      'Then evolve.',
      action='store_true'
  )
  args = parser.parse_args()
  if args.timeout:
    time.sleep(args.timeout)
  args = args.__dict__
  Game(args).run()
