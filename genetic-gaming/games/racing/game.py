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
from numpy.linalg import norm
import uuid


PI_05 = math.pi * 0.5
PI_03 = math.pi * 0.3
PI_01 = math.pi * 0.1


class MapGenerator(object):
  def __init__(self, min_width, max_width, min_length, max_length, game_height, game_width, max_angle, min_angle=0,
               start_point=None, start_angle=45, start_width=300, seed=None, max_tries=10):
    seed = seed or uuid.uuid4().int

    self.random = random.Random(seed)

    print('Map seed: {}'.format(seed))
    self._min_width = min_width
    self._max_width = max_width
    self._max_angle = max_angle
    self._min_length = min_length
    self._max_length = max_length
    self._game_height = game_height
    self._game_width = game_width
    self._start_point = start_point
    self._start_angle = start_angle
    self._start_width = start_width
    self._min_angle = min_angle
    self._max_tries = max_tries
    self.points = []

  def get_next_endings(self, left_start, right_start, last_angle):
    center = Vec2d((left_start.x + right_start.x)/2, (left_start.y+right_start.y)/2)
    length = self.random.uniform(self._min_length, self._max_length)
    angle = self.random.uniform(self._min_angle, self._max_angle)
    angle = self.random.choice([last_angle + angle, last_angle - angle])
    width = self.random.uniform(self._min_width, self._max_width)
    target_center = Vec2d.unit()
    target_center.angle = angle
    target_center.length = length
    target_center = target_center + center

    left_end = Vec2d.unit()
    left_end.angle = angle - PI_05
    left_end.length = width / 2

    right_end = Vec2d.unit()
    right_end.angle = angle + PI_05
    right_end.length = width / 2

    left_end = target_center + left_end
    right_end = target_center + right_end
    return left_end, right_end, angle, target_center

  def is_valid(self, point):
    return 0 < point.x < self._game_width and 0 < point.y < self._game_height

  def zero_border_vector(self, point):
    def zero_value(val, min=0, max=100):
      if val < min:
        return min
      if val > max:
        return max
      return val

    point.x = zero_value(point.x, max=self._game_width)
    point.y = zero_value(point.y, max=self._game_height)

    return point

  def get_start_points(self):
    if self._start_point is None:
      self._start_point = Vec2d(30, 30)

    left_end = Vec2d.unit()
    left_end.angle = self._start_angle-PI_05
    left_end.length = self._start_width / 2

    right_end = Vec2d.unit()
    right_end.angle = self._start_angle+PI_05
    right_end.length = self._start_width / 2

    return self.zero_border_vector(self._start_point+left_end), \
           self.zero_border_vector(self._start_point+right_end)

  def get_wall(self, start_point, end_point):
    return {
      'start': start_point,
      'end': end_point
    }

  def generate(self):
    last_left, last_right = self.get_start_points()
    last_angle = self._start_angle
    tries_left = self._max_tries

    found = [self.get_wall(last_left, last_right)]
    centers = [Vec2d(self._start_point)]
    while tries_left > 0:
      next_left, next_right, angle, center = self.get_next_endings(last_left, last_right, last_angle)

      if self.is_valid(next_left) and self.is_valid(next_right):
        found.append(self.get_wall(last_left, next_left))
        found.append(self.get_wall(last_right, next_right))
        centers.append(center)
        tries_left = self._max_tries
        last_left = next_left
        last_right = next_right
        last_angle = angle
      else:
        tries_left -= 1

    found.append(self.get_wall(last_left, last_right))

    return found, centers


class FitnessCalculator(object):
  def __init__(self, game):
    self._game = game

  def __call__(self, car):
    raise NotImplementedError()


class DistanceToStartCalculator(FitnessCalculator):
  def __call__(self, car):
    return (car.car_body.position - self._game.centers[0]).length


class DistanceToEndCalculator(FitnessCalculator):
  def __call__(self, car):
    return -(car.car_body.position - self._game.centers[-1]).length


class TimeCalculator(FitnessCalculator):
  def __call__(self, car):
    return time.time() - self._game.start_time


class FastestCalculator(FitnessCalculator):
  def __call__(self, car):
    return max(car.velocities)


class FastestAverageCalculator(FitnessCalculator):
  def __call__(self, car):
    return sum(car.velocities)/len(car.velocities)


class AverageDistanceToPathCalculator(FitnessCalculator):
  def __init__(self, game):
    super().__init__(game)
    self.tracker = game.tracker

  def __call__(self, car):
    return -(sum(self.tracker.distances[car])/len(self.tracker.distances[car]))


def find_closest(points, pos, amount=2):
  dist = np.sum((points - (pos.x, pos.y))**2, axis=1)
  return tuple(np.argsort(dist)[:amount])


class PathDistanceCalculator(FitnessCalculator):
  def __init__(self, game):
    super().__init__(game)
    self._centers = np.asarray([[c.x, c.y] for c in game.centers])
    self._distances = self.calculate_distances()

  def calculate_distances(self):
    results = [0]
    distance = 0
    last = self._game.centers[0]

    for center in self._game.centers[1:]:
      distance += (last - center).length
      results.append(distance)
      last = center

    return results

  def __call__(self, car):
    last, next = find_closest(self._centers, car.car_body.position)

    if last > next:
      last = next

    dist = self._distances[last]

    return (car.car_body.position-self._game.centers[last]).length + dist


class FastestAveragePathCalculator(FitnessCalculator):
  def __init__(self, game):
    super().__init__(game)
    self._path_distance_calculator = PathDistanceCalculator(game)

  def __call__(self, car):
    WEIGHT_SPEED, WEIGHT_PATH = 1, 10
    return WEIGHT_SPEED * sum(car.velocities)/len(car.velocities) + WEIGHT_PATH * self._path_distance_calculator(car)


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
    self.car_shape.filter = pymunk.ShapeFilter(categories=0x2)
    self.velocities = []

    # Dynamic
    self.reset()

  def reset(self):
    """Reset car to initial settings."""
    # Pymunk
    self.car_body.position = self._position
    self.car_shape.color = self._color
    self.car_shape.elasticity = 1.0
    self.car_shape.sensor = True
    self.car_body.angle = self._rotation
    self.car_body.velocity = Vec2d(0, 0)
    driving_direction = Vec2d(0, 0).rotated(self.car_body.angle)
    self.car_body.apply_impulse_at_world_point(driving_direction)

    # Dynamic
    self.velocity = 0
    self.rotation = self._rotation
    self.current_acceleration_time = 0
    self.is_dead = False
    self.fitness = 0.0
    self.previous_position = self._position

  def add_to_space(self, space):
    """Adds both car_body and car_shape to the space if none has been set
    yet."""
    if self.car_body.space is None:
      space.add(self.car_body, self.car_shape)

  def remove_from_space(self):
    """Removes the car_body and car_shape from their space if one is set."""
    if self.car_body.space is not None:
      self.car_body.space.remove(self.car_body, self.car_shape)

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
      space = self.car_body.space
      query = space.segment_query_first(
          sensor[0], sensor[1],
          radius=0, shape_filter=pymunk.ShapeFilter(mask=0x1))
      if query is not None:
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
    self.velocities.append(self.car_body.velocity.get_length())


  @staticmethod
  def get_rotated_point(x_1, y_1, x_2, y_2, radians):
    """Rotate x_2, y_2 around x_1, y_1 by angle."""
    x_change = (x_2 - x_1) * math.cos(radians) + \
        (y_2 - y_1) * math.sin(radians)
    y_change = (y_1 - y_2) * math.cos(radians) - \
        (x_1 - x_2) * math.sin(radians)
    new_x = x_change + x_1
    new_y = y_change + y_1
    return int(new_x), int(new_y)


class DistanceTracker(object):
  def __init__(self, centers, cars, pause_calls=0):
    self.distances = {}
    self.cars = cars
    self.init_pause_calls = pause_calls
    self.pause_counter = pause_calls
    for car in cars:
      self.distances[car] = []

    self._centers = np.asarray([[c.x, c.y] for c in centers])

  def calc_distance(self, car):
    last, next = find_closest(self._centers, car.car_body.position)
    last = self._centers[last]
    next = self._centers[next]
    return norm(np.cross(next - last, last - car.car_body.position)) / norm(next - last)

  def calculate_distances(self):
    self.pause_counter -= 1
    if self.pause_counter <= 0:
      for car in self.cars:
        self.distances[car].append(self.calc_distance(car))
      self.pause_counter = self.init_pause_calls


class Game(object):
  FITNESS_CALCULATORS = {
    'distance_to_start': DistanceToStartCalculator,
    'distance_to_end': DistanceToEndCalculator,
    'time': TimeCalculator,
    'path': PathDistanceCalculator,
    'fastest': FastestCalculator,
    'fastest_average': FastestAverageCalculator,
    'fastest_average_path': FastestAveragePathCalculator,
    'average_path_distance': AverageDistanceToPathCalculator
  }

  def __init__(self, args, simulator=None):
    # Manual Control
    self.manual = args['manual'] if 'manual' in args else False
    # EvolutionServer
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_CARS = args['num_networks']
    self.SEND_PIXELS = args['send_pixels']
    self.SIMULATOR = simulator
    if args['restore_networks']:
      self.SIMULATOR.restore_networks()

    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

    # Game
    self.STEPPING = args['stepping']
    self.MAP_GENERATOR = args.get('map_generator', 'random')
    self.GAME_SEED = args.get('game_seed', None)
    np.random.seed(self.GAME_SEED // 2**96)
    self.FITNESS_MODE = args.get('fitness_mode', 'distance_to_start')
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

    X_START = 50
    Y_START_MEAN = 65

    # Game vars
    self.walls = []
    self.centers = []

    self.init_cars(x_start=X_START, y_start=Y_START_MEAN)
    self.init_walls(x_start=X_START-10, y_start=Y_START_MEAN)
    self.init_tracker()

    # Init fitness last because calculator might depend on cars/wall/tracker
    self.init_fitness(self.FITNESS_MODE)

    # Dynamic
    self.reset()
    # If -stepping
    self.step = 0

  def init_cars(self, x_start, y_start):
    self.cars = []
    Y_RANDOM_RANGE = 20  # 45 - 85 is valid for this map
    for _ in range(self.NUM_CARS):
      start_x = x_start
      start_y = (np.random.randint(-Y_RANDOM_RANGE,
                                   Y_RANDOM_RANGE) + y_start)
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

  def init_tracker(self):
    self.tracker = DistanceTracker(self.centers, self.cars, 5)

  def init_walls(self, x_start, y_start):
    generators = {
      'random': self.init_walls_randomly,
      'map': self.init_walls_with_map
    }
    generators.get(self.MAP_GENERATOR)(x_start, y_start)

  def init_walls_randomly(self, x_start, y_start):
    self.walls = []

    gen = MapGenerator(
      min_width=40, max_width=70,
      min_angle=PI_01, max_angle=PI_03,
      min_length=100, max_length=200,
      game_height=self.SCREEN_HEIGHT, game_width=self.SCREEN_WIDTH,
      start_point=(x_start, y_start), start_angle=0, start_width=100,
      seed=self.GAME_SEED
    )
    wall_layout, centers = gen.generate()
    self.centers = centers

    def get_wall(start, end):
      body = pymunk.Body(body_type=pymunk.Body.STATIC)
      body.width = 5
      segment = pymunk.Segment(body, start, end, 0)
      segment.filter = pymunk.ShapeFilter(categories=0x1)
      return segment

    for layout in wall_layout:
      wall = get_wall(**layout)
      self.walls.append(wall)

    self.space.add(self.walls)

  def init_walls_with_map(self, x_start, y_start):

    def get_wall(x, y, width, height, color=(104, 114, 117)):
      brick_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
      brick_body.position = x, y
      brick_shape = pymunk.Poly.create_box(brick_body, (width, height))
      brick_shape.filter = pymunk.ShapeFilter(categories=0x1)
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
    X_OFFSET, Y_OFFSET = 13, 17  # to center the map correctly in the window (WTF I know ...)
    for row in level:
      for col in row:
        if col == "W":
          self.walls.append(get_wall(x + X_OFFSET, y + Y_OFFSET, x_step, y_step))
        if col == "E":
          self.walls.append(get_wall(x + X_OFFSET, y + Y_OFFSET, x_step, y_step, color=(255, 0, 0)))
        x += x_step
      y += y_step
      x = 0

    self.space.add(self.walls)

  def reset(self):
    """Reset game state (all cars)."""
    self.round = 0
    self.start_time = time.time()
    self.car_velocity_timer = {}
    for car in self.cars:
      car.reset()
      car.add_to_space(self.space)
      self.car_velocity_timer.update({car: self.start_time})

  def calculate_current_fitness(self, car):
    return self._fitness_calc(car)

  def init_fitness(self, mode):
    self._fitness_calc = self.FITNESS_CALCULATORS[mode](self)

  def build_features(self):
    features = []
    for car in self.cars:
      if car.is_dead:
        # TODO Make this dynamically adjust to num_sensors
        features.append([[0.0 for _ in range(5)]])
      else:
        features.append([car.get_sensor_distances(self.walls)])
    return features

  def predict(self):
    """Predict movements of all cars using `self.SIMULATOR."""
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
    """Triggers movements for all cars and allows manual keyboard control if
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
        self.check_for_car_not_moving(car)
        self.tracker.calculate_distances()

  def kill_car(self, car):
    car.is_dead = True
    car.fitness = self.calculate_current_fitness(car)
    car.remove_from_space()

  def check_for_car_not_moving(self, car):
    x_velocity, y_velocity = car.car_body.velocity
    if x_velocity > 0 or y_velocity > 0:
      self.car_velocity_timer[car] = time.time()
    elif time.time() - self.car_velocity_timer[car] > 3:
      self.kill_car(car)

  def check_for_collision(self, car):
    """Checks is any sensor distance is below the threshold. If so, mark car as
    dead, set cars fitness and remove it from the space."""
    walls_in_range = self.space.point_query(
        car.car_body.position, car._sensor_range, pymunk.ShapeFilter(mask=0x1))
    walls_in_range = [query.shape for query in walls_in_range]
    distances = car.get_sensor_distances(walls_in_range, self.screen)
    for distance in distances:
        # TODO Find suitable collision threshold
        # If you run into a border "distance" will only be zero when the car is
        # already stuck in the center of the wall
      if distance < 5.0:
        self.kill_car(car)

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
      clock.tick(120)

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()

      self.screen.fill((255, 255, 255))

      # Update Track
      self.update_track()

      # Update Cars
      self.trigger_movements()
      self.update_cars()

      # Reset Game & Update Networks, if all cars are dead
      if (all([car.is_dead for car in self.cars]) or
              time.time() - self.start_time > 40):
        fitnesses = [car.fitness for car in self.cars]
        self.reset()
        print('Fitnesses:')
        pprint.pprint(fitnesses)
        # Evolution
        if self.SIMULATOR:
          if sum(fitnesses) == 0:
            print('Resetting networks')
            self.SIMULATOR.reset()
          else:
            print('Evolving')
            self.SIMULATOR.evolve(fitnesses)

      # Draw centers

      for center in self.centers:
        pygame.draw.circle(self.screen, 0x00ff00, (int(round(center.x)), int(round(center.y))), 5)

      # Pymunk & Pygame calls
      self.space.debug_draw(self.draw_options)
      pygame.display.update()
      fps = 60
      dt = 1. / fps
      self.space.step(dt)

  def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
    """Rotates a point (x2, y2) around (x1, y1) by radians."""
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
