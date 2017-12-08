import argparse
import sys
import os
import math
import msgpackrpc
import numpy as np
import time
import pprint
import pygame
import pymunk
import pymunk.pygame_util
from . import maps, fitness
from . import car as car_impl


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
    if args['restore_networks']:
      self.SIMULATOR.restore_networks()

    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

    # Game
    self.STEPPING = args['stepping']
    self.MAP_GENERATOR = args.get('map_generator', 'random')
    self.GAME_SEED = args.get('game_seed', None)
    if self.GAME_SEED is not None:
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
    self.MAP_GENERATOR_CONF = args.get('map_generator_conf', {})

    # Pygame
    self.screen = pygame.display.set_mode(
        (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    pygame.display.set_caption('GG Racing (' + self.FITNESS_MODE + ')')
    asset_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(asset_path, args['assets'])

    # Pymunk
    pymunk.pygame_util.positive_y_is_up = False
    self.space = pymunk.Space(threaded=True)
    self.space.gravity = pymunk.Vec2d(0., 0.)
    self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    X_START = 50
    Y_START_MEAN = 65

    # Game vars
    self.walls = []
    self.centers = []

    self.init_cars(x_start=X_START, y_start=Y_START_MEAN)
    self.init_walls(x_start=X_START - 10, y_start=Y_START_MEAN)
    self.init_tracker()

    # Init fitness last because calculator might depend on cars/wall/tracker
    self.init_fitness(self.FITNESS_MODE)

    # Dynamic
    self.reset()
    # If -stepping
    self.step = 0
    self.round = 1

  def init_cars(self, x_start, y_start):
    self.cars = []
    Y_RANDOM_RANGE = 20  # 45 - 85 is valid for this map
    car_colors = []
    for _ in range(self.NUM_CARS):
      start_x = x_start
      start_y = (np.random.randint(-Y_RANDOM_RANGE,
                                   Y_RANDOM_RANGE) + y_start)

      while True:
        car_color = (np.random.randint(0, 256),
                     np.random.randint(0, 256),
                     np.random.randint(0, 256))
        if car_color not in car_colors:
          break

      car = car_impl.Car(shape=(15, 10),
                         position=(start_x, start_y),
                         rotation=0.0,
                         rotation_speed=0.05,
                         base_velocity=5.0,
                         acceleration=1.1,
                         deceleration=0.8,
                         acceleration_time=20,
                         max_velocity=100,
                         color=car_color,
                         sensor_range=100,
                         num_sensors=2)
      self.cars.append(car)
      car.add_to_space(self.space)

  def init_tracker(self):
    self.tracker = fitness.DistanceTracker(self.centers, self.cars, 5)

  def init_walls(self, x_start, y_start):
    generators = {
        'random': self.init_walls_randomly,
        'map': self.init_walls_with_map
    }
    generators.get(self.MAP_GENERATOR)(x_start, y_start)

  def init_walls_randomly(self, x_start, y_start):
    self.walls = []

    gen = maps.MapGenerator(
        game_height=self.SCREEN_HEIGHT, game_width=self.SCREEN_WIDTH,
        start_point=(x_start, y_start), start_angle=0, start_width=100,
        seed=self.GAME_SEED, **self.MAP_GENERATOR_CONF
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
    """Reset game state (all cars)."""
    self.start_time = time.time()
    self.car_velocity_timer = {}
    for car in self.cars:
      car.reset()
      car.add_to_space(self.space)
      self.car_velocity_timer.update({car: self.start_time})

  def calculate_current_fitness(self, car):
    return self._fitness_calc(car)

  def init_fitness(self, mode):
    self._fitness_calc = fitness.FITNESS_CALCULATORS[mode](self)

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
    if (x_velocity > sys.float_info.epsilon or
            y_velocity > sys.float_info.epsilon):
      self.car_velocity_timer[car] = time.time()
    elif time.time() - self.car_velocity_timer[car] > 3:
      self.kill_car(car)
      car.fitness = -sys.maxsize

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

  def render_statistics(self):
    font = pygame.font.SysFont("Arial", 15)
    x_position = 640 - 70
    self.screen.blit(
        font.render(
            str('Round: {}'.format(self.round)),
            -1,
            (0, 0, 0)),
        (x_position, 20))
    for i, car in enumerate(self.cars):
      y_position = 40 + 20 * i
      pygame.draw.rect(self.screen, car._color,
                       pygame.Rect(x_position, y_position + 5, 15, 10))
      text = 'dead' if car.is_dead else 'alive'
      color = (183, 18, 43) if car.is_dead else (66, 244, 69)
      self.screen.blit(font.render(text, -1, color),
                       (x_position + 18, y_position))

  def run(self):
    # clock = pygame.time.Clock()
    pygame.font.init()

    round_time = time.time()

    while True:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()

      self.screen.fill((255, 255, 255))

      # Update Track
      self.update_track()

      # Update Cars
      self.trigger_movements()
      self.update_cars()

      # Show statistics
      self.render_statistics()

      # Reset Game & Update Networks, if all cars are dead
      if (all([car.is_dead for car in self.cars]) or
              time.time() - self.start_time > 40):
        print('====== Finished round {} in {} sec ======'.format(
            self.round, time.time() - round_time))
        round_time = time.time()
        # Calculate fitness of cars still alive
        for car in self.cars:
          if not car.is_dead:
            self.kill_car(car)
        fitnesses = [car.fitness for car in self.cars]
        self.reset()
        self.round += 1
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
        pygame.draw.circle(self.screen, 0x00ff00, (int(
            round(center.x)), int(round(center.y))), 5)

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
