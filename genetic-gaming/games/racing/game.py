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
from . import maps, fitness, drawoptions
from . import car as car_impl
import json


class Game(object):

  def __init__(self, args, simulator=None):
    # Manual Control
    self.manual = args.get('manual', False)
    # EvolutionServer
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_CARS = args['num_networks']
    self.SEND_PIXELS = args['send_pixels']
    self.SIMULATOR = simulator
    if args['restore_from']:
      self.SIMULATOR.restore_networks(args['restore_from'])

    # Racing game only settings
    game_settings = args['racing_game']
    self.VELOCITY_AS_INPUT = game_settings['velocity_as_input']
    self.NUM_CAR_SENSORS = game_settings['num_car_sensors']

    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

    # Game
    self.fps = 60
    self.STEPPING = args['stepping']
    self.MAP_GENERATOR = args.get('map_generator', 'random')
    self.GAME_SEED = args['game_seed']
    self.current_seed = self.GAME_SEED
    np.random.seed(self.current_seed // 2**96)
    self.FITNESS_MODE = args.get('fitness_mode', 'distance_to_start')
    self.FITNESS_CONF = args.get('fitness_function_conf', [])
    self.SCREEN_RESIZE_SHAPE = args.get('screen_resize_shape', None)
    self.ASSET_DIR = args.get('assets', 'assets')
    self.SCREEN_WIDTH = 640
    self.SCREEN_HEIGHT = 480
    self.GAME_WIDTH = 1280
    self.GAME_HEIGHT = 960
    self.MAP_GENERATOR_CONF = args.get('map_generator_conf', {})
    self.START_MODE = args['start_mode']
    self.RANDOMIZE_MAP = args['randomize_map']
    self.FIX_MAP_ROUNDS = args['fix_map_rounds']
    self.AGGREGATE_MAPS = args['aggregate_maps']
    self.MAX_ROUNDS = args['max_rounds']
    self.fix_map_rounds_left = self.FIX_MAP_ROUNDS
    self.fitness_history = []
    self.finish_history = []
    self.save_to = args['save_to']

    # Pygame
    # flags = pygame.HWSURFACE | pygame.FULLSCREEN
    self.screen = pygame.display.set_mode(
        (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    self.screen.set_alpha(None)
    pygame.display.set_caption('GG Racing (' + self.FITNESS_MODE + ')')
    asset_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(asset_path, self.ASSET_DIR)

    # Pymunk
    pymunk.pygame_util.positive_y_is_up = False
    self.space = pymunk.Space(threaded=True)
    self.space.gravity = pymunk.Vec2d(0., 0.)
    self.draw_options = drawoptions.OffsetDrawOptions(self.screen)

    # Collision Detection
    wall_coll_handler = self.space.add_collision_handler(1, 2)

    def collision_handler(arbiter, space, data):
      shapes = arbiter.shapes
      for shape in shapes:
        car = getattr(shape, 'car', None)
        if car:
          self.kill_car(car)
      return False
    wall_coll_handler.begin = collision_handler

    self.X_START = 50
    self.Y_START = self.GAME_HEIGHT / 2
    self.Y_RANDOM_RANGE = 20

    # Game vars
    self.walls = []
    self.centers = []
    self.start_region = None
    self.finish_region = None
    self.finishing_line_components = []
    self.starting_line_components = []

    self.init_cars(x_start=self.X_START, y_start=self.Y_START)
    self.init_walls(x_start=self.X_START - 10, y_start=self.Y_START)
    self.init_tracker()

    # Init fitness last because calculator might depend on cars/wall/tracker
    self.init_fitness(self.FITNESS_MODE)

    # Dynamic
    self._last_best_car = None
    self._last_best_fitness = None
    self._camera_car = None
    self.reset(no_map_reset=True)
    self.round = self.SIMULATOR.current_step
    self.frames = 0
    # If -stepping
    self.step = 0

  def get_start_pos(self, x, y):
    if self.START_MODE in ['random_first', 'random_each']:
      y = (np.random.randint(-self.Y_RANDOM_RANGE, self.Y_RANDOM_RANGE) + y)
    return x, y

  def init_cars(self, x_start, y_start):
    self.cars = []
    car_colors = []
    for _ in range(self.NUM_CARS):
      start_x, start_y = self.get_start_pos(x_start, y_start)

      while True:
        car_color = (np.random.randint(0, 256),
                     np.random.randint(0, 256),
                     np.random.randint(0, 256))
        if car_color not in car_colors:
          break

      car = car_impl.Car(shape=(15, 10),
                         position=(start_x, start_y),
                         rotation=0.0,
                         rotation_speed=0.2,
                         base_velocity=5.0,
                         acceleration=100,
                         deceleration=0.8,
                         acceleration_time=20,
                         min_velocity=-100,
                         max_velocity=100,
                         color=car_color,
                         sensor_range=500,
                         num_sensors=self.NUM_CAR_SENSORS)
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
    self.space.remove(self.walls)
    self.space.remove(self.finishing_line_components)
    self.space.remove(self.starting_line_components)

    self.walls = []
    self.finishing_line_components = []
    self.starting_line_components = []
    self.centers = []

    gen = maps.MapGenerator(
        game_height=self.GAME_HEIGHT, game_width=self.GAME_WIDTH,
        start_point=(x_start, y_start), start_angle=0, start_width=100,
        seed=self.current_seed, **self.MAP_GENERATOR_CONF
    )
    wall_layout, centers = gen.generate()
    self.centers = centers

    def get_wall(start, end):
      body = pymunk.Body(body_type=pymunk.Body.STATIC)
      body.width = 5
      segment = pymunk.Segment(body, start, end, 0)
      segment.filter = pymunk.ShapeFilter(categories=0x1)
      segment.collision_type = 2
      return segment

    def get_dashed_line(start, end, dash_length=4, color=(0, 0, 0)):
      """Creates a black & white dashed line from start to end"""
      length = start.get_distance(end)
      displacement = end - start
      slope = displacement / length
      line_parts = []
      for i in range(0, int(length / dash_length), 2):
        new_start = start + (slope * i * dash_length)
        new_end = start + (slope * (i + 1) * dash_length)
        if start.get_distance(new_end) > length:
          new_end = end
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        segment = pymunk.Segment(body, new_start, new_end, 3)
        segment.filter = pymunk.ShapeFilter(categories=0x2)
        segment.color = color
        line_parts.append(segment)
      return line_parts

    for layout in wall_layout:
      wall = get_wall(**layout)
      self.walls.append(wall)

    self.space.add(self.walls)

    if len(wall_layout) > 2:
      # Create starting line
      start_wall_1 = wall_layout[1]
      start_wall_2 = wall_layout[2]
      start_wall_1_half = (start_wall_1['start'] + start_wall_1['end']) / 2
      start_wall_2_half = (start_wall_2['start'] + start_wall_2['end']) / 2
      line_parts = get_dashed_line(start_wall_1_half, start_wall_2_half)
      self.space.add(line_parts)
      self.starting_line_components = line_parts
      # Create finish line
      finish_wall_1 = wall_layout[-3]
      finish_wall_2 = wall_layout[-2]
      finish_wall_1_half = (finish_wall_1['start'] + finish_wall_1['end']) / 2
      finish_wall_2_half = (finish_wall_2['start'] + finish_wall_2['end']) / 2
      line_parts = get_dashed_line(finish_wall_1_half, finish_wall_2_half)
      self.finishing_line_components = line_parts
      self.space.add(line_parts)

      self.start_region = pymunk.Poly(
          pymunk.Body(body_type=pymunk.Body.STATIC),
          [start_wall_1['start'], start_wall_1_half,
           start_wall_2_half, start_wall_2['start']])
      self.start_region.filter = pymunk.ShapeFilter(categories=0x4)

      self.finish_region = pymunk.Poly(
          pymunk.Body(body_type=pymunk.Body.STATIC),
          [finish_wall_1_half, finish_wall_1['end'],
           finish_wall_2['end'], finish_wall_2_half])
      self.finish_region.filter = pymunk.ShapeFilter(categories=0x4)

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

  def reset(self, no_map_reset=False):
    """Reset game state (all cars) and if """
    if self.RANDOMIZE_MAP and self.fix_map_rounds_left <= 0 and not no_map_reset:
      self.current_seed += 1
      self.init_walls(x_start=self.X_START - 10, y_start=self.Y_START)
      self.init_tracker()

    self.frames = 0
    self.round_finish_timer = None
    self.car_idle_frames = {}
    for car in self.cars:
      new_pos = self.get_start_pos(self.X_START, self.Y_START) \
          if self.START_MODE == 'random_each' else None
      car.reset(new_pos)
      car.add_to_space(self.space)
      self.car_idle_frames.update({car: self.frames})

  def calculate_current_fitness(self, car):
    return self._fitness_calc(car)

  def init_fitness(self, mode):
    self._last_fitnesses = []
    self._fitness_calc = fitness.FITNESS_CALCULATORS[mode](
        self, self.FITNESS_CONF)

  def build_features(self):
    features = []
    # TODO Make inputs list [[]] great again
    for car in self.cars:
      if car.is_dead:
        num_inputs = car.num_sensors + int(self.VELOCITY_AS_INPUT)
        features.append([[0.0 for _ in range(num_inputs)]])
      else:
        inputs = [car.get_sensor_distances(self.walls)]
        if self.VELOCITY_AS_INPUT:
          inputs[0] += [car.velocity.length / car._max_velocity]
        features.append(inputs)
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

  def select_new_camera_car(self):
    furthest = self.cars[0]
    furthest_dist = 0
    for car in self.cars:
      dist = (car.car_body.position - self.centers[0]).length
      if dist > furthest_dist and not car.is_dead:
        furthest = car
        furthest_dist = dist

    return furthest

  def update_offset(self):
    self._camera_car = self.select_new_camera_car()

    self.draw_options.offset = \
        (-self._camera_car.car_body.position +
         (pymunk.Vec2d(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) * 0.5))

    for car in self.cars:
      car.update_offset(self.draw_options.offset)

  def trigger_movements(self):
    """Triggers movements for all cars and allows manual keyboard control if
    `self.manual` is set."""
    # Get driving predictions
    if not self.manual:
      movements = self.predict()
      for movement, car in zip(movements, self.cars):
        if not car.is_dead:
          car.trigger_rotation(movement[0])
          car.last_turn = movement[0]
          car.trigger_acceleration(movement[1])
          car.last_acceleration = movement[1]
    else:
      self.manual_controls()

  def update_cars(self):
    """Updates the position of all cars with the triggered movements."""
    # Move the cars on screen
    cars_in_start_region = self.get_cars_in_region(self.start_region)
    cars_in_finish_region = self.get_cars_in_region(self.finish_region)
    finishes = []
    for car in self.cars:
      if not car.is_dead:
        car.move()
        self.render_car_sensor(car)
        self.kill_car_if_idle(car, car in cars_in_start_region)
        self.tracker.calculate_distances()
      if car in cars_in_finish_region:
        if self.round_finish_timer is None:
          self.round_finish_timer = self.frames
        finishes.append(1)
      else:
        finishes.append(0)
    self.finish_history.append(finishes)

  def kill_car(self, car):
    car.is_dead = True
    car.fitness = self.calculate_current_fitness(car)
    car.car_shape.color = (205, 206, 214)
    car.car_body.velocity = pymunk.Vec2d(0, 0)

  def get_cars_in_region(self, region):
    query_shapes = [q.shape for q in self.space.shape_query(region)]
    cars = []
    for car in self.cars:
      if car.car_shape in query_shapes:
        cars.append(car)
    return cars

  def kill_car_if_idle(self, car, car_in_start_region):
    x_velocity, y_velocity = car.car_body.velocity
    if (x_velocity > sys.float_info.epsilon or
            y_velocity > sys.float_info.epsilon) and not car_in_start_region:
      self.car_idle_frames[car] = self.frames
    elif self.frames - self.car_idle_frames[car] > self.fps * 4:
      self.kill_car(car)
      car.fitness = -sys.maxsize

  def render_car_sensor(self, car):
    """Checks is any sensor distance is below the threshold. If so, mark car as
    dead, set cars fitness and remove it from the space."""
    walls_in_range = self.space.point_query(
        car.car_body.position, car._sensor_range, pymunk.ShapeFilter(mask=0x1))
    walls_in_range = [query.shape for query in walls_in_range]
    car.get_sensor_distances(walls_in_range, self.screen)

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

  def render_sidebar(self):
    font = pygame.font.SysFont("Arial", 15)
    bar_length = 150
    x_position = 20
    pygame.draw.rect(self.screen, (255, 255, 255),
                     pygame.Rect(0, 0, bar_length + 35,
                                 self.GAME_HEIGHT))
    self.screen.blit(
        font.render(
            str('Round: {}'.format(self.round)),
            -1,
            (0, 0, 0)),
        (x_position, 20))
    if self._last_best_car is not None:
      self.screen.blit(
          font.render(
              str('Best:        {:.2f}'.format(self._last_best_fitness)),
              -1,
              (0, 0, 0)),
          (x_position + 80, 20))
      pygame.draw.rect(self.screen, self._last_best_car._color,
                       pygame.Rect(x_position + 110, 20 + 5, 15, 10))
    i = 0
    for car in self.cars:
      if not car.is_dead:
        y_position = 45 + 30 * i
        bar_y_position = 40 + 30 * i
        pygame.draw.rect(self.screen, car._color,
                         pygame.Rect(x_position, y_position + 5, 15, 10))
        alive_text = 'dead' if car.is_dead else 'alive'
        alive_color = (183, 18, 43) if car.is_dead else (66, 244, 69)
        self.screen.blit(font.render(alive_text, -1, alive_color),
                         (x_position + 18, y_position))
        pygame.draw.line(self.screen, (95, 105, 119),
                         (x_position, bar_y_position),
                         (x_position + bar_length, bar_y_position))

        def create_move_bar(probability, x, y, width, max_height):
          probability *= -1
          bar_color = (183, 18, 43) if probability > 0 else (66, 244, 69)
          max_height = max_height / 2
          y += max_height
          # -2 so it does not overlap with the horizontal lines
          bar_height = (max_height - 2) * probability
          bar_rect = pygame.Rect(x, y, width, bar_height)
          pygame.draw.rect(self.screen, bar_color, bar_rect)

        bar_width = 20
        bar_max_height = 30
        # Right turn
        x_bar = x_position + 70
        x_text = x_position + 55
        rotate_label = 'R' if car.last_turn > 0 else 'L'
        self.screen.blit(font.render('{}:'.format(rotate_label), -1,
                                     (0, 0, 0)), (x_text, y_position))
        create_move_bar(car.last_turn, x_bar,
                        bar_y_position, bar_width, bar_max_height)
        # Left turn
        x_bar = x_position + 110
        x_text = x_position + 95
        self.screen.blit(font.render('A:', -1, (0, 0, 0)),
                         (x_text, y_position))
        create_move_bar(car.last_acceleration, x_bar,
                        bar_y_position, bar_width, bar_max_height)
        i += 1

    # Render last line
    bar_y_position = 40 + 30 * i
    pygame.draw.line(self.screen, (95, 105, 119), (x_position, bar_y_position),
                     (x_position + bar_length, bar_y_position))

  def run(self):
    self.clock = pygame.time.Clock()
    self.start_time = time.time()
    pygame.font.init()

    round_time = time.time()

    while True:
      self.clock.tick()
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()

      self.screen.fill((255, 255, 255))

      # Update Track
      self.update_offset()

      # Update Cars
      self.trigger_movements()
      self.update_cars()

      # Reset Game & Update Networks, if all cars are dead
      if (all([car.is_dead for car in self.cars]) or
          (self.round_finish_timer is not None and
           self.frames - self.round_finish_timer > self.fps * 5)):
        print('====== Finished step {}/{} in round {} in {} sec ======'.format(
            len(self._last_fitnesses), self.AGGREGATE_MAPS, self.round,
            time.time() - round_time))
        if self.fix_map_rounds_left > 0:
          print('Rounds left until randomization: {}'.format(
                self.fix_map_rounds_left))
        self.fix_map_rounds_left -= 1
        round_time = time.time()
        # Calculate fitness of cars still alive
        for car in self.cars:
          if not car.is_dead:
            self.kill_car(car)

        self.store_fitnesses()
        fitnesses = [car.fitness for car in self.cars]
        self.fitness_history.append(fitnesses)

        if 0 < self.MAX_ROUNDS <= self.round:
          print('###### EXITING BECAUSE OF ROUND LIMIT IN ROUND {}'
                ' #####'.format(self.round))
          if self.save_to:
            with open(os.path.join(self.save_to, "fitness_history.json"),
                      "w") as f:
              json.dump(self.fitness_history, f)
            with open(os.path.join(self.save_to, "finish_history.json"),
                      "w") as f:
              json.dump(self.finish_history, f)
          return

        if len(self._last_fitnesses) == self.AGGREGATE_MAPS:
          self.run_evolution()

        self.reset()

      # Draw centers
      for center in self.centers:
        pygame.draw.circle(self.screen, 0x00ff00, (int(
            round(center.x + self.draw_options.offset.x)),
            int(round(center.y + self.draw_options.offset.y))), 5)

      # Pymunk & Pygame calls
      if os.environ.get('SDL_VIDEODRIVER') is None:
        self.space.debug_draw(self.draw_options)
        # Show sidebar here so it overlays the map
        self.render_sidebar()
        pygame.display.update()
      dt = 1. / (self.fps)
      self.space.step(dt)
      self.frames += 1

  def run_evolution(self):
    fitnesses = [0 for _ in self._last_fitnesses[0]]

    for round_results in self._last_fitnesses:
      for k, v in enumerate(round_results):
        fitnesses[k] += v

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

  def store_fitnesses(self):
    fitnesses = [car.fitness for car in self.cars]
    self._last_best_car = max(self.cars, key=lambda car: car.fitness)
    self._last_best_fitness = self._last_best_car.fitness
    if len(self._last_fitnesses) == self.AGGREGATE_MAPS:
      self._last_fitnesses.pop(0)
    self._last_fitnesses.append(fitnesses)

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
