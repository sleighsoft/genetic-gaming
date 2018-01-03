import sys
import msgpackrpc
import numpy as np
import time
import pprint
import pygame
from ..saver import load_saved_data, save_data, get_current_git_hash
from .gameinstance import GameInstance


class Player(object):
  def __init__(self, color):
    self.color = color
    self._cars = []

  def add_car(self, car):
    self._cars.append(car)

  def get_fitness(self):
    return sum(c.fitness for c in self._cars)


class Game(object):

  def __init__(self, args, simulator=None):
    self._args = args

    self._games = []
    self._surfaces = []
    self._players = []

    restore_dir = None
    if args['restore_from'] is not None:
        restore_dir = args['restore_from']
        save_dir = args['save_to']
        try:
            saved_data = load_saved_data(restore_dir)
            version, args = saved_data['version'], saved_data['args']
            if save_dir is not None:
              args['save_dir'] = save_dir  # Keep on saving
            else:
              del args['save_dir']  # Don't overwrite saved state
            current_hash = get_current_git_hash()
            if current_hash != version:
                print("Attention: The saved game was compiled in commit {} "
                      "while the current commit is {}.".format(version, current_hash))
        except ValueError as e:
            print("An error occurred while trying to restore the specified data: {}".format(e))
    elif args['save_to'] is not None:
        try:
            save_data(args)
        except ValueError as e:
            print("An error occurred while trying to save the specified data: {}".format(e))

    # Manual Control
    self.manual = args.get('manual', False)
    # EvolutionServer
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_CARS = args['num_networks']
    self.SEND_PIXELS = args['send_pixels']
    self.SIMULATOR = simulator
    if restore_dir is not None:
      self.SIMULATOR.restore_networks(restore_dir)

    # Racing game only settings
    game_settings = args['racing_game']
    self.VELOCITY_AS_INPUT = game_settings['velocity_as_input']
    self.NUM_CAR_SENSORS = game_settings['num_car_sensors']

    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

    # Game

    # Hardcoded for now. Do not change this because otherwise rendering logic does not work.
    self.NUM_GAMES = 4

    self.STEPPING = args['stepping']
    self.MAP_GENERATOR = args.get('map_generator', 'random')
    self.GAME_SEED = args.get('game_seed', None)
    if self.GAME_SEED is not None:
      np.random.seed(self.GAME_SEED // 2**96)
    self.FITNESS_MODE = args.get('fitness_mode', 'distance_to_start')
    self.FITNESS_CONF = args.get('fitness_function_conf', {})
    self.SCREEN_RESIZE_SHAPE = args.get('screen_resize_shape', None)
    self.ASSET_DIR = args.get('assets', 'assets')
    self.SCREEN_WIDTH = 640
    self.SCREEN_HEIGHT = 480
    self.GAME_WIDTH = 1280
    self.GAME_HEIGHT = 960
    self.MAP_GENERATOR_CONF = args.get('map_generator_conf', {})
    self.START_MODE = args['start_mode']

    # Pygame
    # flags = pygame.HWSURFACE | pygame.FULLSCREEN
    self.init_screen()
    self.init_players()
    self.init_games()
    self.round = 0

  def init_screen(self):
    self.screen = pygame.display.set_mode(
        (self.SCREEN_WIDTH*2, self.SCREEN_HEIGHT*2))
    self.screen.set_alpha(None)
    pygame.display.set_caption('GG Racing (' + self.FITNESS_MODE + ')')

    # Initialize 4 Game screens
    size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
    self._surfaces.append(self.screen.subsurface((0, 0), size))
    self._surfaces.append(self.screen.subsurface((self.SCREEN_WIDTH, 0), size))
    self._surfaces.append(self.screen.subsurface((0, self.SCREEN_HEIGHT), size))
    self._surfaces.append(self.screen.subsurface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), size))

  def init_games(self):
    for i in range(0, self.NUM_GAMES):
        self._games.append(GameInstance(self._args, self, self._surfaces[i], self._players))

  def init_players(self):
    car_colors = []
    for i in range(0, self.NUM_CARS):
      while True:
        car_color = (np.random.randint(0, 256),
                     np.random.randint(0, 256),
                     np.random.randint(0, 256))
        if car_color not in car_colors:
          break
      self._players.append(Player(car_color))

  def reset(self):
    for g in self._games:
        g.reset()

  def predict(self, features):
    if self.SIMULATOR:
      movements = self.SIMULATOR.predict(features)
      if movements is False:
        print('[Error] Prediction failed!')
        sys.exit()
    return movements

  def run(self):
    # clock = pygame.time.Clock()
    pygame.font.init()

    round_time = time.time()
    fps = 60
    dt = 1. / fps

    while True:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()

      for g in self._games:
          if not g.is_finished():
            g.run()

      if all(g.is_finished() for g in self._games):
        [g.kill_all() for g in self._games]
        print('====== Finished round {} in {} sec ======'.format(
            self.round, time.time() - round_time))
        fitnesses = [p.get_fitness() for p in self._players]
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
