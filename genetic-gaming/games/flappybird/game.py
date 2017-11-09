import pygame
import sys
import random
import os
import argparse
import msgpackrpc
import time
import pprint
from pygame.locals import *
from PIL import Image
import numpy as np


class Bird(object):
  def __init__(self, shape, jump_time, jump_speed, jump_speed_decline,
               gravity, gravity_acceleration):
    # Static
    self._shape = shape
    self._jump_time = jump_time
    self._jump_speed = jump_speed
    self._jump_speed_decline = jump_speed_decline
    self._gravity = gravity
    self._gravity_acceleration = gravity_acceleration
    # Dynamic
    self.rect = pygame.Rect(shape)
    self.jump_time = 0
    self.gravity = 0
    self.jump_speed = 0
    self.dead = False
    self.fitness = 0

  def trigger_jump(self):
    self.jump_speed = self._jump_speed
    self.gravity = self._gravity
    self.jump_time = self._jump_time

  def is_jumping(self):
    return self.jump_time > 0

  def is_dead(self):
    return self.dead

  def move(self):
    if self.is_jumping():
      self.jump_speed -= self._jump_speed_decline
      self.rect.y -= self.jump_speed
      self.jump_time -= 1
    else:
      self.rect.y += self.gravity
      self.gravity += self._gravity_acceleration

  def reset(self):
    self.rect = pygame.Rect(shape)
    self.jump_time = 0
    self.gravity = 0
    self.jump_speed = 0
    self.dead = False
    self.fitness = 0


class Game(object):
  def __init__(self, args, simulator=None):
    # Static
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_BIRDS = args['num_networks']
    self.SIMULATOR = simulator
    self.SEND_PIXELS = args['send_pixels']
    self.SCREEN_WIDTH = 400
    self.SCREEN_HEIGHT = 708
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
    self.STEPPING = args['stepping']
    self.SCREEN_RESIZE_SHAPE = args['screen_resize_shape']
    if 'assets' not in args:
      args['assets'] = 'assets'
    # Pygame
    self.screen = pygame.display.set_mode(
      (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    asset_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(asset_path, args['assets'])
    asset_background_path = os.path.join(asset_path, 'background.png')
    asset_1_path = os.path.join(asset_path, '1.png')
    asset_2_path = os.path.join(asset_path, '2.png')
    asset_dead_path = os.path.join(asset_path, 'dead.png')
    asset_bottom_path = os.path.join(asset_path, 'bottom.png')
    asset_top_path = os.path.join(asset_path, 'top.png')

    self.background = pygame.image.load(asset_background_path).convert()
    self.birdSprites = [pygame.image.load(asset_1_path).convert_alpha(),
                        pygame.image.load(asset_2_path).convert_alpha(),
                        pygame.image.load(asset_dead_path)]
    self.bottom_pipe = pygame.image.load(asset_bottom_path).convert_alpha()
    self.top_pipe = pygame.image.load(asset_top_path).convert_alpha()
    # Dynamic
    self.round = 0
    self.reset()
    # If -stepping
    self.step = 0
    # RPC proxy to machine learning agent
    self.client = msgpackrpc.Client(
      msgpackrpc.Address(self.ML_AGENT_HOST, self.ML_AGENT_PORT))

  def random_bird_shape(self):
    half_h = self.BIRD_HEIGHT // 2
    random_y = random.randint(half_h, self.SCREEN_HEIGHT - half_h)
    return (self.BIRD_X, random_y, self.BIRD_WIDTH, self.BIRD_HEIGHT)

  def reset(self):
    self.current_wall_x = self.WALLX
    self.counter = 0
    self.offset = random.randint(-110, 110)
    self.travelled_distance = 0
    self.birds = [Bird(self.random_bird_shape(), self.JUMP_TIME,
                       self.JUMP_SPEED, self.JUMP_SPEED_DECLINE,
                       self.GRAVITY, self.GRAVITY_ACCELERATION)
                  for x in range(self.NUM_BIRDS)]
    self.birds_alive = self.NUM_BIRDS
    self.step = 0

  def reset_screen(self):
    self.current_wall_x = self.WALLX
    self.counter = 0
    self.offset = random.randint(-110, 110)
    self.travelled_distance = 0

  def updateWalls(self):
    self.travelled_distance += 2
    self.current_wall_x -= 2
    if self.current_wall_x < -80:
      self.current_wall_x = self.WALLX
      self.counter += 1
      self.offset = random.randint(-110, 110)

  def get_gap_center(self):
    return (0 - self.GAP - self.offset +
            self.top_pipe.get_height() + self.GAP // 2)

  def get_bird_center(self, bird):
    return bird.rect.y + bird.rect.h // 2

  def get_distance_to_gap_center(self, bird):
    return self.get_gap_center() - self.get_bird_center(bird)

  def get_distance_to_wall(self, bird):
    return self.current_wall_x + self.bottom_pipe.get_width() - bird.rect.x

  def get_top_pipe_y(self):
    return 0 - self.GAP - self.offset

  def get_bottom_pipe_y(self):
    return 360 + self.GAP - self.offset

  def build_features(self):
    features = []
    if self.STEPPING:
      if self.SEND_PIXELS:
        features = [self.get_screen_pixels().tolist()]
      else:
        bird = self.birds[self.step]
        distance_to_wall = self.get_distance_to_wall(bird)
        distance_to_gap_center = self.get_distance_to_gap_center(bird)
        features = [[distance_to_wall, distance_to_gap_center]]
    else:
      if self.SEND_PIXELS:
        features = [[self.get_screen_pixels().tolist()]] * self.NUM_BIRDS
      else:
        for bird in self.birds:
          distance_to_wall = self.get_distance_to_wall(bird)
          distance_to_gap_center = self.get_distance_to_gap_center(bird)
          features.append([[distance_to_wall, distance_to_gap_center]])
    return features

  def predict(self):
    features = self.build_features()
    step = self.step if self.STEPPING else None
    if self.SIMULATOR:
      do_jump = self.SIMULATOR.predict(features, step)
    else:
      do_jump = self.client.call('predict', features, step)
    if do_jump is False:
      print('[Error] Prediction failed')
      sys.exit()
    return do_jump

  def get_birds(self):
    return [self.birds[self.step]] if self.STEPPING else self.birds

  def calculate_current_fitness(self, bird):
    return (self.travelled_distance - self.get_distance_to_wall(bird) -
            self.get_distance_to_gap_center(bird))

  def birdUpdate(self):
    # Pipe bounding boxes
    bottom_pipe_bbox = pygame.Rect(self.current_wall_x,
                                   self.get_bottom_pipe_y() + 10,
                                   self.bottom_pipe.get_width() - 10,
                                   self.bottom_pipe.get_height())
    top_pipe_bbox = pygame.Rect(self.current_wall_x,
                                self.get_top_pipe_y() - 10,
                                self.top_pipe.get_width() - 10,
                                self.top_pipe.get_height())
    # Bird moves & collision
    do_jump = self.predict()
    birds = self.get_birds()
    for do_jump, bird in zip(do_jump, birds):
      if not bird.is_dead():
        # Move bird
        if do_jump[0] > 0.5:
          bird.trigger_jump()
        bird.move()
        # Check collision
        if (bottom_pipe_bbox.colliderect(bird.rect) or
              top_pipe_bbox.colliderect(bird.rect) or
              not 0 < bird.rect.y < self.SCREEN_HEIGHT):
          bird.dead = True
          bird.fitness = self.calculate_current_fitness(bird)
          self.birds_alive -= 1
      # Animation of bird falling offscreen
      elif bird.rect.y < self.SCREEN_HEIGHT + 10:
        bird.move()

  def all_birds_dead(self):
    return all([bird.is_dead() for bird in self.birds])

  def run(self):
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 50)
    # self.client.call('reset')
    while True:
      clock.tick(60)
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()

      self.screen.fill((255, 255, 255))
      # Draw background
      self.screen.blit(self.background, (0, 0))
      # Draw bottom pipe
      self.screen.blit(self.bottom_pipe,
                       (self.current_wall_x, self.get_bottom_pipe_y()))
      # Draw top pipe
      self.screen.blit(self.top_pipe,
                       (self.current_wall_x, self.get_top_pipe_y()))
      # Draw score
      self.screen.blit(font.render(
        str('Round: {}'.format(self.round)),
        -1,
        (255, 255, 255)),
        (200, 5))
      self.screen.blit(font.render(
        str('Score: {}'.format(self.counter)),
        -1,
        (255, 255, 255)),
        (200, 50))
      self.screen.blit(font.render(
        str('Alive: {}'.format(self.birds_alive)),
        -1,
        (255, 255, 255)),
        (200, 100))
      if self.all_birds_dead():
        fitnesses = [bird.fitness for bird in self.birds]
        self.reset()
        pprint.pprint(fitnesses)
        if self.SIMULATOR:
          self.SIMULATOR.evolve(fitnesses)
        else:
          self.client.call('evolve', fitnesses)
        self.round += 1
      else:
        birds = self.get_birds()
        for bird in birds:
          if bird.is_dead():
            sprite = self.birdSprites[2]
            self.step += 1
            if self.STEPPING:
              self.reset_screen()
          elif bird.is_jumping():
            sprite = self.birdSprites[1]
          else:
            sprite = self.birdSprites[0]
          self.screen.blit(sprite, (self.BIRD_X, bird.rect.y))
      self.updateWalls()
      self.birdUpdate()
      pygame.display.update()

  def get_screen_pixels(self):
    screen = pygame.surfarray.array3d(self.screen)
    pil_img = Image.fromarray(screen)
    greyscale = pil_img.convert('L')
    # # Screen is somehow mirrored and turned on the side
    # mirrored = greyscale.transpose(Image.FLIP_LEFT_RIGHT)
    # rotated = mirrored.transpose(Image.ROTATE_90)
    # return rotated
    if self.SCREEN_RESIZE_SHAPE:
      greyscale.thumbnail((self.SCREEN_RESIZE_SHAPE, self.SCREEN_RESIZE_SHAPE))
    return np.reshape(np.asarray(greyscale), -1)


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
