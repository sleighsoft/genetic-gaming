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
               acceleration, deceleration, acceleration_time, turn_speed,
               max_velocity, color):
    # Static
    self._shape = shape
    self._position = position
    self._rotation = rotation
    self._rotation_speed = rotation_speed
    self._base_velocity = base_velocity
    self._acceleration = acceleration
    self._deceleration = deceleration
    self._acceleration_time = acceleration_time
    self._turn_speed = turn_speed
    self._max_velocity = max_velocity
    self._color = color

    # Pymunk
    inertia = pymunk.moment_for_box(1, self._shape)
    self.car_body = pymunk.Body(1, inertia)
    self.car_body.position = self._position
    self.car_shape = pymunk.Poly.create_box(self.car_body, self._shape)
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

  def add_to_space(self, space):
    space.add(self.car_body, self.car_shape)

  def rotate_left(self):
    self.rotation += self._rotation_speed

  def rotate_right(self):
    self.rotation -= self._rotation_speed

  def trigger_acceleration(self):
    if self.current_acceleration_time == 0:
      self.velocity = self._base_velocity
    else:
      self.velocity = max(self._max_velocity,
                          self.velocity * self._acceleration)
    self.current_acceleration_time = self._acceleration_time

  def move(self):
    if self.current_acceleration_time > 0:
      self.current_acceleration_time -= 1
    else:
      self.velocity = max(0, self.velocity * self._deceleration)
    driving_direction = Vec2d(1, 0).rotated(self.rotation)
    self.car_body.angle = self.rotation
    self.car_body.velocity = self.velocity * driving_direction

  def reset(self):
    self.car_body.position = self._position
    self.car_shape.color = self._color
    self.car_body.angle = self._rotation
    self.car_body.velocity = Vec2d(0, 0)
    driving_direction = Vec2d(0, 0).rotated(self.car_body.angle)
    self.car_body.apply_impulse_at_world_point(driving_direction)


class Game(object):
  def __init__(self, args, simulator=None):
    # EvolutionServer
    self.ML_AGENT_HOST = args['host']
    self.ML_AGENT_PORT = args['port']
    self.NUM_BIRDS = args['num_networks']
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
    self.space = pymunk.Space()
    self.space.gravity = pymunk.Vec2d(0., 0.)
    self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    # TODO
    # asset_background_path = os.path.join(asset_path, 'background.png')
    # asset_1_path = os.path.join(asset_path, '1.png')
    # asset_2_path = os.path.join(asset_path, '2.png')
    # asset_dead_path = os.path.join(asset_path, 'dead.png')
    # asset_bottom_path = os.path.join(asset_path, 'bottom.png')
    # asset_top_path = os.path.join(asset_path, 'top.png')

    # self.background = pygame.image.load(asset_background_path).convert()
    # self.birdSprites = [pygame.image.load(asset_1_path).convert_alpha(),
    #                     pygame.image.load(asset_2_path).convert_alpha(),
    #                     pygame.image.load(asset_dead_path)]
    # self.bottom_pipe = pygame.image.load(asset_bottom_path).convert_alpha()
    # self.top_pipe = pygame.image.load(asset_top_path).convert_alpha()

    # Dynamic
    self.round = 0
    self.reset()
    # If -stepping
    self.step = 0

    # Setup
    for car in self.cars:
      car.add_to_space(self.space)

  def reset(self):
    """Reset game state."""
    center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
    self.cars = [
      Car(shape=(10, 10),
          position=center,
          rotation=0.0,
          rotation_speed=0.2,
          base_velocity=5.0,
          acceleration=1.1,
          deceleration=0.8,
          acceleration_time=10,
          turn_speed=0.2,
          max_velocity=1.3,
          color=(0, 0, 0))]

  def update_track():
    pass

  def update_cars():
    pass

  def run(self):
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 50)

    # TODO playground
    def get_border(x, y):
      border = pygame.Rect(x, y, 100, 10)
      return border

    border_color = (104, 114, 117)

    while True:
      clock.tick(60)
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_RIGHT:
            self.cars[0].rotate_right()
          if event.key == pygame.K_LEFT:
            self.cars[0].rotate_left()
          if event.key == pygame.K_UP:
            self.cars[0].trigger_acceleration()

      if sum(pygame.key.get_pressed()):
        pressed_key = pygame.key.get_pressed()
        if pressed_key[pygame.K_RIGHT]:
          self.cars[0].rotate_right()
        if pressed_key[pygame.K_LEFT]:
          self.cars[0].rotate_left()
        if pressed_key[pygame.K_UP]:
          self.cars[0].trigger_acceleration()

      self.screen.fill((255, 255, 255))

      # Cars
      self.cars[0].move()

      # Walls
      border1 = get_border(self.SCREEN_WIDTH // 2 - 50,
                           self.SCREEN_HEIGHT // 2 - 50)
      self.screen.fill(border_color, border1)

      # Pymunk
      self.space.debug_draw(self.draw_options)

      # if border1.colliderect(self.cars[0].rect):
      #   self.reset()

      pygame.display.update()

      fps = 60
      dt = 1. / fps
      self.space.step(dt)


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
