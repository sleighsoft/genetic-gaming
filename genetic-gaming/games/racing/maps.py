import uuid
import random
from . import constants
from pymunk import Vec2d


class MapGenerator(object):
  def __init__(self, min_width, max_width, min_length, max_length, game_height,
               game_width, max_angle, min_angle=0, start_point=None,
               start_angle=45, start_width=300, seed=None, max_tries=10):
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
    center = Vec2d((left_start.x + right_start.x) / 2,
                   (left_start.y + right_start.y) / 2)
    length = self.random.uniform(self._min_length, self._max_length)
    angle = self.random.uniform(self._min_angle, self._max_angle)
    angle = self.random.choice([last_angle + angle, last_angle - angle])
    width = self.random.uniform(self._min_width, self._max_width)
    target_center = Vec2d.unit()
    target_center.angle = angle
    target_center.length = length
    target_center = target_center + center

    left_end = Vec2d.unit()
    left_end.angle = angle - constants.PI_05
    left_end.length = width / 2

    right_end = Vec2d.unit()
    right_end.angle = angle + constants.PI_05
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
    left_end.angle = self._start_angle - constants.PI_05
    left_end.length = self._start_width / 2

    right_end = Vec2d.unit()
    right_end.angle = self._start_angle + constants.PI_05
    right_end.length = self._start_width / 2

    return self.zero_border_vector(self._start_point + left_end), \
        self.zero_border_vector(self._start_point + right_end)

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
      next_left, next_right, angle, center = self.get_next_endings(
          last_left, last_right, last_angle)

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
