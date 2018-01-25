import uuid
import random
from . import constants
from pymunk import Vec2d


def ccw(a, b, c):
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def find_intersection(p0, p1, p2, p3):

    s10_x = p1.x - p0.x
    s10_y = p1.y - p0.y
    s32_x = p3.x - p2.x
    s32_y = p3.y - p2.y

    denom = s10_x * s32_y - s32_x * s10_y

    if denom == 0:
      return None  # collinear

    denom_is_positive = denom > 0

    s02_x = p0.x - p2.x
    s02_y = p0.y - p2.y

    s_numer = s10_x * s02_y - s10_y * s02_x

    if (s_numer < 0) == denom_is_positive:
      return None  # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x

    if (t_numer < 0) == denom_is_positive:
      return None  # no collision

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive:
      return None  # no collision

    # collision detected
    t = t_numer / denom

    intersection_point = Vec2d(p0.x + (t * s10_x), p0.y + (t * s10_y))

    return intersection_point


# Return true if line segments AB and CD intersect
def intersect(p0, p1, p2, p3):
  intersect_point = find_intersection(p0, p1, p2, p3)
  return intersect_point is not None and intersect_point not in [p0, p1, p2, p3]


class MapGenerator(object):
  def __init__(self, min_width, max_width, min_length, max_length, game_height,
               game_width, max_angle, seed, min_angle=0, start_point=None,
               start_angle=45, start_width=300,  max_tries=10):
    random.seed(seed)
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
    length = random.uniform(self._min_length, self._max_length)
    angle = random.uniform(self._min_angle, self._max_angle)
    angle = random.choice([last_angle + angle, last_angle - angle])
    width = random.uniform(self._min_width, self._max_width)
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
    return (0 < point.x < self._game_width and 0 < point.y < self._game_height)

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

  def is_valid_line(self, a, b, others):
    return all(not intersect(a, b, w['start'], w['end']) for w in others)

  def generate_first_segment(self, last_left, last_right, center):
      add_vec = Vec2d(100, 0)
      return last_left + add_vec, last_right + add_vec, center + add_vec

  def generate(self):
    last_left, last_right = self.get_start_points()
    last_angle = self._start_angle
    tries_left = self._max_tries

    next_left, next_right, center = self.generate_first_segment(last_left, last_right, Vec2d(self._start_point))
    found = [self.get_wall(last_left, last_right),
             self.get_wall(last_left, next_left), self.get_wall(last_right, next_right)]
    last_left = next_left
    last_right = next_right
    centers = [Vec2d(self._start_point), center]
    while tries_left > 0:
      next_left, next_right, angle, center = self.get_next_endings(
          last_left, last_right, last_angle)

      if self.is_valid(next_left) and self.is_valid(next_right) and \
         self.is_valid_line(last_left, next_left, found) and self.is_valid_line(last_right, next_right, found):
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
