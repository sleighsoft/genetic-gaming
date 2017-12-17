import time
import numpy as np
from numpy.linalg import norm


def find_closest(points, pos, amount=2):
  dist = np.sum((points - (pos.x, pos.y))**2, axis=1)
  return tuple(np.argsort(dist)[:amount])


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
    return (norm(np.cross(next - last, last - car.car_body.position)) /
            norm(next - last))

  def calculate_distances(self):
    self.pause_counter -= 1
    if self.pause_counter <= 0:
      for car in self.cars:
        self.distances[car].append(self.calc_distance(car))
      self.pause_counter = self.init_pause_calls


class FitnessCalculator(object):
  def __init__(self, game, **kwargs):
    self._game = game
    self._conf = kwargs

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
    return sum(car.velocities) / len(car.velocities)


class CloseToPathCalculator(FitnessCalculator):
  def __init__(self, game, **kwargs):
    super().__init__(game, **kwargs)
    self.tracker = game.tracker

  def __call__(self, car):
    return -(sum(self.tracker.distances[car]) /
             len(self.tracker.distances[car]))


class PathDistanceCalculator(FitnessCalculator):
  def __init__(self, game, **kwargs):
    super().__init__(game, **kwargs)
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

    return (car.car_body.position - self._game.centers[last]).length + dist


class MixedCalculator(FitnessCalculator):
  def __init__(self, game, **kwargs):
    super().__init__(game, **kwargs)
    self._calc_a = FITNESS_CALCULATORS.get(self._conf['func_a'])(game, **kwargs)
    self._calc_b = FITNESS_CALCULATORS.get(self._conf['func_b'])(game, **kwargs)

  def __call__(self, car):
    return (self._calc_a(car) * self._conf['weight_a']) + (self._calc_b(car) * self._conf['weight_b'])


class FastestPathCalculator(PathDistanceCalculator):
  def __call__(self, car):
    return super().__call__(car) / (time.time() - self._game.start_time)


FITNESS_CALCULATORS = {
    'distance_to_start': DistanceToStartCalculator,
    'distance_to_end': DistanceToEndCalculator,
    'time': TimeCalculator,
    'path': PathDistanceCalculator,
    'fastest': FastestCalculator,
    'fastest_average': FastestAverageCalculator,
    'close_to_path': CloseToPathCalculator,
    'mixed': MixedCalculator,
    'fastest_path': FastestPathCalculator
}
