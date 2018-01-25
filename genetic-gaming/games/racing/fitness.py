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
  def __init__(self, game, conf):
    self._game = game
    self._conf = conf

  def __call__(self, car):
    raise NotImplementedError()


class BasicDistanceToStartCalculator(FitnessCalculator):
  def __call__(self, car):
    return (car.car_body.position - self._game.centers[0]).length


class BasicDistanceToEndCalculator(FitnessCalculator):
  def __call__(self, car):
    return -(car.car_body.position - self._game.centers[-1]).length


class BasicTimeCalculator(FitnessCalculator):
  def __call__(self, car):
    return time.time() - self._game.start_time


class BasicFramesCalculator(FitnessCalculator):
  def __call__(self, car):
    return self._game.frames


class BasicFastestCalculator(FitnessCalculator):
  def __call__(self, car):
    return max(car.velocities)


class BasicFastestAverageCalculator(FitnessCalculator):
  def __call__(self, car):
    return sum(car.velocities) / len(car.velocities)


class BasicCloseToPathCalculator(FitnessCalculator):
  def __init__(self, game, conf):
    super().__init__(game, conf)
    self.tracker = game.tracker

  def __call__(self, car):
    return -(sum(self.tracker.distances[car]) /
             len(self.tracker.distances[car]))


class BasicPathDistanceCalculator(FitnessCalculator):
  def __init__(self, game, conf):
    super().__init__(game, conf)
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


class BasicPathDistanceEndCalculator(BasicPathDistanceCalculator):
  def calculate_distances(self):
    results = [0]
    distance = 0
    last = self._game.centers[-1]

    for center in reversed(self._game.centers[:-1]):
      distance += (last - center).length
      results.insert(0, distance)
      last = center

    return results

  def __call__(self, car):
    last, next = find_closest(self._centers, car.car_body.position)

    if last < next:
      last = next

    dist = self._distances[last]

    return -((car.car_body.position - self._game.centers[last]).length + dist)


class CompositeCalculator(FitnessCalculator):
  def __init__(self, game, conf):
    super().__init__(game, conf)
    self._calcs = [FITNESS_CALCULATORS.get(c['func'])(game, c.get('params', None)) for c in self._conf]

  def __call__(self, car):
    return sum([calc(car) * self._conf[i]['weight'] for i, calc in enumerate(self._calcs)])


class CompositeDivisionCalculator(CompositeCalculator):
  def __call__(self, car):
    return (self._calcs[0](car) * self._conf[0]['weight']) / (self._calcs[1](car) * self._conf[1]['weight'])


class CompositeMultiplicationCalculator(CompositeCalculator):
  def __call__(self, car):
    return (self._calcs[0](car) * self._conf[0]['weight']) / (self._calcs[1](car) * self._conf[1]['weight'])


class CompositeFastestPathCalculator(BasicPathDistanceCalculator):
  def __call__(self, car):
    return super().__call__(car) / (time.time() - self._game.start_time)


FITNESS_CALCULATORS = {
    'distance_to_start': BasicDistanceToStartCalculator,
    'distance_to_end': BasicDistanceToEndCalculator,
    'time': BasicTimeCalculator,
    'frames': BasicFramesCalculator,
    'path': BasicPathDistanceCalculator,
    'path_end': BasicPathDistanceEndCalculator,
    'fastest': BasicFastestCalculator,
    'fastest_average': BasicFastestAverageCalculator,
    'close_to_path': BasicCloseToPathCalculator,
    'composite': CompositeCalculator,
    'divide': CompositeDivisionCalculator,
    'multiply': CompositeMultiplicationCalculator
}
