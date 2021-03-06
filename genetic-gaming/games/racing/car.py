import math
import pygame
import pymunk
from pymunk import Vec2d


class Car(object):

  def __init__(self, shape, position, rotation, rotation_speed, base_velocity,
               acceleration, deceleration, acceleration_time, min_velocity,
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
    self._min_velocity = min_velocity
    self._max_velocity = max_velocity
    self._color = color
    self._sensor_range = sensor_range
    self._num_sensors = num_sensors
    self._sensor_color = sensor_color
    self._offset = Vec2d(0, 0)

    inertia = pymunk.moment_for_box(1, self._shape)
    self.car_body = pymunk.Body(1, inertia)
    self.car_shape = pymunk.Poly.create_box(self.car_body, self._shape)
    self.car_shape.filter = pymunk.ShapeFilter(categories=0x2)
    self.velocities = []

    # Dynamic
    self.reset()

  def reset(self, new_position=None):
    """Reset car to initial settings."""
    # Pymunk
    if new_position:
      self._position = new_position
    self.car_body.position = self._position
    self.car_shape.color = self._color
    self.car_shape.elasticity = 1.0
    self.car_shape.sensor = True
    self.car_shape.collision_type = 1
    self.car_shape.car = self
    self.car_body.angle = self._rotation
    self.car_body.velocity = Vec2d(0, 0)
    driving_direction = Vec2d(0, 0).rotated(self.car_body.angle)
    self.car_body.apply_impulse_at_world_point(driving_direction)

    # Dynamic
    self._velocity = 0
    self.rotation = self._rotation
    self.current_acceleration_time = 0
    self.is_dead = False
    self.fitness = 0.0
    self.previous_position = self._position
    # Last car movement
    self.last_turn = 0.0
    self.last_acceleration = 0.0

  def update_offset(self, offset):
    self._offset = offset

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

    def rotate_point(length, radians):
      x = length * math.sin(radians)
      y = length * math.cos(radians)
      return x, y

    sensor_directions = []
    for i in range(self._num_sensors):
      sensor_directions.append(
          start + rotate_point(self._sensor_range, i * math.pi /
                               (self._num_sensors - 1)))

    for sensor_direction in sensor_directions:
      rotation = self.car_body.angle
      rotated_end = Car.get_rotated_point(
          s_x, s_y, sensor_direction[0], sensor_direction[1], rotation)
      sensors.append((start, rotated_end))

    return sensors

  def show_sensors(self, screen, points_of_impact):
    for i, sensor in enumerate(self.get_sensors()):
      end = sensor[1] if points_of_impact[i] is None else points_of_impact[i]
      pygame.draw.line(screen, self._sensor_color, sensor[0] + self._offset,
                       end + self._offset)

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
      normalized_distance = min(max(min_distance / self._sensor_range, 0), 1.0)
      distances.append(normalized_distance)
      points_of_impact.append(end)

    if screen:
      self.show_sensors(screen, points_of_impact)

    return distances

  def trigger_rotate_left(self):
    self.rotation -= self._rotation_speed

  def trigger_rotate_right(self):
    self.rotation += self._rotation_speed

  def trigger_rotation(self, value):
    self.rotation += self._rotation_speed * value

  def trigger_acceleration(self, value):
    actual_velocity = self._velocity + (self._acceleration * value)
    self._velocity = max(self._min_velocity, min(
        self._max_velocity, actual_velocity))

  def move(self):
    """Perform all triggered movements."""
    driving_direction = Vec2d(1, 0).rotated(self.rotation)
    self.car_body.angle = self.rotation
    self.car_body.velocity = self._velocity * driving_direction
    self.velocities.append(self._velocity)

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

  @property
  def num_sensors(self):
    return self._num_sensors

  @property
  def velocity(self):
    return self.car_body.velocity
