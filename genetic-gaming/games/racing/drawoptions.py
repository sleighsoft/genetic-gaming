import pymunk
import pygame
from pymunk import Vec2d


positive_y_is_up = False


class OffsetDrawOptions(pymunk.SpaceDebugDrawOptions):

    def __init__(self, surface):
        self.surface = surface
        super().__init__()
        self.offset = Vec2d(0, 0)

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        p = to_pygame(pos, self.surface) + self.offset

        pygame.draw.circle(self.surface, fill_color, p, int(radius), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        pygame.draw.lines(self.surface, outline_color, False, [p, p2], line_r)

    def draw_segment(self, a, b, color):
        p1 = to_pygame(a, self.surface) + self.offset
        p2 = to_pygame(b, self.surface) + self.offset

        pygame.draw.aalines(self.surface, color, False, [p1, p2])

    def draw_fat_segment(self, a, b, radius, outline_color, fill_color):
        p1 = to_pygame(a, self.surface) + self.offset
        p2 = to_pygame(b, self.surface) + self.offset

        r = int(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color, False, [p1, p2], r)

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [to_pygame(v, self.surface) + self.offset for v in verts]
        ps += [ps[0]]

        pygame.draw.polygon(self.surface, fill_color, ps)

        if radius < 1 and False:
            pygame.draw.lines(self.surface, outline_color, False, ps)
        else:
            pygame.draw.lines(self.surface, outline_color, False, ps, int(radius * 2))


def draw_dot(self, size, pos, color):
    p = to_pygame(pos, self.surface)
    pygame.draw.circle(self.surface, color, p, int(size), 0)


def get_mouse_pos(surface):
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p, surface):
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return int(p[0]), surface.get_height() - int(p[1])
    else:
        return int(p[0]), int(p[1])


def from_pygame(p, surface):
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)
