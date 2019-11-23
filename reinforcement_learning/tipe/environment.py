import random
import pygame
import numpy as np


class Vec2:

    def __init__(self, x=0, y=0):
        self.coord = (x, y)
        self.x = x
        self.y = y

    def plus(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def minus(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def scale_prod(self, k):
        return Vec2(self.x * k, self.y * k)

    def dot_prod(self, v):
        return self.x * v.x + self.y * v.y

    def norm(self):
        return self.dot_prod(self)**(1/2)

    def dist(self, v):
        return self.minus(v).norm()

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Rectangle:

    def __init__(self, tlc, brc, convex=True):
        self.tlc = tlc
        self.brc = brc
        self.convex = convex

    def collide(self, v):
        cond = self.brc.x >= v.x >= self.tlc.x and self.brc.y <= v.y <= self.tlc.y
        return cond and self.convex or not (cond or self.convex)


class Boat:

    def __init__(self, p0, v0):
        self.pos = p0
        self.vel = v0

    def apply_action(self, a):
        new_pos = self.pos.plus(self.vel.plus(a))
        collided = False
        for obstacle in obstacles:
            collided = collided or obstacle.collide(new_pos)
        if not collided:
            self.vel = new_pos.minus(self.pos)
            self.pos = new_pos
            return self.pos.dist(goal)


def create_background():
    bg = np.zeros(dims.coord + (3,), dtype=np.int)
    for i in range(dims.x):
        for j in range(dims.y):
            point = Vec2(i, j)
            if point.dist(goal) < goal_radius:
                bg[i, j] = GREEN
            for obs in obstacles:
                if obs.convex and obs.collide(point):
                    bg[i, j] = RED
    return bg


def current_frame(point):
    bg = background.copy()
    bg[point.x, point.y] = BLUE
    return bg


def numpy_to_pygame(array):
    surface = pygame.surfarray.make_surface(array)
    return pygame.transform.scale(surface, img_size)


RED = np.array([128, 0, 0])
GREEN = np.array([0, 128, 0])
BLUE = np.array([0, 0, 128])

unit = 36
dims = Vec2(24, 24)
img_size = (dims.scale_prod(unit)).coord

obstacles = [Rectangle(Vec2(0, dims.y-1), Vec2(dims.x-1, 0), False),
             Rectangle(Vec2(3, 17), Vec2(5, 15)),
             Rectangle(Vec2(15, 5), Vec2(17, 3))]
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
billy = Boat(Vec2(dims.x // 3, dims.y // 3), Vec2())

goal = Vec2(2 * dims.x // 3, 2 * dims.y // 3)
goal_radius = 6

background = create_background()

pygame.init()
game_state = "in_game"
screen = pygame.display.set_mode(img_size)

while game_state == "in_game":
    current = current_frame(billy.pos)
    screen.blit(numpy_to_pygame(current), (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            action = random.choice(actions)
            print(action)
            res = billy.apply_action(action)
            if res is None:
                game_state = "loose"
            elif res < goal_radius:
                game_state = "win"
    pygame.display.flip()

print(game_state)
pygame.quit()
