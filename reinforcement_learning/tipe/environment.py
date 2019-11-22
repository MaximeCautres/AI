import random
import pygame
from PIL import Image, ImageDraw


class Infix:

    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


class Vec2:

    def __init__(self, x, y):
        self.coord = (x, y)
        self.x = x
        self.y = y

    def sum(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def dif(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def scale_prod(self, k):
        return Vec2(self.x * k, self.y * k)

    def dot_prod(self, v):
        return self.x * v.x + self.y * v.y

    def norm(self):
        return self.dot_prod(self)**(1/2)

    def dist(self, v):
        return self.dif(v).norm()

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


p = Infix(lambda u, v: u.sum(v))
m = Infix(lambda u, v: u.dif(v))
s = Infix(lambda u, k: u.scale_prod(k))
d = Infix(lambda u, v: u.dist(v))


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
        self.p = p0
        self.v = v0

    def apply_action(self, a):
        np = self.p | p | self.v | p | a
        collided = False
        for obstacle in obstacles:
            collided = collided or obstacle.collide(np)
        if not collided:
            self.v = np | m | self.p
            self.p = np
            return 6


def pil_pyg(v):
    img = background.copy()
    img_draw = ImageDraw.Draw(img)
    img_draw.rectangle([v.x, v.y] * 2, fill='blue')
    pil_img = img.resize(img_size)
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()
    return pygame.image.fromstring(data, size, mode)


unit = 36
dims = Vec2(24, 24)
img_size = (dims | s | unit).coord
obstacles = [Rectangle(Vec2(0, dims.y-1), Vec2(dims.x-1, 0), False)]
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
billy = Boat(Vec2(dims.x // 2, dims.y // 2), Vec2(0, 0))

background = Image.new('RGB', dims.coord, color='black')
bg_draw = ImageDraw.Draw(background)

for obs in obstacles:
    if obs.convex:
        (x0, y0), (x1, y1) = obs.tlc.coord, obs.brc.coord
        bg_draw.rectangle([x0, y0, x1, y1], fill='red')

pygame.init()
screen = pygame.display.set_mode(img_size)
inGame = True

while inGame:
    screen.blit(pil_pyg(billy.p), (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            action = random.choice(actions)
            print(action)
            res = billy.apply_action(action)
            if not res:
                inGame = False
    pygame.display.flip()

pygame.quit()
