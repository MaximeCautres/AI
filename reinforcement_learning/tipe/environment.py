import random

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
d = Infix(lambda u, v: u.dist(v))
m = Infix(lambda u, v: u.dif(v))


class Rectangle:

    def __init__(self, tlc, brc, convex):
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

    def apply_action(self, action):
        np = self.p | p | self.v | p | action
        collided = False
        for obstacle in obstacles:
            collided = collided or obstacle.collide(np)
        if not collided:
            self.v = np | m | self.p
            self.p = np

        else:
            exit("error 42")


obstacles = [Rectangle(Vec2(0, 20), Vec2(20, 0), False)]
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
billy = Boat(Vec2(10, 10), Vec2(0, 0))

while True:
    action = random.choice(actions)
    for i in range(3):
        billy.apply_action(action)
        print(billy.p)