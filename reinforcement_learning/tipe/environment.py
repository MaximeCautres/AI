import random
from tkinter import *
from PIL import Image, ImageTk, ImageDraw


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

    def __init__(self, tlc, brc, convex = True):
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
        else:
            print(42)
            exit()


class Frame:

    def __init__(self, bg):

        self.bg = ImageTk.PhotoImage(bg)
        self.dim = (bg.width // 2, bg.height // 2)

        self.canvas = Canvas(window, width=self.dim[0], height=self.dim[1], background='green')
        self.canvas.pack(side=LEFT, padx=1, pady=1)

        self.canvas.create_image(0, 0, image=self.bg)


unit = 56
height, width = 24, 24
obstacles = [Rectangle(Vec2(1, height), Vec2(width, 1), False),
             Rectangle(Vec2(5, 20), Vec2(10, 7))]
actions = [Vec2(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
billy = Boat(Vec2(width // 2, height // 2), Vec2(0, 0))

background = Image.new('RGB', (height, width), color='red')
draw = ImageDraw.Draw(background)

for obs in obstacles:
    if obs.convex:
        tlc, brc = obs.tlc, obs.brc
        draw.rectangle([tlc, brc], fill='white')

background.resize((height * unit, width * unit))

window = Tk()
window.title('Title')

frame = Frame(background)

window.mainloop()

while True:
    action = random.choice(actions)
    for i in range(3):
        billy.apply_action(action)
        print(billy.p, billy.v)
