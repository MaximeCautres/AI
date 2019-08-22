""" The Game class """

import time
from tkinter import *
import neural_network as nn

bot_a = nn.restore('test_a.txt')
bot_b = nn.restore('test_b.txt')

COUNT = 1

YELLOW = 0
RED = 1

width = 7
height = 6

path = [{'x': 1, 'y': 1}, {'x': 1, 'y': 0}, {'x': 1, 'y': -1}, {'x': 0, 'y': -1}]


class Game:

    def __init__(self):

        self.winner = None
        self.color = None
        self.current = 0
        self.array = [0.5 for _ in range(width * height)]

        self.window = Tk()
        self.window.title('Connect 4')

        self.canvas = Canvas(self.window, width=width * 100, height=height * 100, background='blue')
        self.canvas.pack(side=BOTTOM, padx=1, pady=1)
        self.canvas.bind('<Button-1>', self.play)

        for i in range(width):
            for j in range(height):
                self.canvas.create_oval(i * 100, j * 100, (i + 1) * 100, (j + 1) * 100, fill='white')

        self.window.mainloop()

    def play(self, event=None):

        if self.winner is not None:
            print(self.color + ' win !')
            print('Waiting for closing...')
            time.sleep(1)
            exit()

        if COUNT == 0:
            x = int(event.x / 100)
        elif self.current == 0:
            if COUNT == 2:
                x = bot_a.generate(self.array)
            else:
                x = int(event.x / 100)
        else:
            x = bot_b.generate(self.array)

        y = -1

        while y < height - 1 and self.array[x + 7 * (y + 1)] == 0.5:
            y += 1

        if y != -1:

            for i in range(len(path)):
                dx = path[i]['x']
                dy = path[i]['y']
                score = 0
                for unit in range(-1, 2, 2):
                    vec = unit
                    pos = x + dx * vec + width * (y + dy * vec)
                    while 0 <= pos < width * height and self.array[pos] == self.current:
                        score += 1
                        vec += unit
                        pos = x + dx * vec + width * (y + dy * vec)
                if score == 3:
                    self.winner = self.current

            if self.current == YELLOW:
                self.color = 'yellow'
                self.array[x + 7 * y] = YELLOW
                self.current = RED
            else:
                self.color = 'red'
                self.array[x + 7 * y] = RED
                self.current = YELLOW

            self.canvas.create_oval(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100, fill=self.color)


party = Game()
