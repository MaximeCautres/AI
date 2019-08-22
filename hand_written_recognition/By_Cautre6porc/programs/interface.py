""" Use Tkinter to draw number """

import math
from tkinter import *
import neural_network as nn

" Restore or create a neural network "

net = nn.restore('96.37.txt')
# net = nn.Network([784, 10])

" Train the neural network "

# net.training(6, 20, 1)

" Set law about reading "

impact_x = 1
impact_y = 1.2


class Text:

    """ Initialize a new Text area """

    def __init__(self, root, side, text):

        self.label = Label(root, fg='blue', text=text, font='Verdana 20 bold')
        self.label.pack(side=side, padx=20, pady=5)

    """ Update the text of this Text object """

    def update(self, text):

        self.label.config(text=text)


class Drawing:

    """ Initialize a new drawing area """

    def __init__(self, root, w, h):

        self.r = 6

        self.w = w
        self.h = h

        self.matrix = [[False for _ in range(w)] for _ in range(h)]
        self.save = []

        self.numbers = []
        self.state = True

        self.canvas = Canvas(root, width=w, height=h, background='black')
        self.canvas.pack(side=BOTTOM, padx=1, pady=1)

        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.read)
        self.canvas.bind('<B3-Motion>', self.clean)
        self.canvas.bind('<ButtonRelease-3>', self.clear)
        self.canvas.bind('<MouseWheel>', self.wheel)

    """ This method is call whenever the left click is pressed and moving to draw numbers """

    def draw(self, event):

        y, x = event.y, event.x
        r = self.r

        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', width=0)

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if r ** 2 >= i ** 2 + j ** 2:
                    dy, dx = self.confine(y + i, x + j)
                    self.matrix[dy][dx] = True

    """ This method is call whenever this mouse is wheeled to change the font size """

    def wheel(self, event):

        pr = self.r
        self.r = max(min(self.r + int(event.delta / 120), 16), 2)
        if self.r != pr:
            font_size_text.update('Font size : ' + str(self.r))

    """ This method is call whenever the right click is pressed and moving to clean numbers """

    def clean(self, event):

        y, x = event.y, event.x
        r = self.r + 16
        self.state = False

        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', width=0)

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if r ** 2 >= i ** 2 + j ** 2:
                    dy, dx = self.confine(y + i, x + j)
                    self.matrix[dy][dx] = False

    """ This method is call whenever the right click is released to clear numbers """

    def clear(self, event=None):

        if self.state:
            self.canvas.delete("all")
            self.matrix = [[False for _ in range(self.w)] for _ in range(self.h)]

        self.state = True
        self.read()

    """ This method is call whenever the left click is released to return a guess of what it's written """

    def read(self, event=None):

        self.save = [x[:] for x in self.matrix]

        for i in range(self.h):
            for j in range(self.w):
                if self.matrix[i][j]:
                    self.create_number(i, j).analyze()

        length = len(self.numbers)
        string = 'Guess : '

        if not length:
            string += 'None'

        for _ in range(length):
            best_score = self.w ** impact_x + self.h ** impact_y + 1
            best = self.numbers[0]
            for x in self.numbers:
                if x.score < best_score:
                    best_score = x.score
                    best = x
            string += str(best.guess)
            self.numbers.remove(best)

        result_text.update(string)

        self.numbers = []
        self.matrix = [x[:] for x in self.save]

    """ Return a new Number object and clear it in the matrix """

    def create_number(self, y, x):

        num = Number(self.w, self.h)
        self.numbers.append(num)

        stack = [(y, x)]
        num.matrix[y][x] = True
        self.matrix[y][x] = False

        while stack:

            y, x = stack[0]
            del stack[0]

            for i in range(-1, 2):
                for j in range(-1, 2):
                    dy, dx = self.confine(y + i, x + j)
                    if i * j == 0 and self.matrix[dy][dx] and not num.matrix[dy][dx]:
                        stack.append((dy, dx))
                        num.matrix[dy][dx] = True
                        self.matrix[dy][dx] = False

        return num

    """ Return dy and dx which are the confined version of y and x between 0 and self.h and self.w """

    def confine(self, y, x):

        return max(min(y, self.h - 1), 0), max(min(x, self.w - 1), 0)


class Number:

    """ Initialize a new Number object """

    def __init__(self, w, h):

        self.w = w
        self.h = h

        self.matrix = [[False for _ in range(w)] for _ in range(h)]

        self.guess = 0
        self.score = 0

    """ Analyze this Number object to guess what number it is """

    def analyze(self):

        x_min, x_max = self.w, 0
        y_min, y_max = self.h, 0

        for i in range(self.h):
            for j in range(self.w):
                if self.matrix[i][j]:
                    if j < x_min:
                        x_min = j
                    if j > x_max:
                        x_max = j
                    if i < y_min:
                        y_min = i
                    if i > y_max:
                        y_max = i

        mx = (x_min + x_max) * 0.5
        my = (y_min + y_max) * 0.5

        self.score = mx ** impact_x + my ** impact_y

        side = int(math.ceil(max(x_max - x_min, y_max - y_min) / 20))

        x = x_min - int((side * 20 - x_max + x_min) * 0.5)
        y = y_min - int((side * 20 - y_max + y_min) * 0.5)

        square20 = [[0 for y in range(20)] for x in range(20)]

        for j_res in range(20):
            for i_res in range(20):
                stack = 0
                for j in range(side):
                    for i in range(side):
                        dx = x + side * i_res + i
                        dy = y + side * j_res + j
                        if 0 <= dx < self.w and 0 <= dy < self.h:
                            stack += self.matrix[dy][dx]
                square20[j_res][i_res] = min(round(stack / (side * side), 1) * 1.2, 1)

        row = []
        col = []

        for x in range(20):
            stack_row = 0
            stack_col = 0
            for y in range(20):
                stack_row += square20[x][y]
                stack_col += square20[y][x]
            row.append(stack_row)
            col.append(stack_col)

        pond_row = 0
        pond_col = 0

        for i in range(20):
            pond_row += i * row[i]
            pond_col += i * col[i]

        average_row = max(min(pond_row / (sum(row) + 0.01), 14), 6)
        average_col = max(min(pond_col / (sum(col) + 0.01), 14), 6)

        lc = round(13.5 - average_row)
        hc = round(13.5 - average_col)

        result = [[0 for _ in range(28)] for _ in range(28)]

        for x in range(20):
            for y in range(20):
                result[lc + x][hc + y] = square20[x][y]

        request = []
        for x in result:
            request += x

        self.guess = net.generate(request)


" Manage the main window "

window = Tk()
window.title('Handwritten digit recognition')

drawingZone = Drawing(window, 800, 500)
result_text = Text(window, LEFT, 'Guess : None')
font_size_text = Text(window, RIGHT, 'Font size : 6')

window.mainloop()
