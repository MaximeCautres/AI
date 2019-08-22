""" Use Tkinter to draw number """

import math
from tkinter import *
import backpropagation as bp


class Drawing:

    """ Initialize a new drawing area """

    def __init__(self, root, w, h, k):

        self.r = 6

        self.k = k
        self.w = w
        self.h = h

        self.matrix = [[False for _ in range(w)] for _ in range(h)]

        self.canvas = Canvas(root, width=w, height=h, background='black')
        self.canvas.pack()

        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-2>', self.read)
        self.canvas.bind('<Button-3>', self.clear)
        self.canvas.bind('<MouseWheel>', self.wheel)

    """ This method is call whenever the left click is pressed and moving to draw number """

    def draw(self, event):

        x, y = event.x, event.y
        r = self.r

        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', width=0)

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if r ** 2 >= i ** 2 + j ** 2:
                    self.matrix[y + j][x + i] = True

    """ This method is call whenever the middle click is pressed to ask to the neural network an answer """

    def read(self, event=None):

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

        side = int(math.ceil(max(x_max - x_min, y_max - y_min) / 20))

        x = x_min - int((side * 20 - x_max + x_min) * 0.5)
        y = y_min - int((side * 20 - y_max + y_min) * 0.5)

        square20 = [[0 for y in range(20)] for x in range(20)]

        for j_res in range(20):
            for i_res in range(20):
                stack = 0
                for j in range(side):
                    for i in range(side):
                        stack += self.matrix[y + side * j_res + j][x + side * i_res + i]
                square20[j_res][i_res] = min(round(stack / (side * side), 1) * self.k, 1)

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

        displayZone.show(result)

        print('Guess : {0}'.format(net.generate(request)))

    """ This method is call whenever the right click is pressed to clear this drawing area """

    def clear(self, event=None):

        self.canvas.delete('all')
        self.matrix = [[False for _ in range(self.w)] for _ in range(self.h)]

    """ This method is call whenever this mouse is wheeled to change the font size """

    def wheel(self, event):

        pr = self.r
        self.r = max(min(self.r + int(event.delta / 120), 16), 2)
        if self.r != pr:
            print('The font size has been updated to {0}.'.format(self.r))


class Display:

    """ Initialize a new display area """

    def __init__(self, root, w, h):

        self.canvas = Canvas(root, width=w, height=h, background='black')
        self.canvas.pack()

    """ Display on this area a matrix """

    def show(self, matrix):

        r = 16
        data = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        for i in range(28):
            for j in range(28):
                color = '#' + str(data[int(matrix[j][i] * 15)]) * 3
                self.canvas.create_rectangle(i * r, j * r, (i + 1) * r, (j + 1) * r, fill=color, width=0)


" Create a new neural network "
net = bp.Network([784, 30, 10])
net.training(3, 200, 6)

" Restore a save "
# net = bp.restore('94.97.txt')

" Manage the main window "
window = Tk()
drawingZone = Drawing(window, 600, 600, 1.2)
displayZone = Display(window, 448, 448)
window.mainloop()
