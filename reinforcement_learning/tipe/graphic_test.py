from tkinter import *
import random


class Drawing:

    def __init__(self, root, w, h):
        self.w = w
        self.h = h

        self.objects = []
        self.items = ['state', 'action', 'reward', 'link']
        self.current_item = 0

        self.canvas = Canvas(root, width=w, height=h, background='black')
        self.canvas.pack(side=BOTTOM, padx=1, pady=1)

        self.canvas.bind('<Button-1>', self.add_node)
        self.canvas.bind('<Button-3>', self.undo)
        self.canvas.bind('<MouseWheel>', self.wheel)

    def add_node(self, event):
        r = 24
        x, y = event.x, event.y
        c = color_map[self.items[self.current_item]]
        obj = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=c)
        self.objects.append(obj)

    def undo(self, event=None):
        if self.objects:
            obj = self.objects.pop()
            self.canvas.delete(obj)

    def wheel(self, event):
        self.current_item = int((self.current_item + event.delta / 120) % len(self.items))
        item_selected.update(self.items[self.current_item])
        self.change_color(1, random.choice(list(color_map.values())))

    def change_color(self, tag, nc):
        self.canvas.itemconfigure(tag, fill=nc)

    def get_coord(self, tag):
        return self.canvas.coords(tag)

    def move(self, tag, p):
        self.canvas.coords(tag, p)


class Text:

    def __init__(self, root, side, text):
        self.label = Label(root, fg='orange', text=text, font='Verdana 20 bold')
        self.label.pack(side=side, padx=20, pady=5)

    def update(self, text):
        self.label.config(text=text, fg=color_map[text])


color_map = {'state': 'orange', 'action': 'blue', 'reward': 'green', 'link': 'gray'}

window = Tk()
window.title('I like trains !')

drawingZone = Drawing(window, 800, 500)
item_selected = Text(window, TOP, 'state')

window.mainloop()
