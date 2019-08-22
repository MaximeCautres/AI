""" Simulate and draw a physical pendulum """

from scipy.integrate import odeint
from tkinter import *
import numpy as np
from math import *

M = 2
m = 0.5
L = 1
g = 10
d = 0.5
index = None


def pend(y, t, u, M, m, L, g, d):

    x, x_p, theta, theta_p = y

    Sy = sin(theta)
    Cy = cos(theta)
    A = M + m * Sy**2
    B = m * L * theta_p**2 * Sy

    dy_1 = x_p
    dy_2 = (1/A) * (B - m * g * Cy * Sy - d * x_p + u)
    dy_3 = theta_p
    dy_4 = (1 / (L * A)) * ((M + m) * g * Sy - (B - d * x_p + u) * Cy)

    return [dy_1, dy_2, dy_3, dy_4]


def simulate(length, y0, u):

    t = np.linspace(0, length, 1000*length)
    dy = odeint(pend, y0, t, args=(u, M, m, L, g, d))

    return dy


def show(episodes_save):

    def animate():
        global index
        if index < frame_count:
            for k in range(episode_save_count):

                cart, stick, ball, text = objects[k]
                x = episodes_save[k][0][index, 0]
                theta = episodes_save[k][0][index, 2]

                cx = (x * unit + cart_r + 800) % 1600
                cy = road_h - cart_r
                px = cx + L * sin(theta) * unit
                py = cy - L * cos(theta) * unit

                canvas.coords(cart, cx - cart_r, cy - cart_r, cx + cart_r, cy + cart_r)
                canvas.coords(stick, cx, cy, px, py)
                canvas.coords(ball, px - ball_r, py - ball_r, px + ball_r, py + ball_r)
                canvas.coords(text, cx, 100 + 20 * k)

            index += 1
            canvas.after(1, animate)
        return

    global index
    index = 0
    unit = 50
    road_h = 300
    cart_r = 0.3 * unit
    ball_r = 0.2 * unit

    episode_save_count = len(episodes_save)
    frame_count = episodes_save[0][0].shape[0]

    window = Tk()
    window.title('Results')
    canvas = Canvas(window, width=1600, height=400, background='black')
    canvas.pack()
    canvas.create_line(0, road_h, 1600, road_h, fill='white')

    objects = []
    for k_ in range(episode_save_count):
        episode_number = str(episodes_save[k_][1])
        cart_ = canvas.create_rectangle(0, 0, 0, 0, fill='blue', outline='white')
        stick_ = canvas.create_line(0, 0, 0, 0, fill='white')
        ball_ = canvas.create_oval(0, 0, 0, 0, fill='red', outline='white')
        text_ = canvas.create_text(0, 0, text=episode_number, fill='white', font=('Helvetica', '20', 'bold'))
        objects.append((cart_, stick_, ball_, text_))

    start_button = Button(window, text="Start", command=animate)
    start_button.pack()

    mainloop()
