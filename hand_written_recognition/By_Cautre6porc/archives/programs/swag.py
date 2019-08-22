from PIL import Image
import random as rd

img = Image.open('train.png')
imgPixs = img.load()

new = Image.new('RGB', (3248, 2184))
newPixs = new.load()

for x in range(0, 3248, 28):
    for y in range(0, 2184, 28):
        i = rd.randrange(0, 20440, 28)
        j = rd.randrange(0, 2044, 28)
        r, g, b = rd.randrange(80, 256), rd.randrange(80, 256), rd.randrange(80, 256)
        for px in range(28):
            for py in range(28):
                gray = imgPixs[i + px, j + py] / 255
                color = (int(r * gray), int(g * gray), int(b * gray))
                newPixs[x + px, y + py] = color

new.save('rand.png')
