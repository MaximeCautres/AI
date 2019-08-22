import pickle
import numpy as np
from PIL import Image


data = pickle.load(open('cifar_data_set_int', 'rb'))

img_pix = np.zeros((32, 32, 3, 60000), dtype=np.uint8)
img_pix[:, :, :, :50000] = data['train_x']
img_pix[:, :, :, 50000:] = data['test_x']

new = Image.new('RGB', (90*32, 60*32))
new_pix = new.load()

for x in range(90):
    for y in range(60):
        for px in range(32):
            for py in range(32):
                color = tuple(img_pix[py, px, :, x*32+y])
                new_pix[x*32 + px, y*32 + py] = color

new.save('wallpaper.png')
