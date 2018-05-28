from PIL import Image, ImageFilter
import pathlib

number = 1  # nmber of experiments
seed = 1  # seed
strings = "{0}_{1}".format(number, seed)
# Pillow のGIF生成，画像読み込みは以下のサイトを参照
# https://note.nkmk.me/python-pillow-gif/
# https://note.nkmk.me/python-pillow-basic/
path = pathlib.Path("result_{}/preview".format(strings))

# store image to use as frame to array "imgs"
imgs = []
for epoch in range(1, 301):
    img = Image.open(path / "image_{}epoch.jpg".format(epoch))
    imgs.append(img)

# make gif
imgs[0].save('anim_{}.gif'.format(strings), save_all=True, append_images=imgs[1:],
             optimize=False, duration=200, loop=0)
