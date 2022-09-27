from PIL import Image
from parsing.face_parsing import parsing, vis_parsing_maps

image = Image.open('imgs/9.jpg')

res = parsing(image, cp='79999_iter.pth')
vis_parsing_maps(image, res, show=False, save_im=True)
