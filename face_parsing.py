from PIL import Image
from parsing.face_parsing import parsing, vis_parsing_maps
import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch_device = "mtgpu"
torch_device = "cuda"
#torch_device = "cpu"
if torch_device == "mtgpu":
    import musa_torch_extension
    a = torch.tensor([1])
    a.to("mtgpu")
    print("-------")
device = torch.device(torch_device)

image = Image.open('imgs/liwei.jpg')

res = parsing(image, cp='face_parsing_79999_iter.pth', device=torch_device)
vis_parsing_maps(image, res, show=False, save_im=True)
