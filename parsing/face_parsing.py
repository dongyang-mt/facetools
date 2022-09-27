#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .model import BiSeNet


def vis_parsing_maps(im, parsing_anno, stride=1, show=False, save_im=False, save_path='imgs/'):

    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if show:
        cv2.imshow('parsing res', vis_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save result or not
    if save_im:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(osp.join(save_path, 'parsing_maps.png'), vis_parsing_anno)
        cv2.imwrite(osp.join(save_path, 'parsing.jpg'), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

import time
def save_benchmark(timecost, modulename, torch_device="cpu", folder="benchmark"):
    cost_np = np.array(timecost)
    print("==>> cost_np.shape: ", cost_np.shape)
    print("==>> type(cost_np): ", type(cost_np))
    import csv
    os.makedirs(folder, exist_ok=True)
    csvFile = open(os.path.join(folder, torch_device + "_analyze.csv"), 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerow([torch_device, "mean", "std", "min", "max", "count"])
    for i, name in enumerate(modulename):
        writer.writerow([name, cost_np[:,i].mean(), cost_np[:,i].std(), cost_np[:,i].min(), cost_np[:,i].max(), cost_np[:,i].shape[0]])
    np.savetxt(os.path.join(folder, torch_device + "_timecost.csv"), cost_np, delimiter=",")


def parsing(imgs, cp='checkpoint/face_parsing.pth', device="cpu"):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(cp, map_location="cpu"))
    net.to(device)
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if not isinstance(imgs, list):
            shape = imgs.size
            image = imgs.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)
            # warm up
            for i in range(10):
                out = net(img)[0]

            timecost = []
            for i in range(100):
                t1 = time.time()
                out = net(img)[0]
                t2 = time.time()
                timecost.append([(t2 - t1) * 1000.0])
            save_benchmark(timecost, ["face_parsing"], device)
            parsing_maps = out.squeeze(0).cpu().numpy().argmax(0).astype('float32')
            parsing_maps = cv2.resize(parsing_maps, shape, interpolation=cv2.INTER_NEAREST)
            return parsing_maps

        else:
            parsing_list = []
            for img in imgs:
                shape = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                out = net(img)[0]
                parsing_maps = out.squeeze(0).cpu().numpy().argmax(0).astype('float32')
                parsing_maps = cv2.resize(parsing_maps, shape, interpolation=cv2.INTER_NEAREST)
                parsing_list.append(parsing_maps)
            return parsing_list


