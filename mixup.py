import cv2
import numpy as np
from PIL import Image, ImageDraw

def rand(a = 1, b =1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, jetter =.3, hue=.1, sat=0.7,val=0.4, random=True):
    line = annotation_line.split()
    # 读取图像并转换成RGB图像
    image = Image.open(line[0])
    image = image.convert('RGB')
    # 获取图像的高宽与目标高宽
    iw, ih = image.size
    h,w = input_shape
    # 获取预测框
    box = np.array([list(map(int, box.split(','))) for box in line[1:]]) 
    if not random:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        # 将图像多余的部分加上padding灰条
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB',(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image_data = np.array(new_image, np.float32)
        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:,[0,2]] = box[:,[0,2]] * nw/iw + dx
            