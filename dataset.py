import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CatSegmetationDataset(Dataset):
    # 模型的输入是3通道
    in_channels = 3
    # 模型的输出是1 通道数据
    out_channels = 1
    def __init__(self,images_dir,transform=None,image_size=256):
        print("Reading images...")
        # 将图片与mask读入后，分别存在在image_slices 与 mask_slices 中
        self.images_slices = []
        self.mask_slices = []
        image_root_path = os.path.join(images_dir,"JPEGImages")
        mask_root_path = os.path.join(images_dir,"SegmentationClassPNG")
        for image_name in os.listdir(image_root_path):
            mask_name = image_name.split(".")[0] + ".png"
            image_path = os.path.join(image_root_path,image_name)
            mask_path = os.path.join(mask_root_path,mask_name)
            im = np.asarray(Image.open(image_path).resize((image_size,image_size)))
            mask =  np.asarray(Image.open(mask_path).resize((image_size,image_size)))
            self.images_slices.append(im/255.)
            self.mask_slices.append(mask)
        self.transform = transform
    
    def __len__(self):
        return len(self.images_slices)
    
    def __getitem__(self, idx):
        image = self.images_slices[idx]
        mask = self.mask_slices[idx]
        # tensor 的顺序是(Batch_size, 通道，高，宽) 而numpy 读入后的顺序是（高，宽，通道）
        image = image.transpose(2,0,1)
        mask = mask[np.newaxis,:,:]
        image = image.astype(np.float32)
        mask = image.astype(np.float32)
        return image, mask
    


if __name__ == "__main__":
    pass