from os import makedirs
import torch
from torch.utils.data import DataLoader
from torch import optim
import tqdm
from dataset import CatSegmetationDataset as Dataset
from torchvision import transforms
from unet import UNet,DiceLoss
import numpy as np

def data_loaders(args):
    dataset_train = Dataset(
        images_dir=args.images,
        image_size=args.image_size,
        transform=transforms.ToTensor()
        )
    
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
        )
    return loader_train

def main(args):

    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    loader_train= data_loaders(args)
    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)
    # 损失函数
    dsc_loss = DiceLoss()
    # 优化方法
    optimizer = optim.Adam(unet.parameters(), lr = args.lr)
    
    loss_train = []
    step = 0
    for epoch in tqdm(range(args.epochs),total= args.epochs):
        unet.train()
        for i, data in enumerate(loader_train):
            step += 1
            x,y_true = data
            x,y_true = x.to(device), y_true.to(device)
            y_pred = unet(x)
            optimizer.zero_grad()
            loss = dsc_loss(y_pred, y_true)
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print("Step ",step, "Loss",np.mean(loss_train))
    torch.save(unet,args.weights + 'unet_epoch_{}.pth'.format(epoch))
    
if __name__ == "__main__":
    main(args)