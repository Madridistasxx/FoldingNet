import argparse
import os
import time
import sys
sys.path.append("../MT_3967")

import torch
import torch.optim as optim

from datasets import ShapeNetPartDataset
from model import AutoEncoder

from src.data.ddacs import PointCloudDDACSDataset
from torch_geometric.loader import DataLoader
from chamfer_test import ChamferDistanceTorch


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="/mnt/data/datasets/ddacs/")
parser.add_argument('--npoints', type=int, default=2048)
parser.add_argument('--mpoints', type=int, default=2025)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--log_dir', type=str, default='./log')
args = parser.parse_args()


# prepare training and testing dataset
train_dataset = PointCloudDDACSDataset(root=args.root, split="train")
test_dataset = PointCloudDDACSDataset(root=args.root, split="test")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
autoendocer = AutoEncoder()
autoendocer.to(device)

# loss function
cd_loss = ChamferDistanceTorch()
# optimizer
optimizer = optim.Adam(autoendocer.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)

batches = int(len(train_dataset) / args.batch_size + 0.5)

min_cd_loss = 1e3
best_epoch = -1

print('\033[31mBegin Training...\033[0m')
for epoch in range(1, args.epochs + 1):
    # training
    start = time.time()
    autoendocer.train()
    for i, data in enumerate(train_dataloader):
        blank = data['blank']
        point_clouds = blank.x
        B = point_clouds.shape[0]/2048
        point_clouds = point_clouds.view(int(B), 2048, 3)
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = autoendocer(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, args.epochs, i + 1, batches, ls.item() / len(point_clouds)))
    
    # evaluation
    autoendocer.eval()
    total_cd_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            blank = data['blank']
            point_clouds = blank.x
            B = point_clouds.shape[0]/2048
            point_clouds = point_clouds.view(int(B), 2048, 3)
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = autoendocer(point_clouds)
            ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            total_cd_loss += ls.item()
    
    # calculate the mean cd loss
    mean_cd_loss = total_cd_loss / len(test_dataset)

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        torch.save(autoendocer.state_dict(), os.path.join(args.log_dir, 'model_lowest_cd_loss.pth'))
    
    # save the model every 100 epochs
    if (epoch) % 100 == 0:
        torch.save(autoendocer.state_dict(), os.path.join(args.log_dir, 'model_epoch_{}.pth'.format(epoch)))
    
    end = time.time()
    cost = end - start

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
        epoch, args.epochs, mean_cd_loss, min_cd_loss, best_epoch))
    print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))
