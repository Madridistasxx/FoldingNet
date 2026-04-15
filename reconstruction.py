import random
import torch
from datasets import ShapeNetPartDataset
from model import AutoEncoder
from utils import show_point_cloud, compare_point_clouds

import sys
sys.path.append("../MT_3967")
from src.data.ddacs import PointCloudDDACSDataset
from src.metrics.point_cloud_metrics import Chamfer3D


ae = AutoEncoder()
ae.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
ae.eval()

DATASET_PATH = "/mnt/data/datasets/ddacs/"
test_dataset = PointCloudDDACSDataset(root=DATASET_PATH, split="train")
blank0 = test_dataset.__getitem__(0)['blank']
input_pc = blank0.x.contiguous()
# show_point_cloud(input_pc)

input_tensor = input_pc.unsqueeze(0).permute(0, 2, 1)
output_tensor = ae(input_tensor)
# reconstructed_pc = output_tensor.permute(0, 2, 1).squeeze().detach().numpy()
reconstructed_pc = output_tensor.permute(0, 2, 1).contiguous().squeeze().detach()

# show_point_cloud(reconstructed_pc)

compare_point_clouds(input_pc, reconstructed_pc)

cd_loss = Chamfer3D()
print(cd_loss(reconstructed_pc,input_pc))