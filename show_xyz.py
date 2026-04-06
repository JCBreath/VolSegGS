import torch
import numpy as np
import cv2

xyz = torch.load('trash/depth.pth').cpu().numpy()
print(xyz.shape)

x = xyz[0, :, :]
x = (x - x.min()) / (x.max() - x.min()) * 255
y = xyz[1, :, :]
y = (y - y.min()) / (y.max() - y.min()) * 255
z = xyz[2, :, :]
z = (z - z.min()) / (z.max() - z.min()) * 255
xyz = np.stack([x, y, z], axis=2).astype(np.uint8)
print(xyz)
cv2.imwrite('trash/output_image.png', xyz)
