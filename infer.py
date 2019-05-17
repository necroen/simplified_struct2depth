import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import cv2
import matplotlib.pyplot as plt
from imageio import imread

import custom_transforms
from DispNetS import DispNetS


path = "./xx.jpg"  #  128 x 416 

def load_as_float(path):
    return imread(path).astype(np.float32)


img = load_as_float(path)

img_copy = cv2.imread(path)


to_tensor = custom_transforms.Compose([ custom_transforms.ArrayToTensor() ])
image_stack = to_tensor( [img] )

img = image_stack[0]
#print(img.size())

img = img.expand(1,3,128,416)
#print(img.size())

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


disp_net = DispNetS().to(device)
disp_net.init_weights()
disp_net.load_state_dict(torch.load("./disp_net.pth"))
disp_net.eval()


multiscale_disps_i, _ = disp_net(img)

disp = multiscale_disps_i[0]
print(disp.size())

disp = disp.detach().cpu().numpy()
disp = disp.squeeze()

disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
#disp = disp.astype(np.uint8)

plt.figure()  # cv2.cvtColor(cc,  cv2.COLOR_BGR2RGB)
plt.imshow( disp ),plt.show()
plt.title('disp')

plt.figure()  # cv2.cvtColor(cc,  cv2.COLOR_BGR2RGB)
plt.imshow( cv2.cvtColor(img_copy,  cv2.COLOR_BGR2RGB) ),plt.show()
plt.title('img')