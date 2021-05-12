import sys
sys.path.append('.../NIR-ISL2021master/') # change as you need
import torch
from thop.profile import profile
import time
import numpy as np

from models import EfficientUNet

net = EfficientUNet(num_classes=3).cpu()

example_input = torch.randn(1, 3, 480, 640).cuda()
flops, params = profile(net, (example_input,))
print('net FLOPs is: {:.3f} G, Params is {:.3f} M'.format(flops/1e9, params/1e6))

net.eval()
res = []

for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    example_output= net(example_input)
    torch.cuda.synchronize()
    end = time.time()
    res.append(end-start)
print('FPS is {:.3f}'.format(1/(np.mean(res))))