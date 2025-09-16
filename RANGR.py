'''
RANGR:      [R]espiratory [A]uto[N]avigator for [G]olden-angle [R]adial free-breathing MRI.

Author:     J.J. Nario, MD, MS
            The Otazo Lab, MSKCC, New York, NY
'''

########################################################
name2load = 'RANGRweights.pt'

########################################################
#   load libraries
print('Importing libraries...')
import numpy as np
import torch
import torch.nn as nn
import time
import sys
from scipy.io import savemat
print('Libraries imported.')

########################################################
#   RANGR inference prep
img_filepath = sys.argv[1]      # file path for .bin file containing s-t matrix
print('Loading s-t matrix at {}'.format(img_filepath))
t0 = time.time()
img = np.fromfile(img_filepath, dtype="float32")
img = img.reshape((400,900), order='F')
img = torch.from_numpy(img)
tN = time.time()
print('Time to load s-t matrix: {:.6f}sec'.format(tN-t0))

########################################################
#   RANGR ARCHITECTURE
class RANGR(nn.Module):
    def __init__(self):
        super(RANGR, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1))
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1))
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1))
        )
        
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1))
        )
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1))
        )
                
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=(1,1),
                      stride=(1,1),
                      padding=0),
        )
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        signal = self.classifier(x).squeeze()
                
        return signal

########################################################
#   initialize model
print('Loading RANGR architecture...')
tO = time.time()
model = RANGR()
model = model.to('cuda')
tN = time.time()
print('Time to load RANGR architecture: {:.6f}sec'.format(tN-t0))

########################################################
#   load model weights
print('Loading RANGR weights...')
tO = time.time()
model.load_state_dict(torch.load(name2load))
tN = time.time()
print('Time to load RANGR weights: {:.6f}sec'.format(tN-t0))

########################################################
#   RANGR inference
print('Performing DL-based motion estimation...')
model.eval()
with torch.no_grad():
    data = img.to('cuda')
    data = torch.unsqueeze(data,(0))
    print(data.size())
    t0 = time.time()
    output = model(data)
    tN = time.time()
print('Done! RANGR inference time: {:.6f}sec'.format(tN-t0))
output = output.cpu().detach().numpy()
output = (output - output.min()) / (output.max() - output.min())    # 0-1 normalization

########################################################
#   save output .mat file
filename = sys.argv[2]      # file path for output .mat file containing motion estimation waveform
savemat(filename, {"data": output})
print('RANGR output saved at {}'.format(sys.argv[2]))
