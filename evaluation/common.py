import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, log10
from skimage.metrics import structural_similarity, peak_signal_noise_ratio  

import numpy as np
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
    
class SSIM3D(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average).item()


class PSNR3D(nn.Module):
    def __init__(self):
        super(PSNR3D, self).__init__()

    def forward(self, img1, img2):
       data_range = img1.max()
       mse = torch.mean((img1 - img2) ** 2 )
       if mse < 1.0e-10:
          return 100
       return 10 * log10(data_range**2/mse)




def ssim_slice(pred, real):
    c_ssim = 0 
    data_range = real.max() 
    # import pdb 
    # pdb.set_trace()
    for i in range(pred.shape[0]):
        c_ssim += structural_similarity(pred[i, :, :], real[i, :, :], data_range=data_range)
    for i in range(pred.shape[1]):
        c_ssim += structural_similarity(pred[:, i, :], real[:, i, :], data_range=data_range)
    for i in range(pred.shape[2]):
        c_ssim += structural_similarity(pred[:, :, i], real[:, :, i], data_range=data_range)
    return c_ssim/sum(pred.shape)

def psnr_slice(pred, real):
    c_ssim = 0 
    data_range = real.max()
    for i in range(pred.shape[0]):
        c_ssim += peak_signal_noise_ratio(pred[i, :, :], real[i, :, :], data_range=data_range)
    for i in range(pred.shape[1]):
        c_ssim += peak_signal_noise_ratio(pred[:, i, :], real[:, i, :], data_range=data_range)
    for i in range(pred.shape[2]):
        c_ssim += peak_signal_noise_ratio(pred[:, :, i],real[:, :, i], data_range=data_range)
    return c_ssim/sum(pred.shape)



# def ssim3D(img1, img2, window_size = 11, size_average = True):
    # (_, channel, _, _, _) = img1.size()
    # window = create_window_3D(window_size, channel)
    
    # if img1.is_cuda:
        # window = window.cuda(img1.get_device())
    # window = window.type_as(img1)
    
    # return _ssim_3D(img1, img2, window, window_size, channel, size_average).item()


