import torch
import torch.nn as nn
from .common import psnr_slice, ssim_slice, PSNR3D, SSIM3D
from .lpips3d import LPIPS3D 


class Metrics_slices:
    def __init__(self, metric = ['psnr', 'ssim', 'lpips']): 
        super(Metrics_slices, self).__init__() 
        self.metric = metric

        if 'lpips' in metric:

            self.lpips = LPIPS3D(method='alex', device='cuda')
            

    def __call__(self, pred, real): 
        result = {}
        pred = pred/pred.max() 
        real = real/real.max() 
        
        if 'lpips' in self.metric:
            result['lpips'] = self.lpips(pred, real).item() 
        
        
        pred = pred.detach().cpu().numpy().squeeze() 
        real = real.detach().cpu().numpy().squeeze()
        
        if 'psnr' in self.metric:
            result['psnr'] = psnr_slice(pred, real) 
        
        if 'ssim' in self.metric:
            result['ssim'] = ssim_slice(pred, real)

        return result 
    
class Metrics(nn.Module):
    def __init__(self, metric = ['psnr', 'ssim', 'lpips']): 
        super(Metrics, self).__init__()
        self.methods=[]
        self.names=[]
        self.need_normalized=['lpips', 'psnr']
        
        if 'psnr' in metric:
            self.names.append('psnr')
            self.methods.append(PSNR3D())
        if 'ssim' in metric:
            self.names.append('ssim')
            self.methods.append(SSIM3D())
        if 'lpips' in metric:
            self.names.append('lpips')
            self.methods.append(LPIPS3D(method='alex', device='cuda'))
            

    def forward(self, img1, img2):
        result = {}
        for i in range(len(self.names)):
            if self.names[i] in self.need_normalized:
                result[self.names[i]]=self.methods[i](img1/img1.max(), img2/img2.max()) 
            else:
                result[self.names[i]]=self.methods[i](img1, img2)
        return result