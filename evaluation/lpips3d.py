import lpips
import torch
import torch.nn as nn


class LPIPS3D(nn.Module):
    def __init__(self, method='alex', device='cuda'): 
        super(LPIPS3D, self).__init__()
        
        '''
        method = 'alex', 'vgg'
        '''

        self.loss_fn = lpips.LPIPS(net=method).to(device)


    def forward(self, img1, img2):
        # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
        score=[]
        B,C,D,H,W = img1.shape
        for i in range(D):
            score.append(self.loss_fn(img1[:,:,i,:,:],img2[:,:,i,:,:]).item())
    
        for i in range(H):
            score.append(self.loss_fn(img1[:,:,:,i,:],img2[:,:,:,i,:]).item())
    
        for i in range(W):
            score.append(self.loss_fn(img1[:,:,:,:,i],img2[:,:,:,:,i]).item())
        return sum(score)/len(score)



