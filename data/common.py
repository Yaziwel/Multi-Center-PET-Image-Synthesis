import torch
import os
import SimpleITK as sitk 
import pickle
# import pydicom
import numpy as np
import datetime
import pandas as pd

class transformPET:
    def __init__(self, data_range, mode="linear"):
        self.r = data_range
        self.m = mode

    def cut(self,x):
        thresh=self.r
        x[x>thresh]=thresh
        x[x<=0]=0.0
        _, _, d,h,w=x.shape
        x = x[:, :, 4:d-4, 16:h-16, 16:w-16]
        return x 

    def clip(self,x):

        x[x>self.r]=self.r
        x[x<=0]=0.0
        return x 
    
    def get_error_mask(self, low, full, error_thresh=0.01):
        error = torch.abs(full-low) 
        mask = (error>error_thresh).type(torch.FloatTensor).to(full.device)
        return mask
    
    def normalize(self, img, cut=False):
        img = self.clip(img)
        if self.m == "exp":
            c = torch.log(torch.Tensor([self.r+1])).to(img.device)
            img = torch.log(1+img)/c
        else:
            img = img/self.r
        
        if cut:
            img = self.cut(img)
        return img
    def denormalize(self, img):
       	if self.m=='exp':
       		c = torch.log(torch.Tensor([self.r+1])).to(img.device)
       		img = torch.exp(c*img)-1
       	else:
       		img*=self.r
       	img = self.clip(img)
        return img
            
@torch.no_grad()
def synthesisOneAxial(model, img, kernel_size=64, stride=32, crop_size = 3):
    model.eval()
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride==D:
            break
        elif idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ######
        G_patches = model(patches)
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

@torch.no_grad()
def merge_patch(model, img, kernel_size=64, stride=32,crop_size = 3):
    model.eval()
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride==D:
            break
        elif idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ###### 
        G_patches = patches.clone().detach() 
        for i in range(len(patches)):
            G_patches[i] = model(patches[[i],:,:,:,:])
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

class dataIO:
    def __init__(self):
        self.reader = {
            '.img':self.load_itk,
            '.bin':self.load_bin, 
            '.txt':self.load_txt
            
            }
        self.writer = {
            '.img':self.save_itk,
            '.bin':self.save_bin,
            '.csv':self.save_csv,
            '.txt':self.save_txt,
            }
    
    
    def save_itk(self, data, path):
        sitk.WriteImage(sitk.GetImageFromArray(data), path) 
        
    def load_itk(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
        
    def load_bin(self,path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data 
    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read() 
        return data

    def save_bin(self,data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)


    def save_csv(self, data_dict, path):
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)
    
    def save_txt(self, s, path):
        with open(path,'w') as f:
            f.write(s) 
            

        
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex
    
    def load(self, path):
        ex = self.getFileEX(path)
        return self.reader[ex](path)
    def save(self, data, path):
        ex = self.getFileEX(path)
        return self.writer[ex](data, path)



# def load_suv_array(path):
    
#     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
#     arr = np.stack([s.pixel_array for s in slices])
    
#     info = slices[0]
#     activity = info.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
#     weight = info.PatientWeight
#     injectionTime = info.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
#     injectionTime = datetime.datetime.strptime(str(injectionTime),"%H%M%S")
    
#     acqTime = info.AcquisitionTime
#     acqTime = datetime.datetime.strptime(str(acqTime),"%H%M%S")
    
#     DecayCorrection = info.DecayCorrection
    
#     if DecayCorrection == " ADMIN":
#         duration = 0
#     elif DecayCorrection == "START":
#         duration = (acqTime - injectionTime).total_seconds()
#     else:
#         raise Exception("Unknown DecayCorrection")
    
#     halflife = info.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
#     a=info.RescaleSlope
#     b=info.RescaleIntercept
    
#     scale = 2**(duration/halflife) * weight * 1000 / activity
    
#     result = (a*arr+b)*scale
    
#     return result

# def load_mask_array(patient_path, mask_path):
#     slices_patient = [pydicom.read_file(patient_path + '/' + s) for s in os.listdir(patient_path)]
#     slices_patient.sort(key = lambda x: float(x.ImagePositionPatient[2]))
#     pos = [x.ImagePositionPatient[2] for x in slices_patient]
#     del slices_patient
    
    
#     slices_mask = [pydicom.read_file(mask_path + '/' + s) for s in os.listdir(mask_path)]
#     slices_mask.sort(key = lambda x: float(x.ImagePositionPatient[2]))
#     min_pos = slices_mask[0].ImagePositionPatient[2]
#     max_pos = slices_mask[-1].ImagePositionPatient[2]
#     arr=[]
#     empty_slice=np.zeros((192, 192))
#     count=0
#     for i in range(len(pos)):
#         if abs(pos[i]-min_pos)<1 or (pos[i]>=min_pos and pos[i]<=max_pos) or abs(pos[i]-max_pos)<1:
#             arr.append(slices_mask[count].pixel_array)
#             count+=1
#         else:
#             arr.append(empty_slice)
#     if len(arr)!=len(pos):
#         print(mask_path)
#     arr = np.stack(arr)
    
#     return arr 

def normalize(x):
    x = torch.stack([x for i in range(3)], dim=0)
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) 
    
    return x