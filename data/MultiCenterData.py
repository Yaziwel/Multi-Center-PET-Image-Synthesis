from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO 
import glob

io=dataIO() 

def getIDX(path):
    num = io.load(path).split("\n") 
    num = [int(x) for x in num] 
    return num

class Train_Data(Dataset):
    def __init__(self, root, center, sample_size=None, train_num=10000):
        
        self.sample_size=sample_size
        self.Full=[] 
        self.Low=[] 
        for i in range(train_num): 
            self.Low.append(os.path.join(root, center, "train_low_patch", "{}.bin".format(i))) 
            self.Full.append(os.path.join(root, center, "train_full_patch", "{}.bin".format(i)))
        self.length = train_num

    def __len__(self):
        return self.length 
    
    def getCropIdx(self, shape):
        idz = np.random.randint(shape[1]-self.sample_size)
        idx = np.random.randint(shape[2]-self.sample_size)
        idy = np.random.randint(shape[3]-self.sample_size)
        # print(result)
        return idx, idy, idz

    def __getitem__(self, idx):

       
        imgL = io.load(self.Low[idx])
        imgF =io.load(self.Full[idx]) 
        
        if self.sample_size is None:
            return imgL, imgF
        else:
            idx, idy, idz = self.getCropIdx(imgL.shape) 
            return (imgL[:,idz:idz+self.sample_size, idx:idx+self.sample_size, idy:idy+self.sample_size],
                    imgF[:,idz:idz+self.sample_size, idx:idx+self.sample_size, idy:idy+self.sample_size])

    
class Test_Data(Dataset):
    def __init__(self, root, center_list):
        super(Test_Data, self).__init__() 
        self.Full=[]
        self.Low=[] 
        for center in center_list:
            number = getIDX(os.path.join(root, center, "{}-miccai-test.txt".format(center))) 
            for n in number:
                self.Full.append(os.path.join(root, center, "Full", "Patient_{}.bin".format(n))) 
                self.Low.append(os.path.join(root, center, "Low", "Patient_{}.bin".format(n)))

        self.length = len(self.Low)

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        imgL = io.load(self.Low[idx])
        imgF =io.load(self.Full[idx])
        f , file=os.path.split(self.Low[idx])
        name,_ = os.path.splitext(file)
        center = f.split("/")[-2]
        return imgL, imgF, name, center 

class Val_Data(Dataset):
    def __init__(self, root, center_list):
        super(Val_Data, self).__init__() 
        self.Full=[]
        self.Low=[] 
        for center in center_list:
            number = getIDX(os.path.join(root, center, "{}-miccai-test.txt".format(center))) 
            n = number[8]
            self.Full.append(os.path.join(root, center, "Full", "Patient_{}.bin".format(n))) 
            self.Low.append(os.path.join(root, center, "Low", "Patient_{}.bin".format(n)))

        self.length = len(self.Low)

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        imgL = io.load(self.Low[idx])
        imgF =io.load(self.Full[idx])
        f , file=os.path.split(self.Low[idx])
        name,_ = os.path.splitext(file)
        center = f.split("/")[-2]
        return imgL, imgF, name, center 
    

class Test_DRF_Data(Dataset):
    def __init__(self, root, dose_list=['30s', '45s', '60s']):
        super(Test_DRF_Data, self).__init__() 
        self.Full=[]
        self.Low=[] 
        self.Doses=[] 
        for dose in dose_list:
            number = getIDX(os.path.join(root, 'C6-m660', "C6-m660-miccai-test.txt")) 
            for n in number:
                self.Full.append(os.path.join(root, 'C6-m660', "Full", "Patient_{}.bin".format(n))) 
                self.Low.append(os.path.join(root, 'C6-m660', "Low-{}".format(dose), "Patient_{}.bin".format(n))) 
                self.Doses.append(dose)

        self.length = len(self.Low)

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        imgL = io.load(self.Low[idx])
        imgF =io.load(self.Full[idx])
        f , file=os.path.split(self.Low[idx])
        name,_ = os.path.splitext(file)
        dose = self.Doses[idx]
        return imgL, imgF, name, dose 


if __name__=="__main__": 
    from torch.utils.data import DataLoader 
    from tqdm import tqdm 
    center_list = ["C1-m660", "C2-m660", "C3-Flight", "C4-uExplorer", "C5-Simens"] 
    data_root = "/home/data/zhiwen/experiment/Multi_Center/data"
    train_loader_list = []
    for center in center_list: 
        ds = Train_Data(root=data_root, center=center)
        train_loader_list.append(DataLoader(ds, batch_size=1, shuffle=True)) 
    
    valid_loader = DataLoader(Val_Data(root=data_root, center_list=center_list), batch_size=1) 
    
    test_loader = DataLoader(Test_Data(root=data_root, center_list=center_list), batch_size=1)
    
    

    for counter,data in enumerate(tqdm(zip(*train_loader_list))): 
        import pdb 
        pdb.set_trace()