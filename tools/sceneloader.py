import os
import pandas as pd
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import cv2
import torch

def pathAndLabel(dir_str):
    imagepaths=[]
    imagelabels=[]
    abspath = os.path.abspath(dir_str)
    for sp in os.listdir(abspath):
        subpath = os.path.join(abspath,sp)
        if not os.path.isdir(subpath):
            continue
        for im in os.listdir(subpath):
            impath = os.path.join(abspath,sp,im)
            imagepaths.append(impath)
            imagelabels.append(sp)
    df = pd.DataFrame({"imagepaths":imagepaths,"imagelabels":imagelabels})
    df = df.sample(frac=1).reset_index(drop=True)
    borderline = len(df)*0.8
    borderline = int(borderline)
    traindf = df.iloc[0:borderline,:]
    valdf = df.iloc[borderline:len(df)]
    traindf.to_csv("/home/data1/ygq/scene/datas/15scene/train.csv")
    valdf.to_csv("/home/data1/ygq/scene/datas/15scene/val.csv")


    
class sceneset(Dataset.Dataset):
    def __init__(self, labelhtml,transform):
        df = pd.read_csv(labelhtml)
        self.Data = df["imagepaths"]
        self.Label = df["imagelabels"]
        self.transform = transform

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = cv2.imread(self.Data[index])
        data = cv2.resize(data,(256,256))
        label = torch.from_numpy(np.array(self.Label[index]))
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # data = torch.from_numpy(data)
        data = self.transform(data)
        return data, label

def create_dataset(transform_train,transform_test):
    train = sceneset("/home/data1/ygq/scene/datas/15scene/train.csv",transform_train)
    val = sceneset("/home/data1/ygq/scene/datas/15scene/val.csv",transform_test)

    return train,val


if __name__ == "__main__":
    # pathAndLabel( '/home/data1/ygq/scene/datas/15scene/' )
    # sceneset("/home/data1/ygq/scene/datas/15scene/train.csv")
    trainset,valset=create_dataset()
    dataloader = DataLoader.DataLoader(trainset,batch_size= 1, shuffle = True, num_workers= 0)
    for x,y in dataloader:
        print(x.shape,y)
