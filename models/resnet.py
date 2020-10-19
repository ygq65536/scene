#导入必要模块
import torch
import torch.nn as nn
import models.resnetFromTorchvision as models

def resnet50():
    model=models.resnet50(pretrained=False)
    model.load_state_dict(torch.load("/home/data1/ygq/scene/pretrainedweights/resnet50.pth"))
    # for param in model.parameters():
    #     param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False

    class_num = 15
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in,class_num)
    return model

if __name__=="__main__":
    model = resnet50()
    print(model.fc)
    