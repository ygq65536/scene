import argparse
import os
# import shutil
import time
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import models.resnet as resnet
import tools.sceneloader as loader
import torchvision.transforms as transforms


def main():
    model = resnet.resnet50()
    model = torch.nn.DataParallel(model)
    model.cuda()

    ofpthpath = "/home/data1/ygq/scene/best/best.pth"
    print("=> loading checkpoint '{}'".format(ofpthpath))
    checkpoint = torch.load(ofpthpath)
    model.load_state_dict(checkpoint['state_dict'])


    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    transform_train = transform_test
    trainset,valset = loader.create_dataset(transform_train,transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32, shuffle=True,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=10, shuffle=False,
        num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    model.train()
    featurepath = "/home/data1/ygq/scene/datas/15scene/"
    trainmat = validate(train_loader, model, criterion)
    scio.savemat(featurepath+"train.mat", trainmat)
    model.eval()
    valmat = validate(val_loader, model, criterion)
    scio.savemat(featurepath+"val.mat", valmat)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(self.val,self.count)


def validate(val_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    allfeatures = np.zeros((1,2048))
    alllabels = np.zeros((1,1))
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            features,output = model(input_var)
            # print(features.shape)
            allfeatures = np.vstack((allfeatures, features.cpu().numpy()))
            # alllabels.append(target.cpu().numpy())

            # print("-----------------------------------")

            # print(target)
            # print(target.cpu().numpy().reshape((-1,1)))
                        
            # print("-----------------------------------")

            alllabels = np.vstack((alllabels, target.cpu().numpy().reshape((-1,1))))
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    allfeatures = allfeatures[1:]
    alllabels = alllabels[1:]
    print(allfeatures.shape)
    print(alllabels.shape)
    return {"allfeatures":allfeatures,"alllabels":alllabels}



            


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

    # import collections
    # ofpthpath = "/home/data1/ygq/scene/checkpoints/checkpoint.th"
    # filename = "/home/data1/ygq/scene/checkpoints/checkpointed.th"
    # fy = torch.load(ofpthpath)['state_dict']
    # print(type(fy))
    # state = collections.OrderedDict()
    # for key, value in fy.items():
    #     newkey = key[7:]
    #     state[newkey] = value
    # torch.save(state, filename)

    # featurepath = "/home/data1/ygq/scene/datas/15scene/"
    # train = scio.loadmat(featurepath+"val.mat")
    # print(train["allfeatures"].shape)
    # print(train["alllabels"].shape)

    # weightx = scio.loadmat("/home/data1/ygq/scene/pretrainedweights/weights.mat")["X"].astype(np.float32)
    # print(type(weightx),type(weightx[0][0]),weightx.shape)
    # print(weightx)
    # weightx=torch.from_numpy(weightx).cuda()

    

