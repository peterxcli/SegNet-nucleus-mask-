import torch
from torch import nn
from torch.nn.modules import module
from torch.utils.data import Dataset, DataLoader
from os import listdir
import numpy as np
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class NucleusDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.root = './Nucleus dataset/'
        self.allfolderlist = listdir(self.root)
        print("There are total " + str(len(self.allfolderlist)) + " data")
        self.folderlist = []

        self.split = split
        if self.split == 'train':
            print("Get training dataset")
            for n in range(int(len(self.allfolderlist) / 2)):
                self.folderlist.append(self.allfolderlist[n])
            print("There are " + str(len(self.folderlist)) + " data in training dataset")

        elif self.split == 'test':
            print("Get testing dataset")
            for n in range(int(len(self.allfolderlist) / 2), int(len(self.allfolderlist))):
                self.folderlist.append(self.allfolderlist[n])
            print("There are " + str(len(self.folderlist)) + " data in testing dataset")

        else:
            print("Wrong!")

    def __len__(self):
        return len(self.folderlist)

    def __getitem__(self, index):

        foldername = self.folderlist[index]
        filename = foldername.split(".")[0] + ".png"

        img = cv.imread(self.root + foldername + "/images/" + filename)
        img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)
        img_np = np.array(img, dtype=np.float32)
        flat_img_np = np.empty(shape=(3, 224, 224))
        for x in range(224):
            for y in range(224):
                flat_img_np[0][x][y] = (img_np[x][y][0] + img_np[x][y][1] + img_np[x][y][2]) / 765

        sum = 0
        for x in range(224):
            for y in range(224):
                sum += flat_img_np[0][x][y]
        flat_img_np = flat_img_np * 0.5 / sum * 224 * 224

        outputpath = self.root + foldername + "/masks/"

        isfirst = True
        for objectpic in listdir(outputpath):
            obimg = cv.imread(outputpath + objectpic)
            obimg = cv.resize(obimg, (224, 224), interpolation=cv.INTER_LINEAR)
            if isfirst:
                obimg_np = np.array(obimg, dtype=np.float32)
                isfirst = False
            else:
                obimg_np += np.array(obimg, dtype=np.float32)
                obimg_np / 2

        flat_obimg_np = np.empty(shape=(1, 224, 224))
        for x in range(224):
            for y in range(224):
                if obimg_np[x][y][0] == 255:
                    flat_obimg_np[0][x][y] = 1
                else:
                    flat_obimg_np[0][x][y] = 0

        return flat_img_np, flat_obimg_np


class SegNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()  # create a copy of the state dict
        th = torch.load(model_path).state_dict()  # load the weigths
        # for name in th:
        # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)


def train(model, dataloader, bs, optimizer, loss_fn, epochs, filename):
    model.to(device)
    model.train(True) # Set trainind mode = true

    f = open("./limf" + filename + ".txt", 'a')
    # 每個epoch會看過一遍所有資料
    for epoch in range(epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch+1, epochs))

        step = 0
        eloss =0
        eiou = 0
        emae = 0
        ef = 0

        # 透過訓練用的dataloader隨機獲取資料
        for x, y in dataloader:
            x = x.float()
            x.requires_grad = True
            x = x.to(device)
            y = y.float()
            y = y.to(device)
            # .cuda()表放入GPU

            step += 1
            optimizer.zero_grad() # 將gradient歸零

            output = model(x)  # 令圖片經過神經網路(黑盒子)，得出預測結果
            output = output.to(device)

            loss = loss_fn(output, y)    # 計算損失
            iou = IOU(output, y, bs)
            mae = MAE(output, y, bs)
            fmeasure = Fmeasure(output, y, bs=bs)

            y = y.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            output = 1*(output[:,:,:,:]>0.5)

            ys = ""
            os = ""
            for n in range(224):
                for m in range(224):
                    ys+=str(int(y[0][0][n][m]))
                    os+=str(int(output[0][0][n][m]))
                print(ys+"     "+os,file=f)
                ys = ""
                os = ""
            print("----------",file=f)

            eloss+=loss
            eiou+=iou
            emae+=mae
            ef+=fmeasure

            loss.backward()   # 反向傳播
            optimizer.step()  # 進行參數更新

            print('Current step: {}  Loss: {}IOU: {} MAE: {} F-score: {}'.format(step, loss, iou, mae, fmeasure),file=f)
            if step % 10 == 0:
                print('Current step: {}  Loss: {}  MAE: {} F-score: {}'.format(step, loss, mae,fmeasure))

        if (epoch+1) % 10 == 0 or epoch == 0:
            torch.save(model, "Model/SegNet_bce_e{}.sav".format(epoch+1))

        eloss/=step
        eiou/=step
        emae/=step
        ef/=step
        print('-----Epoch {} finish Loss: {}  IOU: {} MAE: {} F: {}'.format(epoch+1, eloss, eiou, emae, ef),file=f)
        print('-----Epoch {} finish Loss: {}   MAE: {} F: {}'.format(epoch+1, eloss, emae, ef))
    f.close()


def IOU(prediction, groundtruth, bs=15):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1 * (prediction[:, :, :, :] > 0.5)

    intersection = 0
    union = 0
    for l in range(bs):
        for n in range(224):
            for m in range(224):
                if prediction[l][0][n][m] == 1 and groundtruth[l][0][n][m] == 1:
                    intersection += 1

                if prediction[l][0][n][m] == 1 or groundtruth[l][0][n][m] == 1:
                    union += 1

    iou_score = intersection / union

    return iou_score


def MAE(prediction, groundtruth, bs=15):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1 * (prediction[:, :, :, :] > 0.5)

    error = 0

    for l in range(bs):
        for x in range(224):
            for y in range(224):
                if prediction[l][0][x][y] != groundtruth[l][0][x][y]:
                    error += 1

    return error / 224 / 224 / bs


def Fmeasure(prediction, groundtruth, bs=15, b=1):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1 * (prediction[:, :, :, :] > 0.5)

    TP = 0
    FP = 0
    FN = 0

    for l in range(bs):
        for x in range(224):
            for y in range(224):
                if prediction[l][0][x][y] == 1 and groundtruth[l][0][x][y] == 1:
                    TP += 1
                if prediction[l][0][x][y] == 1 and groundtruth[l][0][x][y] == 0:
                    FP += 1
                if prediction[l][0][x][y] == 0 and groundtruth[l][0][x][y] == 1:
                    FN += 1

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0

    return ((1 + b * b)  *precision * recall) / (b * b * (precision + recall))

if __name__ == '__main__':
    train_dataset = NucleusDataset("train")
    bs = 10  # batch size
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, drop_last=True, shuffle=False)

    # model = SegNet(3, 1)
    model = torch.load("Model/SegNet_bce_e1.sav")
    train(model, train_loader, bs, torch.optim.Adam(model.parameters(), lr=0.01), nn.BCEWithLogitsLoss(), 20, "_bce_e10")

    # torch.save(model, "/Model/SegNet_bce_e10.sav")

  # model_e10 = torch.load("C:/Users/CVML/Desktop/CVML142/Model/SegNet_bce_e10.sav")
  # train(model_e10, train_loader, torch.optim.Adam(model_e10.parameters(), lr=0.005), nn.BCELoss(), 10, "_bce_e20")
  # torch.save(model_e10, "C:/Users/CVML/Desktop/CVML142/Model/SegNet_bce_e20.sav")

  # model_e20 = torch.load("C:/Users/CVML/Desktop/CVML142/Model/SegNet_bce_e20.sav")
  # train(model_e20, train_loader, torch.optim.Adam(model_e20.parameters(), lr=0.0025), nn.BCELoss(), 10, "_bce_e30")
  # torch.save(model_e20, "C:/Users/CVML/Desktop/CVML142/Model/SegNet_bce_e30.sav")
