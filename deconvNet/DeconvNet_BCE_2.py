import torch
from torch.utils.data import Dataset, DataLoader
from  torch import nn
import cv2 as cv
import numpy as np
from PIL import Image
from os import listdir
from Net import DeconvNet
from util import *

def train(model, dataloader, optimizer, loss_fn, epochs,filename):

    model.cuda()
    model.train(True) # Set trainind mode = true

    f = open("C:/Codeing area/Python/DeconvNet/acc"+filename+".txt", 'a')
    # 每個epoch會看過一遍所有資料
    with torch.no_grad():

        step = 0
        eloss =0
        eiou = 0
        emae = 0
        ef = 0

        # 透過訓練用的dataloader隨機獲取資料
        for x, y, z in dataloader:
            torch.cuda.empty_cache()
            x = x.float()
            x.requires_grad = True
            x = x.cuda()
            y = y.float()
            y = y.cuda()
            # .cuda()表放入GPU

            step += 1
            #optimizer.zero_grad() # 將gradient歸零

            output = model(x)  # 令圖片經過神經網路(黑盒子)，得出預測結果
            output = output.cuda()

            PP(output, y, z, step, filename)

            loss = loss_fn(output, y)    # 計算損失
            iou = IOU(output, y)
            mae = MAE(output, y)
            fmeasure = Fmeasure(output, y)

            eloss+=loss
            eiou+=iou
            emae+=mae
            ef+=fmeasure

            #loss.backward()   # 反向傳播
            #optimizer.step()  # 進行參數更新

            print('Current step: {}  Loss: {}  IOU: {} MAE: {} F: {}'.format(step, loss, iou, mae, fmeasure), file=f)

        eloss/=step
        eiou/=step
        emae/=step
        ef/=step
        print('finish Loss: {}  IOU: {} MAE: {} F: {}'.format(eloss, eiou, emae, ef),file=f)
        print('finish Loss: {}  IOU: {} MAE: {} F: {}'.format(eloss, eiou, emae, ef))
    f.close()

import matplotlib.pyplot as plt
import random as rand

def show(dataloader, num_image=4, img_size=10, offset=0, shuffle=False):
    _, figs = plt.subplots(num_image, 3, figsize=(img_size, img_size))
    for i in range(num_image):
        # if(shuffle==True):
        #     offset = rand.randint(0, min(len(dataloader)-i-1, len(camvid_test)-i-1))
        img_data, img_label = dataloader[i]
        pred, label = predict(img_data, img_label)
        img_data = Image.open(camvid_test.data_list[i+offset])
        img_label = Image.open(camvid_test.label_list[i+offset])
        img_data, img_label = crop(img_data, img_label)
        figs[i, 0].imshow(img_data)  # 原始图片
        figs[i, 0].axes.get_xaxis().set_visible(False)  # 去掉x轴
        figs[i, 0].axes.get_yaxis().set_visible(False)  # 去掉y轴
        figs[i, 1].imshow(img_label)                    # 标签
        figs[i, 1].axes.get_xaxis().set_visible(False)  # 去掉x轴
        figs[i, 1].axes.get_yaxis().set_visible(False)  # 去掉y轴
        figs[i, 2].imshow(pred)                         # 模型输出结果
        figs[i, 2].axes.get_xaxis().set_visible(False)  # 去掉x轴
        figs[i, 2].axes.get_yaxis().set_visible(False)  # 去掉y轴

    # 在最后一行图片下面添加标题
    figs[num_image-1, 0].set_title("Image", y=-0.2*(10/img_size))
    figs[num_image-1, 1].set_title("Label", y=-0.2*(10/img_size))
    figs[num_image-1, 2].set_title("segnet", y=-0.2*(10/img_size))
    plt.show()

if __name__ == '__main__':

    #see whether gpu can be used
    device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
    print("device: "+str(device))

    #load dataset
    train_dataset = NucleusDataset("test")
    bs=15      #batch size
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, drop_last=True, shuffle=True)

    """ model = DeconvNet()
    learing_rate = 0.01
    now_epoch = 8
    for i in range(9):
        train(model, train_loader, torch.optim.Adam(model.parameters(), lr=learing_rate), nn.BCELoss(), 8, "_bce_e"+str(now_epoch))
        torch.save(model, "C:/Codeing area/Python/DeconvNet/Model/DeconvNet_bce_e"+str(now_epoch)+".sav")
        learing_rate/=2
        now_epoch+=8 """

    model = torch.load("C:/Codeing area/Python/DeconvNet/Model/DeconvNet_bce_e72.sav")
    train(model, train_loader, torch.optim.Adam(model.parameters(), lr=0.00001953), nn.BCELoss(), 1, "test")
