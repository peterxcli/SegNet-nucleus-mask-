import torch
from torch.utils.data import Dataset, DataLoader
from  torch import nn
import cv2 as cv
import numpy as np
from PIL import Image
from os import listdir
from Net import DeconvNet

class NucleusDataset(Dataset):

    def __init__(self, tot):    #tot = train or test
        super().__init__()
        self.root = 'C:/Codeing area/Python/DeconvNet/Nucleus dataset/'
        self.allfolderlist = listdir(self.root)
        print("There are "+str(len(self.allfolderlist))+" data totally")
        self.folderlist = []

        self.tot = tot
        if self.tot == 'train':
          print("Get training dataset")
          for n in range(int(len(self.allfolderlist)/2)):
            self.folderlist.append(self.allfolderlist[n])
          print("There are "+str(len(self.folderlist))+" data in training dataset")

        elif self.tot == 'test':
          print("Get testing dataset")
          for n in range(int(len(self.allfolderlist)/2),int(len(self.allfolderlist))):
            self.folderlist.append(self.allfolderlist[n])
          print("There are "+str(len(self.folderlist))+" data in testing dataset")

        else:
            print("Choose train or test")

    def __len__(self):
        return len(self.folderlist)

    def __getitem__(self, index):

        foldername = self.folderlist[index]
        filename = foldername.split(".")[0]+".png"
        filepath = self.root+foldername+"/images/"+filename

        img = cv.imread(filepath)
        img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)
        img_np = np.array(img, dtype=np.float32)
        flat_img_np = np.empty(shape=(1,224,224))
        for x in range(224):
          for y in range(224):
              flat_img_np[0][x][y] = (img_np[x][y][0]+img_np[x][y][1]+img_np[x][y][2])/765

        sum = 0
        for x in range(224):
          for y in range(224):
              sum+=flat_img_np[0][x][y]
        flat_img_np = flat_img_np*0.5/sum*224*224

        outputpath = self.root+foldername+"/masks/"

        isfirst = True
        for objectpic in listdir(outputpath):
          obimg = cv.imread(outputpath+objectpic)
          obimg = cv.resize(obimg, (224, 224), interpolation=cv.INTER_LINEAR)
          if isfirst:
            obimg_np = np.array(obimg, dtype=np.float32)
            isfirst = False
          else:
            obimg_np += np.array(obimg, dtype=np.float32)
            obimg_np/2

        flat_obimg_np = np.empty(shape=(1,224,224))
        for x in range(224):
          for y in range(224):
            if obimg_np[x][y][0]==255:
              flat_obimg_np[0][x][y]=0
            else:
              flat_obimg_np[0][x][y]=1

        return flat_img_np, flat_obimg_np, img_np

def PP(prediction, groundtruth, image, step, filename, bs=15):
    image = image.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1*(prediction[:,:,:,:]>0.5)
    prediction = prediction*255
    groundtruth = groundtruth*255

    #image = image.astype('uint8')
    #prediction = prediction.astype('uint8')
    #groundtruth = groundtruth.astype('uint8')

    for i in range(bs):
        image3d = image[i]

        groundtruth3d = np.empty(shape=(3,224,224))
        prediction3d = np.empty(shape=(3,224,224))
        for j in range(3):
          groundtruth3d[j] = groundtruth[i][0]
          prediction3d[j] = prediction[i][0]

        groundtruth3d = np.transpose(groundtruth3d, (1, 2, 0))
        prediction3d = np.transpose(prediction3d, (1, 2, 0))

        input = Image.fromarray(np.uint8(image3d))
        label = Image.fromarray(np.uint8(groundtruth3d))
        output = Image.fromarray(np.uint8(prediction3d))

        cimage = np.concatenate((input, label))
        cimage = np.concatenate((cimage, output))

        cv.imwrite("C:/Codeing area/Python/DeconvNet/Test Compare Pic/pic"+filename+"_s"+str(step)+"_b"+str(i)+".jpg" , cimage)

    return

def IOU(prediction, groundtruth, bs=15):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1*(prediction[:,:,:,:]>0.5)

    intersection = 0
    union = 0
    for l in range(bs):
      for n in range(224):
        for m in range(224):
          if prediction[l][0][n][m]==0 and groundtruth[l][0][n][m]==0:
            intersection+=1

          if prediction[l][0][n][m]==0 or groundtruth[l][0][n][m]==0:
            union+=1

    iou_score = intersection / union  # 交集除以聯集

    return iou_score

def MAE(prediction, groundtruth, bs=15):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1*(prediction[:,:,:,:]>0.5)

    error = 0

    for l in range(bs):
      for x in range(224):
        for y in range(224):
          if prediction[l][0][x][y]!=groundtruth[l][0][x][y]:
            error+=1

    return error/224/224/bs

def Fmeasure(prediction, groundtruth, bs=15, b=1):
    prediction = prediction.cpu().detach().numpy()
    groundtruth = groundtruth.cpu().detach().numpy()

    prediction = 1*(prediction[:,:,:,:]>0.5)

    TP = 0
    FP = 0
    FN = 0

    for l in range(bs):
      for x in range(224):
        for y in range(224):
          if prediction[l][0][x][y]==0 and groundtruth[l][0][x][y]==0:
            TP+=1
          if prediction[l][0][x][y]==0 and groundtruth[l][0][x][y]==1:
            FP+=1
          if prediction[l][0][x][y]==1 and groundtruth[l][0][x][y]==0:
            FN+=1

    if (TP+FP)==0:
      precision=0
    else:
      precision = TP/(TP+FP)

    if (TP+FN)==0:
      recall=0
    else:
      recall = TP/(TP+FN)

    if precision+recall==0:
      return 0

    return ((1+b*b)*precision*recall)/(b*b*(precision+recall))