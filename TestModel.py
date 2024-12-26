from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
#import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import v2
import cv2 as cv
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pygame as pyg

#*************************************************************************************
#
# CLICK ON THE FOUR CORNERS OF THE CHESS BOARD AND THE MODEL WILL PREDICT COLORS AND PIECES
#
#*************************************************************************************


class ChessPieceDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = read_image(img_path)
        label = self.annotations.iloc[index, 1]
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = torchvision.transforms.RandomCrop(size=(196,196), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')(image)
        image = v2.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.25)(image)
        return image, label
#Setup CNN
class NetworkExist(nn.Module):
    def __init__(self):
        super(NetworkExist, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(30976, 256)
        self.dropout1 = nn.Dropout(.3)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

class NetworkColor(nn.Module):
    def __init__(self):
        super(NetworkColor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(30976, 256)
        self.dropout1 = nn.Dropout(.3)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def circle(point):
    pyg.draw.circle(screen, red, (point), 10)

#Resizes the chessboard to fit the screen
def chessBoard():
    img = cv.imread(r"C:\Users\graha\Downloads\RealLifeTest.jpg")
    scale = 900.0 / float(len(img))
    height = int(scale * len(img))
    width = int(scale * len(img[0]))
    img = cv.resize(img, (width,height))
    cv.imwrite("currentImg.png", img)
    screen.blit(pyg.image.load('currentImg.png'), (0, 0))

#Draws the square type over a chess square
def square(color, location):
    s = pyg.Surface((100,100))  
    s.set_alpha(128)
    if(color[0] == 1):
        if(color[1] == torch.tensor(0)):
            s.fill((255,255,255))
            screen.blit(s, location)
        else:
            s.fill((0,0,0))
            screen.blit(s, location)

#Test a photo
def runModel():
    image = read_image("croppedImg.png")
    image = v2.ToDtype(torch.float32, scale=True)(image)
    image = torchvision.transforms.RandomCrop(size=(196,196), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')(image)
    image.unsqueeze_(0) #removes extra dimension
    modelE.eval()
    modelC.eval()
    resultE = modelE(image)
    resultC = modelC(image)
    with torch.no_grad():
        _, predictedE = torch.max(resultE.data, 1)
        _, predictedC = torch.max(resultC.data, 1)
    return (predictedE, predictedC)

#Correct the image perspective
def  perspectiveCorrection(corners1, img):
    
    corners1 = np.array(corners1)
    corners2 = np.array([[50,50],[850,50],[850,850],[50,850]])
    
    H, _ = cv.findHomography(corners1, corners2)
    img1_warp = cv.warpPerspective(img, H, (900, 900))

    return img1_warp

#Initialize Models
modelE = NetworkExist()
inputPath = 'pnpModel.pth'
modelE.load_state_dict(torch.load(inputPath))

modelC = NetworkColor()
inputPath = 'cncModel.pth'
modelC.load_state_dict(torch.load(inputPath))

#Pygame window loop
pyg.init()
screen = pyg.display.set_mode((1900, 900)) #(W, H)
playerImg = pyg.image.load(r"C:\Users\graha\Downloads\RealLifeTest.jpg")
red = pyg.Color(255, 0, 0)
black = pyg.Color(0, 0, 0)
chessBoard()
corners = []
running = True

while running:
    for event in pyg.event.get(): #Loop pygame events
        if event.type == pyg.QUIT:
            running = False
        if event.type == pyg.MOUSEBUTTONDOWN:
            point = pyg.mouse.get_pos()
            circle(point)
            corners.append(point)
            pyg.display.update()
            if len(corners) == 4:
                screen.fill(black)
                cv.imwrite("currentImg.png", perspectiveCorrection(corners, cv.imread("currentImg.png")))
                
                screen.blit(pyg.image.load('currentImg.png'), (0, 0))
                img = cv.imread("currentImg.png")
                for i in range(8):
                    for j in range(8):
                        cv.imwrite("croppedImg.png", img[i * 100:i * 100 + 200,j * 100:j * 100 + 200])
                        square(runModel(),(j * 100 + 50, i * 100 + 50))  
    pyg.display.update()