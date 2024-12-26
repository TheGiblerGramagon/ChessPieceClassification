from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import v2
import cv2 as cv
import json
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

#This is important for loading data into the CNN
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


batch_size = 256 #Number of trained images before backpropagation. (Lowers Variance)

#Sets up the datasets using ChessPieceDataset
testSE = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\pnpTest.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\TestFinal")
trainSE = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\pnpTrain.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\TrainFinal")
valSE = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\pnpVal.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\ValFinal")

testSC = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\cncTest.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\TestFinal")
trainSC = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\cncTrain.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\TrainFinal")
valSC = ChessPieceDataset(csv_file=r"C:\Users\graha\New folder\cncVal.csv",root_dir=r"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\ValFinal")

#Sets up the dataloaders for transfer learning
transferPNP = DataLoader(dataset=ChessPieceDataset(csv_file = r"C:\Users\graha\New Folder\transferPNP.csv", root_dir=r"C:\Users\graha\OneDrive\Desktop\RLISimple"), batch_size = 32, shuffle = True)
transferCNC = DataLoader(dataset=ChessPieceDataset(csv_file = r"C:\Users\graha\New Folder\transferCNC.csv", root_dir=r"C:\Users\graha\OneDrive\Desktop\RLISimple"), batch_size = 32, shuffle = True)

#Splits the dataset into desired sizes
testSE, _ = torch.utils.data.random_split(testSE, (8096, len(testSE) - 8096))
#trainSE, _ = torch.utils.data.random_split(trainSE, (32384, len(trainSE) - 32384))
valSE, _ = torch.utils.data.random_split(valSE, (1024, len(valSE) - 1024))

testSC, _ = torch.utils.data.random_split(testSC, (8096, len(testSC) - 8096))
#trainSC, _ = torch.utils.data.random_split(trainSC, (32384, len(trainSC) - 32384))
valSC, _ = torch.utils.data.random_split(valSC, (1024, len(valSC) - 1024))

#sets up the dataloaders
testLE = DataLoader(dataset=testSE, batch_size=batch_size, shuffle=True)
trainLE = DataLoader(dataset=trainSE, batch_size=batch_size, shuffle=True)
valLE = DataLoader(dataset=valSE, batch_size=batch_size, shuffle=True)

testLC = DataLoader(dataset=testSC, batch_size=batch_size, shuffle=True)
trainLC = DataLoader(dataset=trainSC, batch_size=batch_size, shuffle=True)
valLC = DataLoader(dataset=valSC, batch_size=batch_size, shuffle=True)

#Output for user
print("Training set C", len(trainLC))
print("Test set C:", len(testLC))
print("Training set E: ", len(trainLE))
print("Test set: E", len(testLE))
globalPathC = "cncModel.pth"
globalPathE = "pnpModel.pth"

# Defines a convolution neural network (Are the pieces there or not?)
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

#Defines the white or black network
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

# Instantiate the models
modelE = NetworkExist()
modelC = NetworkColor()

#Defines loss functions, optimizer, and schedulers. (Schedulers are only used in transfer learning)
loss_fnE = nn.CrossEntropyLoss()
loss_fnC = nn.CrossEntropyLoss()
optimizerE = Adam(modelE.parameters(), lr=0.0001)
optimizerC = Adam(modelC.parameters(), lr=0.0001)
schedulerE = lr_scheduler.ExponentialLR(optimizerE, .1)
schedulerC = lr_scheduler.ExponentialLR(optimizerC, .1)

# Function to save the model
def saveModel(type):
    if(type == "E"):
        torch.save(modelE.state_dict(), globalPathE)
    if(type == "C"):
        torch.save(modelC.state_dict(), globalPathC)
        
# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(type):
    print(f"testing accuracy {type}")
    accuracy = 0.0
    total = 0.0
    device = torch.device("cpu")
    if(type == "E"):
        modelE.eval()
        with torch.no_grad(): #DON'T UPDATE GRADS (Or accuracy goes down)
            for data in testLE:
                images, labels = data
                outputs = modelE(images.to(device)) #Predict lables
                _, predicted = torch.max(outputs.data, 1) #Choose the most positive one (I think)
                total += labels.size(0)
                accuracy += (predicted == labels.to(device)).sum().item()
    if(type == "C"):
        modelC.eval()
        with torch.no_grad():
            for data in testLC:
                images, labels = data
                outputs = modelC(images.to(device)) #Predict Lables
                _, predicted = torch.max(outputs.data, 1) #Choose the most positive one (I think)
                total += labels.size(0)
                accuracy += (predicted == labels.to(device)).sum().item()
    
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train(type, num_epochs, loader, transfer): #Transfer is a boolean that turns features on or off if we are transfer learning.
    
    best_accuracy = 0.0

    device = torch.device("cpu")
    
    #EXISTING MODEL
    if(type == "E"):
        modelE.to(device)
        for epoch in range(num_epochs):  # loop over the dataset epoch times
            modelE.train()
            running_loss = 0.0
            print("epoch starting E")
            for i, (images, labels) in enumerate(loader): 
                
                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                
                optimizerE.zero_grad() # zero the gradients
                outputs = modelE(images) # predict classes
                loss = loss_fnE(outputs, labels) # compute loss
                loss.backward() #Adjust NN
                optimizerE.step() #Tweak your adjustments to optimize training
                
                #print statistics for every 10 images
                running_loss += loss.item()
                if i % 100 == 0:    
                    print('[%d, %5d] loss: %.4f' %(epoch + 1, i, running_loss/100))
                    running_loss = 0.0
            if not(transfer):    
                accuracy = testAccuracy("E")
                newLr = optimizerE.param_groups[0]["lr"]
                print(f'For epoch {epoch+1} the test accuracy over the whole test set is {accuracy}. The learning rate is {newLr}.')
            
                #save best accuracy
                if accuracy > best_accuracy:
                    saveModel("E")
                    best_accuracy = accuracy
        
        if(transfer):
            saveModel("E")
        
    #COLORS MODEL (Same thing as above, but with colors model)
    if type == "C":
        modelC.to(device)
        for epoch in range(num_epochs):
            modelC.train()
            running_loss = 0.0
            print("epoc starting C")
            for i, (images, labels) in enumerate(loader):
                
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                
                optimizerC.zero_grad()
                outputs = modelC(images)
                loss = loss_fnC(outputs, labels)
                loss.backward()
                optimizerC.step()
                
                running_loss += loss.item()
                if i % 100 == 0:    
                    print('[%d, %5d] loss: %.4f' %(epoch + 1, i, running_loss/100))
                    running_loss = 0.0
            if not(transfer):    
                accuracy = testAccuracy("C")
                newLr = optimizerE.param_groups[0]["lr"]
                print(f'For epoch {epoch+1} the test accuracy over the whole test set is {accuracy}. The learning rate is {newLr}.')
            
                #save best accuracy
                if accuracy > best_accuracy:
                    saveModel("C")
                    best_accuracy = accuracy
        
        if(transfer):
            saveModel("C")

def testBatch(type):
    if type == "E":
        print("testing batch E")
        modelE.eval() #Switch the model to evaluating mode (Important)
        with torch.no_grad():
            for i, (images, labels) in enumerate(valLE): #I believe the "I" is the line number in the CSV file
                outputs = modelE(images)
                _, predicted = torch.max(outputs, 1)
                count = 0
                tot = 0
                for labels, pred in zip(labels, predicted):
                    if labels == pred:
                        count += 1
                    tot += 1
                print(float(count)/float(tot)) #Print accuracy over batch size
                print(f"{count}{tot}")
    if type == "C": #Same thing as test Batch E, but with C
        print("testing batch C")
        modelC.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(valLC):
                outputs = modelC(images)
                _, predicted = torch.max(outputs, 1)
                count = 0
                tot = 0
                for labels, pred in zip(labels, predicted):
                    if labels == pred:
                        count += 1
                    tot += 1
                print(float(count)/float(tot))
                print(f"{count}{tot}")

def transfer(type):
    if type == "E":
        for param in modelE.parameters(): #Make the model not adjustable
            param.requires_grad = False
        for param in modelE.fc1.parameters(): #Make fc1 adjustable
            param.requires_grad = True
        for param in modelE.fc2.parameters(): #Make fc2 adjustable
            param.requires_grad = True
        train("E", 100, transferPNP, True)
        for param in modelE.parameters(): #Everything is adjustable again
            param.requires_grad = True
        schedulerE.step() #Divide learning rate by 10
        train("E", 100, transferPNP, True)
        
        
    if type == "C": #Same as type == "E"
        for param in modelC.parameters():
            param.requires_grad = False
        for param in modelC.fc1.parameters():
            param.requires_grad = True
        for param in modelC.fc2.parameters():
            param.requires_grad = True
        train("C", 100, transferCNC, True)
        for param in modelC.parameters():
            param.requires_grad = True
        schedulerC.step()
        train("C", 100, transferCNC, True)

if __name__ == "__main__":
        train("E", 1, trainLE, False) #Train the model on artificial dataset
        train("C", 1, trainLC, False)
        torch.save(modelC.state_dict(), "Csave.pth") #Make backups
        torch.save(modelE.state_dict(), "Esave.pth") 
        transfer("E") #Transfer to real dataset
        transfer("C")
        testBatch("E") #Check to see that the accuracy lowered slightly on artificial dataset
        testBatch("C")