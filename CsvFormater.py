import numpy as np
import sys
import os
import csv
import math
# default format can be changed as needed
def createFileList(myDir, format='.png'):
    fileList = []
    labels = []
    names = []
    keywords = {'B':0,'W':1, 'E':2}
    for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
            for keyword in keywords:
                if keyword == name[9]:
                    labels.append(keywords[keyword]) 
                else:
                    continue
            names.append(name)
    return fileList, labels, names

def createFileListNoLables(myDir, format='.png'):
    fileList = []
    print(myDir)
    labels = []
    names = []
    batch_size = 10
    for files in os.walk(myDir, topdown=True):
        for i in range(120):
            name = "ChessPiece" + str(i) + ".png"
            labels.append(math.trunc(i / batch_size))
            fullName = os.path.join(r"C:\Users\graha\Downloads\TestImages", name) #r"C:\Users\graha\Downloads\TestImages\ChessPiece" + str(i) + r".jpg"
            fileList.append(fullName)
            names.append(name)
    return fileList, labels, names
    
# load the original image
#myFileList, labels, names  = createFileList(r"C:\Users\graha\Downloads\TestImages")
myFileList, labels, names = createFileList(r"C:\Users\graha\OneDrive\Desktop\RLISimple")
i = 0
for file in myFileList:
    value = np.append(names[i],labels[i])
    i += 1
    with open(r"C:\\Users\graha\transferCNC.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(value)
        
#"C:\Users\graha\OneDrive\Desktop\Complete Chess Piece Dataset\ValFinal\B0024018.png"
        