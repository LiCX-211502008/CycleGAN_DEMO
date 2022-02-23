import numpy
from include import *
import os
import random

class imgs(Dataset):
    def __init__(self,path_X,path_Y,imgsize) -> None:
        self.size = imgsize
        super().__init__()
        self.filelist_X = []
        self.filelist_Y = []
        for i in os.listdir(path_X):
            if not os.path.isdir(str(i)):
                self.filelist_X.append(path_X + "\\" + i)
        for i in os.listdir(path_Y):
            if not os.path.isdir(str(i)):
                self.filelist_Y.append(path_Y + "\\" + i)
    
    def __len__(self):
        return min(len(self.filelist_X),len(self.filelist_Y))

    def __getitem__(self, index):
        Index = index
        if len(self.filelist_X)<len(self.filelist_Y):
            X = cv.imread(self.filelist_X[Index])
            Y = cv.imread(self.filelist_Y[random.randint(0,len(self.filelist_Y))])
            X,Y = cv.resize(X,dsize = self.size),cv.resize(Y,dsize = self.size)
            X,Y = numpy.transpose(X,(2,0,1)),numpy.transpose(Y,(2,0,1))
            X,Y = torch.from_numpy(X).float(),torch.from_numpy(Y).float()
            return X,Y
        else :
            X = cv.imread(self.filelist_X[random.randint(0,len(self.filelist_Y))])
            Y = cv.imread(self.filelist_Y[Index])
            X,Y = cv.resize(X,dsize = self.size),cv.resize(Y,dsize = self.size)
            X,Y = numpy.transpose(X,(2,0,1)),numpy.transpose(Y,(2,0,1))
            X,Y = torch.from_numpy(X).float(),torch.from_numpy(Y).float()
            return X,Y