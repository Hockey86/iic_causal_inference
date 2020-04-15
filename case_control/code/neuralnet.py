#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:22:58 2019

@author: harshparikh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self,inputSize,outputSize,hiddenLayers,isOutputPositive=0):
        super(Net, self).__init__()
        #size of hiddenLayers array in n_layers - 2
        # an affine operation: y = Wx + b
        self.isOutputPositive = isOutputPositive
        self.sizelayers = [inputSize]+hiddenLayers+[outputSize]
        self.n_layers = len(self.sizelayers)
        self.namelayers = ['layer'+str(i) for i in range(0,self.n_layers-1)]
        for i in range(0,self.n_layers-1):
            exec('self.layer%d = nn.Linear(self.sizelayers[%d], self.sizelayers[%d+1])'%(i, i, i)) 

    def forward(self, x):
        d = {0:x}
        for i in range(0,self.n_layers-2):
            exec('d[ %d ] = F.relu( self.layer%d( d[%d] ) )'%(i+1, i, i))
        if self.isOutputPositive==1:
            exec('d[ %d ]  = F.relu( self.layer%d( d[ %d ] ) )'%(self.n_layers-1, self.n_layers-2, self.n_layers-2))
        elif self.isOutputPositive==-1:
            exec('d[ %d ]  = -F.relu( self.layer%d( d[ %d ] ) )'%(self.n_layers-1, self.n_layers-2, self.n_layers-2))
        else:
            exec('d[ %d ]  = self.layer%d( d[ %d ] )'%(self.n_layers-1, self.n_layers-2, self.n_layers-2))
        return d[self.n_layers-1]
    
    def fit(self, x, y=None, loss=None, learningrate=0.01, losscutoff=torch.Tensor([float('inf')]).sum(), verbose=False ):
        yhat = self( x )
        if loss is not None:
            self.zero_grad()
            if (not torch.isnan(loss)) and (loss<losscutoff):
                if verbose:
                    print(loss)
                optimizer = optim.SGD( self.parameters(), lr=learningrate )
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                return 0
        elif y is not None:
            criterion = nn.MSELoss()
            loss = criterion( yhat, y )
            self.zero_grad()
            if (not torch.isnan(loss)) and (loss<losscutoff):
                if verbose:
                    print((y,yhat,loss))
                optimizer = optim.SGD( self.parameters(), lr=learningrate )
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                return 0
        return -1


#net = Net(10,5,[8,9])
#net2 = Net(5,1,[8,9])
#print(net)
#x = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float)
#X = torch.stack([x,x,x,x])
#y = torch.tensor([0.0, 0.0, 1., 1., 1.])
#Y = torch.stack([y,y,y,y])
#
#print(net.forward(X))
#
#net.fit(x=X, loss=-torch.sum(net2(net(X))[:,0]) )
#print(net2(net(X)))