import random
import numpy as np
from scipy.misc import derivative
import scipy.io as sio

#NN Loss Function

#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def nnloss(x,t,dzdy):
    a = int(len(x))
    instanceWeights =np.ones((a,1), np.float32)
 #   np. ones(size(x))
  
    res = x-t
    if(dzdy==0):
       
        y = (1/2) * np.matrix.transpose(instanceWeights) * (res**2)
        
    else:
        y=res
    return y


#Sigmoid Activation Function

def sigm(X):
    O = 1 / (1 + np.exp(-X))
    return O


def sigmDerivative(X):
    Output = 1 / (1 + np.exp(-X));
    Output = Output*(1-Output);
    return Output


#Exp 1
#simple XOR problem with constant coefficients

#XOR input for x1 and x2
input = [[0,0],[0,1],[1,0],[1,1]]
#Desired output of XOR
groundTruth = [0,1,1,0]
#Initialize the bias
bias = [(-1 , -1 , -1)]
#Learning coefficient
coeff = 0.7
#Number of learning iterations
iterations = 10000
#Calculate weights randomly using seed.
# MATLAB : rand('state',sum(100*clock))
randomValue = random.random() * 1
#weights = -1 +2.*rand(3,3);
weights = np.ones((3,3), np.float32)
# weights(1,:) = 0.1
weights[0][0:3] = 0.1
weights[1][0:3] = 0.2
weights[2][0:3] = 0.3

for i in range(iterations):
    out = np.zeros((4,1), np.float32)
    numIn = len(input)
    for j in range(numIn):
        H1 = bias[0][0] * weights[0,0] + input[j][0] * weights[0,1] + input[j][1] * weights[0,2]
        x2 = (0.0,0.0)
        
        y=list(x2)
        
        y[0]=sigm(H1)
        
        H2 = bias[0][1] * weights[1,0] + input[j][0] * weights[1,1] + input[j][1] * weights[1,2]
        
        y[1]=sigm(H2)

        x2 = tuple(y)
        
        x3_1 = bias[0][2] * weights[2,0] + x2[0] * weights[2,1] + x2[1] * weights[2,2]
        
        out[j] = sigm(x3_1)
        
        delta3_1 = out[j]*(1-out[j])*(groundTruth[j]-out[j])
        
        delta2_1 = x2[0]*(1-x2[0])*weights[2,1]*delta3_1
        delta2_2 = x2[1]*(1-x2[1])*weights[2,2]*delta3_1
        
        for k in range(3):
            if (k == 0):
                weights[0,k] = weights[0,k] + coeff * bias[0][0] * delta2_1
                weights[1,k] = weights[1,k] + coeff * bias[0][1] * delta2_2
                weights[2,k] = weights[2,k] + coeff * bias[0][2] * delta3_1
            else:
                weights[0,k] = weights[0,k] + coeff * input[j][k-1] * delta2_1
                weights[1,k] = weights[1,k] + coeff * input[j][k-1] * delta2_2
                weights[2,k] = weights[2,k] + coeff * x2[k-1] * delta3_1;

#test the code

input = [[0,0],[0,1],[1,0],[1,1]]
# Desired output of XOR
groundTruth = [0,1,1,0]
out = np.zeros((4,1), np.float32)
numIn = len(input)

for j in range(numIn):
    H1 = bias[0][0] * weights[0,0] + input[j][0] * weights[0,1] + input[j][1] * weights[0,2]
    x2 = (0.0,0.0)
    y=list(x2)
    y[0]=sigm(H1)
    
    H2 = bias[0][1] * weights[1,0] + input[j][0] * weights[1,1] + input[j][1] * weights[1,2]
    
    y[1]=sigm(H2)
    
    x2 = tuple(y)
    
    x3_1 = bias[0][2] * weights[2,0] + x2[0] * weights[2,1] + x2[1] * weights[2,2]
    
    out[j] = sigm(x3_1)



#Exp 2

def getIrisData():
    f=sio.loadmat('irisData.mat')
    X = f['X']
    trainingSet = np.zeros((5,120), np.float32)
    testSet =  np.zeros((5,30), np.float32)
    y1 = np.zeros((120,3), np.float32)
    y2 = np.zeros((30,3), np.float32)
    
    trainingSet[0:4,0:40] = np.matrix.transpose(X[0:40][:])
    trainingSet[0:4,40:80] = np.matrix.transpose(X[50:90][:])
    trainingSet[0:4,80:120] = np.matrix.transpose(X[100:140][:])
    trainingSet[4,:] = 1
    
    testSet[0:4,0:10] = np.matrix.transpose(X[40:50][:])
    testSet[0:4,10:20] = np.matrix.transpose(X[90:100][:])
    testSet[0:4,20:30] = np.matrix.transpose(X[140:150][:])
    testSet[4:] = 1
    
    y1=np.zeros((120,3), np.float32)
    
    for i in range(40):
        y1[i,:] = [1,0,0]
        y1[i+40,:] = [0,1,0]
        y1[i+80,:] = [0,0,1]
        
        y2=np.zeros((30,3), np.float32)
        
    for i in range(10):
        y2[i,:] = [1,0,0]
        y2[i+10,:] = [0,1,0]
        y2[i+20,:] = [0,0,1]

    return trainingSet, testSet, y1, y2  



[ trainingSet, testSet, y1, y2 ] = getIrisData()

input = trainingSet;

groundTruth = y1;

coeff = 0.1;

iterations = 1;

randomValue = random.random() * 1

inputLength = len(input)

hiddenN=5
outputN=len(groundTruth[0,:]);
num_layers=3;
tol=0.1;
 

stack = []

for k in range(num_layers-1):
    if (k==0):
        stack.append([])
        stack.append([])
        w = np.random.rand(hiddenN, inputLength)
        b = np.random.rand(hiddenN, 1)
        stack[k].append(w)
        stack[k].append(b)
        w = stack[k][0]
        b = stack[k][1]
    elif (k==num_layers-2):
        stack.append([])
        stack.append([])
        w = np.random.rand(outputN, hiddenN)
        b = np.random.rand(outputN, 1)
        stack[k].append(w)
        stack[k].append(b)
        w = stack[k][0]
        b = stack[k][1]
    else:
        stack.append([])
        stack.append([])
        w = np.random.rand(hiddenN, hiddenN)
        b = np.random.rand(hiddenN, 1)
        stack[k].append(w)
        stack[k].append(b)
        w = stack[k][0]
        b = stack[k][1]


outputStack  = ([[-1],[-1],[-1]])
gradStack = []
for i in range(iterations):
    err=0
    for j in range(len(y1[:,0])):
        inputs=input[:,j]
        outputStack[0][0]=inputs
        
        for k in range(num_layers-1):
            if(k!=0):
                matrices = np.matrix.dot(stack[1][0],np.matrix.transpose(outputStack[k][0])) + (np.matrix.transpose(stack[k][1]* (-1)))
            else:
                matrices = np.matrix.dot(stack[0][0],outputStack[k][0]) + (np.matrix.transpose(stack[k][1]* (-1)))
            outputStack[k+1][0]=matrices
            outputStack[k+1][0] = sigm(outputStack[k+1][0])
        p = outputStack[2][0]
        epsilon = nnloss(groundTruth[j,:], p, 1)
        cost = nnloss(groundTruth[j,:], p, 0)
        err = err+cost;


        #for k in range((num_layers-1),0,-1):
        #    gradStack.append([])
        #    epsilon = outputStack[k]* ((1-outputStack[k])*epsilon)
        #    gradStack[k-1].append(epsilon)
        #    w = stack[k][0]
        #    epsilon = stack[k-1][0]*gradStack[k-1][0]




# E N D 