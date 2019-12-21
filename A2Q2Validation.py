import numpy as np
import math
from svmutil import *
import time

datax = np.genfromtxt('train.csv', delimiter=',')


Y=datax[:,784]
m=Y.shape[0]
print(Y.shape)
vs=math.floor(m/10)
print(vs)
datax=np.delete(datax,784,1)
datax=datax/255.0
X=datax

idx=np.random.choice(m, m-vs, replace=False)
trainY=Y[idx]
trainX=X[idx, :]

vY = Y[[i for i in range(m) if i not in idx]]
vX = X[[i for i in range(m) if i not in idx],:]

prob=svm_problem(trainY,trainX)

m1=svm_train(prob,'-g 0.05 -t 2 -c 0.00001 -b 1')
m2=svm_train(prob,'-g 0.05 -t 2 -c 0.001 -b 1')
m3=svm_train(prob,'-g 0.05 -t 2 -c 1 -b 1')
m4=svm_train(prob,'-g 0.05 -t 2 -c 5 -b 1')
m5=svm_train(prob,'-g 0.05 -t 2 -c 10 -b 1')

print("done")

tdata = np.genfromtxt('test.csv', delimiter=',')
tY=tdata[:,784]
tdata=np.delete(tdata,784,1)
tdata=tdata/255.0

print("test!!!!")


p_labelt1, p_acct1, p_valt1 = svm_predict(tY, tdata, m1)
print(p_acct1)

p_labelt2, p_acct2, p_valt2 = svm_predict(tY, tdata, m2)
print(p_acct2)
p_labelt3, p_acct3, p_valt3 = svm_predict(tY, tdata, m3)
print(p_acct3)
p_labelt4, p_acct4, p_valt4 = svm_predict(tY, tdata, m4)
print(p_acct4)
p_labelt5, p_acct5, p_valt5 = svm_predict(tY, tdata, m5)
print(p_acct5)


print("validation!!!!")


p_labelt1, p_acct1, p_valt1 = svm_predict(vY, vX, m1)
print(p_acct1)

p_labelt1, p_acct1, p_valt1 = svm_predict(vY, vX, m2)
print(p_acct1)

p_labelt1, p_acct1, p_valt1 = svm_predict(vY, vX, m3)
print(p_acct1)

p_labelt1, p_acct1, p_valt1 = svm_predict(vY, vX, m4)
print(p_acct1)

p_labelt1, p_acct1, p_valt1 = svm_predict(vY, vX, m5)
print(p_acct1)