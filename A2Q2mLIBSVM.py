import numpy as np
import math
from svmutil import *
import time

datax = np.genfromtxt('train.csv', delimiter=',')

s=time.time()
Y=datax[:,784]
datax=np.delete(datax,784,1)
datax=datax/255.0
X=datax
prob=svm_problem(Y,X)

s=time.time()
m=svm_train(prob,'-g 0.05 -t 2 -c 1 -b 1')
end=time.time()
print(end-s)


tdata = np.genfromtxt('test.csv', delimiter=',')
tY=tdata[:,784]
tdata=np.delete(tdata,784,1)
tdata=tdata/255.0
p_label, p_acc, p_val = svm_predict(tY, tdata, m)
print("test!!!!")
print(p_acc)


p_label1, p_acc1, p_val1 = svm_predict(Y, X, m)
print("train!!!!")
print(p_acc1)


# 1242.7007381916046
# Model supports probability estimates, but disabled in predicton.
# Accuracy = 97.23% (9723/10000) (classification)
# test!!!!
# (97.23, 0.5448, 0.9360124703766993)
# Model supports probability estimates, but disabled in predicton.
# Accuracy = 99.92% (19984/20000) (classification)
# train!!!!
# (99.92, 0.0154, 0.9981340911956579)
