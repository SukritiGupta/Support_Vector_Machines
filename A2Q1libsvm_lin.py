import numpy as np
import math
from svmutil import *
import time

datax = np.genfromtxt('train.csv', delimiter=',')

s=time.time()
dig=datax[:,784]
# m=dig.shape[0]
d4=datax[np.ix_(dig==4,)]
d5=datax[np.ix_(dig==5,)]
# print(d4.shape)
d4=np.delete(d4,784,1)
d5=np.delete(d5,784,1)
d4=d4/255.0
d5=d5/255.0
X=np.concatenate((d4,d5), axis=0)
n4=d4.shape[0]
n5=d5.shape[0]
Y=np.concatenate((np.ones(n4)*(-1),np.ones(n5)),axis=0)
prob=svm_problem(Y,X)

s=time.time()
m=svm_train(prob,'-t 0 -c 1 -b 1')
end=time.time()
print(end-s)
datax = np.genfromtxt('test.csv', delimiter=',')

tdig=datax[:,784]
# m=dig.shape[0]
td4=datax[np.ix_(tdig==4,)]
td5=datax[np.ix_(tdig==5,)]
# print(d4.shape)
td4=np.delete(td4,784,1)
td5=np.delete(td5,784,1)
td4=td4/255.0
td5=td5/255.0
tX=np.concatenate((td4,td5), axis=0)
tn4=td4.shape[0]
tn5=td5.shape[0]
tY=np.concatenate((np.ones(tn4)*(-1),np.ones(tn5)),axis=0)

p_label, p_acc, p_val = svm_predict(tY, tX, m)
print(p_acc)


# .*.*
# optimization finished, #iter = 2441
# nu = 0.004359
# obj = -6.975490, rho = -1.334623
# nSV = 132, nBSV = 0
# Total nSV = 132
# .*.*
# optimization finished, #iter = 2670
# nu = 0.004650
# obj = -7.440754, rho = -1.308209
# nSV = 147, nBSV = 0
# Total nSV = 147
# .*.*
# optimization finished, #iter = 2909
# nu = 0.005086
# obj = -8.137393, rho = -1.166223
# nSV = 144, nBSV = 0
# Total nSV = 144
# .*.*
# optimization finished, #iter = 2606
# nu = 0.004506
# obj = -7.210552, rho = -1.321723
# nSV = 138, nBSV = 0
# Total nSV = 138
# .*.*
# optimization finished, #iter = 2637
# nu = 0.004419
# obj = -7.070901, rho = -1.325996
# nSV = 142, nBSV = 0
# Total nSV = 142
# ..*.*
# optimization finished, #iter = 3634
# nu = 0.005158
# obj = -10.315969, rho = -1.456484
# nSV = 150, nBSV = 0
# Total nSV = 150
# 3.7267205715179443
# Model supports probability estimates, but disabled in predicton.
# Accuracy = 98.666% (1849/1874) (classification)
# (98.66595517609392, 0.05336179295624333, 0.9472437480073788)
