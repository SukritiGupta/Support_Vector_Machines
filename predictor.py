import numpy as np

work=np.genfromtxt('work_copy.csv', delimiter=' ',dtype=int)

pred = np.genfromtxt('test.csv', delimiter=',')
tdig=pred[:,784]

tot=tdig.shape[0]

cm=np.zeros((10,10),dtype=int)
correct=0

for i in range(tot):
	actual=(int)(tdig[i])
	print(work[i])
	print(work[i].shape)
	b=np.bincount(work[i])
	b=np.flip(b,axis=0)
	prediction=np.argmax(b)
	prediction=9-prediction
	if prediction==actual:
		correct+=1
	cm[prediction][actual]=cm[prediction][actual]+1

print(cm)
print(correct/tot)
