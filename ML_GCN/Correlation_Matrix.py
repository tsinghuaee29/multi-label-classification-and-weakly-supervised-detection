import os
import numpy as np

def C_M_construct(p=0.2, t=0.4,cls_num=20, path='/home/tsinghuaee29/annotations.txt'):
	# Construct Correlation Matrix for GCN
	File = open(path,'r')	
	M = np.zeros([cls_num, cls_num])
	for line in File:
		if line[:4] in ['2009','2010','2011']:
			continue
		labels = line.strip().split(' ')[1:]
		for l1 in labels:
			for l2 in labels:
				M[int(l1),int(l2)] += 1
	N = [ M[k,k] for k in range(cls_num)]
	N = np.array(N)
	
	P = np.zeros([cls_num, cls_num])
	for c in range(cls_num):
		P[c,:] = M[c,:] / N[c]
		M[c,c] = 0
	
	A = np.where(P>=t,1,0)
	
	A_ = np.zeros([cls_num, cls_num])
	for cls_i in range(cls_num):
		for cls_j in range(cls_num):
			if cls_i != cls_j:
				A_[cls_i, cls_j] = p * (sum(A[cls_i ,:]) - A[cls_i, cls_i])
			else:
				A_[cls_i, cls_j] = 1 - p
	Pickle = {'nums':N,'adj':M}
	import pickle
	Output = open('./data/voc/voc_adj2012_test.pkl','wb')
	pickle.dump(Pickle, Output)
	Output.close()
	return A_
	
if __name__ == '__main__':
	C_M_construct()


	
