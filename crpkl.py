import pickle
neg=[]
pos=[]
with open('Negative.txt','r') as f:
	for line in f.readlines():
        		line= line[:-1]
        		neg.append(str(line))		
with open('Positive.txt', 'r') as f:
        for line in f.readlines():
        		line= line[:-1]
        		pos.append(str(line))

with open('neg.pkl','wb') as f:
	pickle.dump(neg,f)
with open('pos.pkl','wb') as f:
	pickle.dump(pos,f)
with open('pos.pkl','rb') as f:
        l=pickle.load(f)
for x in l:
	print(x)
