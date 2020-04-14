
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#train.csv is the dataset
df = pd.read_csv('train.csv')
print(df.shape)

print(df.columns)
df.head(n=5)
#pixel representation

data = df.values
print(data.shape)
print(type(data))
X = data[:,1:]
Y = data[:,0]

print(X.shape,Y.shape)

split = int(0.8*X.shape[0])
print(split)

X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]

#spliting data set into training and testing dat
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Visualise SOme Samples
def drawImg(sample):
    img = sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()
    
drawImg(X_train[0])
print(Y_train[0])
    
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        #updating distance for each query
        vals.append((d,Y[i]))
        
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    vals = np.array(vals)
    #print(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred

#making predictions
pred = knn(X_train,Y_train,X_test[1])
print(int(pred)) 

rawImg(X_test[1])
print(Y_test[1])
#drawing araw image out of the prediction
