import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from math import log

def reduce(xFile, n, d, cutoff):
    xf = open(xFile)
    inDocFreq = [d]
    reducedIndx = {}

    for line in xf:
        termInfo = line.split(' ')
        inDocFreq[int(termInfo[1])] += 1

    index = 0
    for i in range(d):
        if(math.log(n / (1 + inDocFreq[i])) > cutoff):
            reducedIndx[d] = index
            index += 1

    close(xf)
    return reducedIndx

def handleInX(xFile, n, d, ri):
    xf = open(xFile)
    x = np.zeros((n,len(ri)))

    for line in xf:
        termInfo = line.split(' ')
        if(int(termInfo[1]) in ri):
            x[int(termInfo[0]), ri[int(termInfo[1])]] = int(termInfo[2])

    return x

def def_model():
    model = Sequential()
    model.add(Dense(100, input_dim=75000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

trainXf = "/proj_data/task4/train.sparseX"
configf = "/proj_data/task4/task4.config"
cf = open(conFile)
ntrn = int(cf.readline(1).split(' ')[1])
d = int(cf.readline(3).split(' ')[1])
close(cf)

redMap = reduce(trainXf, ntrn, d, 0.1)
newX = handleInX(trainXf, ntrn, d, redMap)
print(newX.shape())

#seed = 9
#np.random.seed(seed)
#estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbos=0)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)


# space
