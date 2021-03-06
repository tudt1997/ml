import numpy as np
import csv
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

raw_data = open('wdbc.data', 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
data = list(reader)
n = len(data[0])
print n
data = np.array(data)
data[:, 2:] = preprocessing.scale(data[:, 2:])
print data
good = np.zeros([357, data.shape[1] - 2])
n_good = 0
bad = np.zeros([212, data.shape[1] - 2])
n_bad = 0

for i in data:
    if i[1] == 'B':
        good[n_good] = i[2:]
        n_good += 1
    else:
        bad[n_bad] = i[2:]
        n_bad += 1

print n_good
print n_bad
from sklearn.model_selection import train_test_split  

train_good_data, test_good_data = train_test_split(good, test_size = 0.1)
train_bad_data, test_bad_data = train_test_split(bad, test_size = 0.999)

train_data = np.append(train_good_data, train_bad_data, axis=0)
# train_data = preprocessing.scale(train_data)
test_data = np.append(test_good_data, test_bad_data, axis=0)
# test_data = preprocessing.scale(test_data)

print train_good_data.shape
print train_bad_data.shape
print test_good_data.shape
print test_bad_data.shape
print train_data.shape
print test_data.shape

# train_target = np.ones([train_data.shape[0]])
test_target = np.append(np.ones([test_good_data.shape[0]], dtype=int), -np.ones([test_bad_data.shape[0]], dtype=int))

best_nu = 0
best_gamma = 0
best_auc = 0
best_model = 0

for i in range(1, 20):
    for j in range(1, 20):
        nu = i * 0.01
        gamma = j * 0.01
        model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)  
        model.fit(train_data)

        # values_preds = model.predict(train_data)  
        # values_targs = train_target
        # f1_train = 100 * metrics.f1_score(values_targs, values_preds)

        values_preds = model.predict(test_data)
        values_targs = test_target
        auc_test = 100 * metrics.roc_auc_score(values_targs, values_preds)
        print("nu = %.2f, gamma = %.2f, auc = %.2f" % (nu, gamma, auc_test))
        if best_auc < auc_test:
            best_nu = nu
            best_gamma = gamma
            best_auc = auc_test
            best_model = model

print("Best: nu = %.2f, gamma = %.2f, auc_test = %.2f" % (best_nu, best_gamma, best_auc))
values_preds = best_model.predict(test_data)
values_targs = test_target
cm = sklearn.metrics.confusion_matrix(values_targs, values_preds)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("Confusion matrix: ")
print cm