import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

#hdf5 stuff
#5 seconds is approximately 500 rows
#charlotte data = 35500 rows
#alex = 32503 rows
#kevin = 31500 rows
alex = pd.read_csv("AlexLabelledData.csv")
charlotte = pd.read_csv("CharlotteLabelledData.csv")
kevin = pd.read_csv("KevinDataLabelled.csv")

#merging the 2 data files to allow for a dataset subgroup incluing everones collected data
#this will be used for testing and training
frames =[alex, charlotte, kevin]
merged = pd.concat(frames, ignore_index=True, sort=False)
data =merged.iloc[:, 0:-1]
labels = merged.iloc[:, -1]

#group in 5 second increments, use merged when all csv files are added
n = 500
chunkedData = [data[i:i+n] for i in range (0, len(data), n)]
chunkedLabels = [labels[i:i+n] for i in range (0, len(labels), n)]

X_train, X_test, Y_train, Y_test = \
    train_test_split(chunkedData, chunkedLabels, test_size=0.1, shuffle=True, random_state=0)

#writing to HDF5
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    hdf.create_dataset('datasetAlex', data=alex)
    hdf.create_dataset('datasetCharlotte', data=charlotte)
    hdf.create_dataset('datasetKevin', data=kevin)
    G11 = hdf.create_group('dataset')
    G11.create_dataset('testing', data=X_test[:])
    G11.create_dataset('testingLabels', data=Y_test[:])
    G11.create_dataset('training', data=X_train[:])
    G11.create_dataset('trainingLabels', data=Y_train[:])

#reading from HDF5
with h5py.File('./hdf5_data.h5', 'r') as hdf:
    G1 = hdf.get('/dataset')
    testing = G1.get('testing')
    X_testing = np.array(testing)
    testingLabels = G1.get('testingLabels')
    Y_testing = np.array(testingLabels)
    training = G1.get('training')
    X_training = np.array(training)
    trainingLabels = G1.get('trainingLabels')
    Y_training = np.array(trainingLabels)

#reshape array to be 2 dimensional, no longetr in seperate "chunks"
#convert to dataframe
X_testing = X_testing.reshape(10000,5)
X_testing = pd.DataFrame(X_testing)
#labels corresponding to acceleration data, only 1 column ->convert to 1D
Y_testing = Y_testing.reshape(10000)
X_training = X_training.reshape(89500,5)
X_training = pd.DataFrame(X_training)
Y_training=Y_training.reshape(89500)

#Filtering
X_training.interpolate(method = 'linear', inplace=True)
X_testing.interpolate(method = 'linear', inplace=True)
X_testing.rolling(window=1000).mean()
X_training.rolling(window=200).mean()

#visulaization
#plots the acceleration vs time of the indicated portion of data, along x, y, z and the absolute acceleration too
fig = plt.figure(figsize=(10,10))
fig.suptitle('Acceleration vs. Time Plots')
gs = gridspec.GridSpec(2,2, figure=fig, left =0.1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(X_testing.iloc[0:500,0], X_testing.iloc[0:500,1])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Linear Acceleration x (m/s^2)')
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(X_testing.iloc[0:500, 0], X_testing.iloc[0:500, 2])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Linear Acceleration y (m/s^2)')
ax3 = fig.add_subplot(gs[1,0])
ax3.plot(X_testing.iloc[0:500,0], X_testing.iloc[0:500,3])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Linear Acceleration z (m/s^2)')
ax4 = fig.add_subplot(gs[1,1])
ax4.plot(X_testing.iloc[0:500,0], X_testing.iloc[0:500,4])
ax4.set_xlabel('Time (s)')
ax3.set_ylabel('Absolute acceleration (m/s^2)')
plt.show()

#metadata visualization
zeros = np.count_nonzero(Y_training == 0)
ones = np.count_nonzero(Y_training == 1)
metadata = {'Walking':zeros, 'Jumping':ones}
labels = list(metadata.keys())
values = list(metadata.values())
fig = plt.figure(figsize=(10,10))
plt.bar(labels, values, width=0.5)
plt.xlabel("MetaData")
plt.ylabel("No. of Values")
plt.title("Amount of Zeros vs. Ones")
plt.show()


#features
#varience, zero-crossing-rate, rms, auto correlation coeffs, # of peaks, peak to peak distance,
windowSize =50
testingfeatures = pd.DataFrame(columns = ['mean', 'std', 'max', 'min', 'kurtosis', 'skew', 'median', 'corr', 'sem', 'rank'])
trainingfeatures = pd.DataFrame(columns = ['mean', 'std', 'max', 'min', 'kurtosis', 'skew', 'median', 'corr', 'sem', 'rank'])
x= 0
y=500
for i in range(19):
    for j in range (5):

        testingfeatures['mean'] = X_testing.iloc[x:y, j].rolling(window=windowSize).mean()
        testingfeatures['std'] = X_testing.iloc[x:y, j].rolling(window=windowSize).std()
        testingfeatures['max'] = X_testing.iloc[x:y, j].rolling(window=windowSize).max()
        testingfeatures['min'] = X_testing.iloc[x:y, j].rolling(window=windowSize).min()
        testingfeatures['kurtosis'] = X_testing.iloc[x:y, j].rolling(window=windowSize).kurt()
        testingfeatures['skew'] = X_testing.iloc[x:y, j].rolling(window=windowSize).skew()
        testingfeatures['median'] = X_testing.iloc[x:y, j].rolling(window=windowSize).median()
        testingfeatures['corr'] = X_testing.iloc[x:y, j].rolling(window=windowSize).corr()
        testingfeatures['sem'] = X_testing.iloc[x:y, j].rolling(window=windowSize).sem()
        testingfeatures['rank'] = X_testing.iloc[x:y, j].rolling(window=windowSize).rank()
        x+=500
        y += 500
        testingfeatures = testingfeatures.dropna()

for i in range(178):
    for j in range (5):
        trainingfeatures['mean'] = X_training.iloc[x:y, j].rolling(window=windowSize).mean()
        trainingfeatures['std'] = X_training.iloc[x:y, j].rolling(window=windowSize).std()
        trainingfeatures['max'] = X_training.iloc[x:y, j].rolling(window=windowSize).max()
        trainingfeatures['min'] = X_training.iloc[x:y, j].rolling(window=windowSize).min()
        trainingfeatures['kurtosis'] = X_training.iloc[x:y, j].rolling(window=windowSize).kurt()
        trainingfeatures['skew'] = X_training.iloc[x:y, j].rolling(window=windowSize).skew()
        trainingfeatures['median'] = X_training.iloc[x:y, j].rolling(window=windowSize).median()
        trainingfeatures['corr'] = X_training.iloc[x:y, j].rolling(window=windowSize).corr()
        trainingfeatures['rank'] = X_training.iloc[x:y, j].rolling(window=windowSize).rank()
        x+=500
        y += 500
        trainingfeatures = trainingfeatures.dropna()

#training classifier
#normalization portion
sc = StandardScaler()

X_training = sc.fit_transform(X_training)
X_testing = sc.fit_transform(X_testing)
lgr = LogisticRegression(max_iter=10000)
pca = PCA(n_components=2)

clf = make_pipeline(sc, lgr)
pca_pipe = make_pipeline(sc, pca)
X_train_pca = pca_pipe.fit_transform(X_training)
X_test_pca = pca_pipe.fit_transform(X_testing)

clf.fit(X_train_pca, Y_training)
y_pred_pca = clf.predict(X_test_pca)
y_clf_prob = clf.predict_proba(X_test_pca)

acc = accuracy_score(Y_testing, y_pred_pca)
print('accuracy is: ', acc)

recall = recall_score(Y_testing, y_pred_pca)
print('recall is: ', recall)

#decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method="predict",
    xlabel='X1', ylabel='X2',
    alpha=0.5,
)
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1],c=Y_training)
plt.show()

conf = confusion_matrix(Y_testing, y_pred_pca)
conf_display = ConfusionMatrixDisplay(conf).plot()
plt.show()

# ROC curve plot
fpr, tpr, _ = roc_curve(Y_testing, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# calculating the AUC
auc = roc_auc_score(Y_testing, y_clf_prob[:, 1])
print('the AUC is: ', auc)
