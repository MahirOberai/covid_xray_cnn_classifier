import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report
from keras.utils import to_categorical
from tqdm import tqdm
import math
import comparison
import preprocess_images as pro

mode = 1 #0 for VGG16 cnn, 1 for other classifers
submit = 0 #0 for no submit, 1 for creating submission csv

classifiers = [DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), svm.SVC()]

train = pd.read_csv('train.csv')
train_image_path = "train"
labels = ['normal','bacterial','viral','covid']
class_weight = {0: 1.,
                1: 1.,
                2: 1.,
                3: 5.}
input_array = np.genfromtxt('train.csv', delimiter=',',skip_header=1,usecols=[1,2],dtype='U')


train_image = []
y = []


for i in tqdm(range(train.shape[0])):
    
    
    img = image.load_img('train/'+train['filename'][i], target_size=(128,128,1), color_mode='grayscale', interpolation='nearest')
    #img = no_noise_image[i]
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
    
    
    for idx in range(0, len(labels)):
        if input_array[i][1] == labels[idx]:
            y.append(idx)

X = np.array(train_image)

print(X.shape)

n_samples = X.shape[0]
X_reshaped = X.reshape(n_samples, -1)

if mode == 0:

    #y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))

    INIT_LR = 1e-2
    BATCH_SIZE = 32
    EPOCHS = 40
    TRAINING_SIZE = len(X_train)
    VALIDATION_SIZE = len(X_test)

    opt = 'Adam'

    model.summary()

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))

    steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)
    val_steps = compute_steps_per_epoch(VALIDATION_SIZE)

    checkpoint = ModelCheckpoint("covidcnn_5.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    
    saved_model = load_model("covidcnn_3.h5")

    
    hist = saved_model.fit(X_train, y_train, 
                        epochs=100, 
                        #steps_per_epoch=5,
                        #validation_steps = 1, 
                        validation_data=(X_test, y_test),
                        class_weight=class_weight,
                        callbacks=[checkpoint,early])
                        #shuffle = True)
    
    

    """
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    #plt.plot(hist.history['loss'])
    #plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy"])#,"loss","Validation Loss"])
    plt.show()
    """

    #y_test_rounded = np.argmax(y_test, axis=1)

    y_pred = saved_model.predict(X_test)
    y_pred_rounded = np.argmax(y_pred, axis=1)

    classification_report = classification_report(y_test, y_pred_rounded,target_names=labels)

    print(classification_report)
   
    plt.title("cnn classification report")
    comparison.plot_classification_report(classification_report)
    plt.show()
    
   

else:
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, random_state=42, test_size=0.2)
    for clf in classifiers:
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        accuracy = accuracy_score(y_test, y_pred)
        #precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
        print("classifer: ", clf)
        print("accuracy: ", accuracy)
        classification_report = classification_report(y_test, y_pred,target_names=labels)
        print(classification_report)
        comparison.plot_classification_report(classification_report)
        plt.show()
        

if submit == 1:

    saved_model = load_model("covidcnn.h5")

    test = pd.read_csv('test.csv')

    test_image = []
    for i in tqdm(range(test.shape[0])):
        img = image.load_img('test/'+test['filename'][i], target_size=(128,128,1), grayscale=False, interpolation='nearest')
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    test = np.array(test_image)
    test_reshaped = test.reshape(test.shape[0], -1)

    print("test shape: ",test.shape)

    if mode == 0:
        y_pred = saved_model.predict_classes(test)

    predicted_labels = []
    for i in range(0, len(y_pred)):
        predicted_labels.append(labels[y_pred[i]])

    sample = pd.read_csv('sample_submission.csv')
    sample['label'] = predicted_labels
    sample.to_csv('covid_cnn_submission5.csv', header=True, index=False)

    