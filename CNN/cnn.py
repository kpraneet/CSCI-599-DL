import os
import cv2
import numpy as np
from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPool2D


def pre_process():
    sax = "/Users/praneet/Documents/USC - MS/Fall 2017/CS 599 - DL/CSCI-599-DL/IRMAS-spectrograms/Training/sax"
    vio = "/Users/praneet/Documents/USC - MS/Fall 2017/CS 599 - DL/CSCI-599-DL/IRMAS-spectrograms/Training/vio"
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    test_data = []
    test_label = []
    for root, dirs, files in os.walk(sax):
            for file_name in files:
                if file_name.endswith(".png"):
                    file_path = os.path.abspath(os.path.join(root, file_name))
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    fixed_size = (43, 128)
                    img = cv2.resize(img, dsize=fixed_size)
                    train_data.append(img)
                    train_label.append(0)
    for root, dirs, files in os.walk(vio):
            for file_name in files:
                if file_name.endswith(".png"):
                    file_path = os.path.abspath(os.path.join(root, file_name))
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    fixed_size = (43, 128)
                    img = cv2.resize(img, dsize=fixed_size)
                    train_data.append(img)
                    train_label.append(1)
    eval_data.append(train_data[0])
    eval_label.append(train_label[0])
    test_data.append(train_data[1])
    test_label.append(train_label[1])
    train_data = np.array(train_data, dtype=np.float32) / 255. #, dtype=np.float32)
    train_label = np.array(train_label, dtype=np.int32) #, dtype=np.int32)
    eval_data = np.array(eval_data) #, dtype=np.float32)
    eval_label = np.array(eval_label) #, dtype=np.int32)
    test_data = np.array(test_data) #, dtype=np.float32)
    test_label = np.array(test_label) #, dtype=np.int32)
    print(train_data.shape)
    return train_data, train_label, eval_data, eval_label, test_data, test_label


def main():
    train_data = None
    train_label = None
    eval_data = None
    eval_label = None
    test_data = None
    test_label = None
    train_data, train_label, eval_data, eval_label, test_data, test_label = pre_process()
    epochs = 10
    learn_rate = 0.001
    model = Sequential()
    model.add(BatchNormalization(input_shape=(43, 128, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(GlobalMaxPool2D())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.50))
    model.add(Dense(11, activation='sigmoid'))
    print(model.summary())
    opt = optimizers.Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=2),
                 ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=2)]
    model.fit(train_data, train_label, batch_size=128, verbose=2, epochs=epochs, callbacks=callbacks)
    model.save('cnn_model.h5')
    score = model.evaluate(eval_data, eval_label, batch_size=128)
    print(score)
    saved_model = load_model('cnn_model.h5')
    predictions = saved_model.predict(test_data, batch_size=128)
    print(predictions)


if __name__ == '__main__':
    main()
