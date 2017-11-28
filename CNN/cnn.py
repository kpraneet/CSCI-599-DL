import cv2
import os.path
import numpy as np
from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalMaxPool2D


def main():
    path = "/Users/praneet/Documents/USC - MS/Fall 2017/CS 599 - DL/CSCI-599-DL/CNN/IRMAS-training_spectrograms"
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')
    eval_data = np.load('eval_data.npy')
    eval_label = np.load('eval_label.npy')
    test_data = np.load('test_data.npy')
    test_label = np.load('test_label.npy')
    print('Train data: ', len(train_data))
    print('Eval data: ', len(eval_data))
    print('Test data: ', len(test_data))
    epochs = 1000
    learn_rate = 0.001
    filecount = 0
    data_name = []
    data_list = []
    if os.path.isfile('cnn_model.h5'):
        model = load_model('cnn_model.h5')
    else:
        model = Sequential()
        model.add(BatchNormalization(input_shape=(43, 128, 3)))
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
        model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.50))
        model.add(Dense(11, activation='sigmoid'))
        print(model.summary())
    opt = optimizers.Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2),
                 ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=2),
                 TensorBoard(log_dir='./tensorboard', histogram_freq=3, write_graph=True, write_images=True,
                             write_grads=True)]
    model.fit(x=train_data, y=train_label, validation_data=(eval_data, eval_label), batch_size=128, verbose=2,
              epochs=epochs, callbacks=callbacks, shuffle=True)
    model.save('cnn_model.h5')
    saved_model = load_model('checkpoint.h5')
    score = saved_model.evaluate(x=eval_data, y=eval_label, batch_size=128)
    print(score)
    predictions = saved_model.predict(test_data, batch_size=128)
    print(predictions)
    print(test_label)
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".png"):
                filecount += 1
                file_path = os.path.abspath(os.path.join(root, file_name))
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fixed_size = (128, 43)
                img = cv2.resize(img, dsize=fixed_size)
                data_list.append(img)
                data_name.append(file_name)
    data_list = np.array(data_list, np.float32) / 255.
    for x in range(len(predictions)):
        if np.where(predictions[x] == predictions[x].max()) == np.where(test_label[x] == test_label[x].max()):
            for y in range(len(data_list)):
                if np.array_equal(test_data[x], data_list[y]):
                    print(data_name[y])
    pred_score = saved_model.evaluate(x=test_data, y=test_label, batch_size=128)
    print(pred_score)


if __name__ == '__main__':
    main()
