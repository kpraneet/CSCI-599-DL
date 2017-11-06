from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPool2D


def main():
    train_data = None
    train_label = None
    eval_data = None
    eval_label = None
    test_data = None
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
