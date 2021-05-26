import keras
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[1]) 
plt.show()
print(y_train[1])

#Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


#Normalize the pixel values from a scale out of 255 to a scale out of 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(y_train[0])

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train[0])

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                              input_shape=input_shape))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

hist=model.fit(x_train, y_train,
          batch_size=128,
          epochs=30,
          validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test))

train_acc=hist.history['accuracy']
test_acc=hist.history['val_accuracy']
train_error=hist.history['loss']
test_error=hist.history['val_loss']
xc=range(30)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_error)
plt.plot(xc,test_error)
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.title('train_loss vs test_loss')
plt.legend(['train','test'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,test_acc)
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.title('Train_acc vs Test_acc')
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

model.summary()
#test=99.31
#train=99.53
