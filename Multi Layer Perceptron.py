import datetime
start = datetime.datetime.now()

# Baseline MLP for MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[1])
plt.show()
print(y_train[1])


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
print("Number of :  ", num_pixels)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print("nuber _ classes : ", num_classes)

# define baseline model
def baseline_model():
  model = Sequential()
  model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
  model.add(Dense(500, kernel_initializer='normal', activation='relu'))
  model.add(Dense(num_classes, kernel_initializer='normal', activation='tanh'))
  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
  
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# build the model
model = baseline_model()

# Fit the model
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy rate: " + str(scores[1]*100))

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

end = datetime.datetime.now() - start
print("Total time required : ", end)


# Test Accuracy rate    : 98.85000085830688
# Train Accuracy rate   : 99.92
# Time Taken to execute : 02:32.013941 minutes
# Epochs                : 30

