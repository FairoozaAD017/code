import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train.reshape(-1,784)/255.0,x_test.reshape(-1,784)/255.0

model=Sequential([
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

histories=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=32)

plt.plot(histories.history['accuracy'],label='Training accuracy')
plt.plot(histories.history['val_accuracy'],label="Validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"accuracy :{test_acc*100:.2f}%")