import tensorflow as tf
from tensorflow.keras import models,layers,regularizers
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
x_train=x_train[...,None]
x_test=x_test[...,None]
y_train,y_test=tf.keras.utils.to_categorical(y_train,10),tf.keras.utils.to_categorical(y_test,10)

def create_model(regularizer=None,dropout_rate=None):
    model=models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,(3,3),activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu',kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate) if dropout_rate else layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

configurations=[
    ("Base Model",None,None),
    ("Model with L1 Regularization",regularizers.l1(1e-4),None),
    ("Model with L2 Regularization",regularizers.l2(1e-4),None),
    ("Model with droput",None,0.5)
]

for name,regularizer,dropout_rate in configurations:
    print(f"\n Training {name}...")
    model=create_model(regularizer,dropout_rate)
    model.fit(x_train,y_train,epochs=5,batch_size=32,verbose=2)
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
    print(f"Test accuracy:{test_acc:.4f}")