import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x,y=make_moons(n_samples=500)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
activations=['sigmoid','tanh','relu']
histories={}

for act in activations:
    model=Sequential([
        Dense(16,activation=act,input_dim=2),
        Dense(8,activation=act),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    histories[act]=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,verbose=0).history

for act in activations:
    plt.plot(histories[act]['val_accuracy'],label=act)
plt.title("Validation accuracy")
plt.legend()
plt.show()