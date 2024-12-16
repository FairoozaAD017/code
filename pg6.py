import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Bidirectional

max_features=10000
max_len=200

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)

model=Sequential([
    Embedding(input_dim=max_features,output_dim=64,input_length=max_len),
    Bidirectional(LSTM(32)),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=64,validation_split=0.2)

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test accuracy:{test_acc:.2f}")

example_review="The movie was absolutely amazing,I loved it!"
word_index=imdb.get_word_index()
encoded_review=[word_index.get(word,2) + 3 for word in example_review.lower().split()]
padded_review=pad_sequences([encoded_review],maxlen=max_len)
prediction=model.predict(padded_review)[0][0]
print(f"{'Positive' if prediction > 0.5 else 'Negative'} sentiment with confidence {prediction:.2f}")




