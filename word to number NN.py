#!/usr/bin/env python
# coding: utf-8

# In[287]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Flatten, Reshape
high = 1000000
digits = len(str(high))
pad = 10

from num2words import num2words
import random

def index_list(pos):
    index_list = [0] * (pos)
    index_list.append(1)
    index_list += [0] * (10-pos-1)
    return index_list

def create_data(high, num):
    x_data = []
    y_data = []
    for i in range(num):
        a = random.randrange(0, high)
        b = a
        c = str(b).zfill(digits)
        x_data.append(num2words(b).replace("-", " ").replace(",", "").replace(" and "," "))
        num_list = []
        for i in range(digits):
            num_list.append(index_list(int(c[i])))
        y_data.append(num_list)
    return x_data, np.array(y_data)


# In[288]:


x_train, y_train = create_data(high, 60000)
x_test, y_test = create_data(high, 40000)


# In[289]:


num_words = 0
for i in x_train:
    num_words += len(i.split(" "))
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = np.array([[0]*(pad-len(i)) + i for i in x_train])

x_test = tokenizer.texts_to_sequences(x_test)
x_test = np.array([[0]*(pad-len(i)) + i for i in x_test])

print(tokenizer.sequences_to_texts(x_train)[342])
print(x_train[342])
print(y_train[342])

vocab_size = len(tokenizer.word_index) + 1


# In[290]:


model = tf.keras.models.Sequential()
model.add(Embedding(vocab_size, pad, input_length=pad))
model.add(Flatten())
model.add(Dense(digits*10, activation=tf.nn.softmax))
model.add(Reshape((digits,10)))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=35)


# In[291]:


val_loss, val_acc = model.evaluate(x_test, y_test)


# In[292]:


model.save('count2.model')
new_model = tf.keras.models.load_model('count2.model')


# In[299]:


x = tokenizer.texts_to_sequences(["twelve"
                                 ,"thirteen"
                                 ,"one hundred twenty three"
                                 ,"four hundred seventy two thousand two hundred twenty two"
                                 ,"two hundred thirty seven thousand one hundred forty"])
x = np.array([[0]*(pad-len(i)) + i for i in x])
predictions = new_model.predict(np.array(x))

for prediction in predictions:
    num = ""
    for i in prediction:
        num += str(i.argmax())
    print(int(num))


# In[ ]:




