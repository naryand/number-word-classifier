#!/usr/bin/env python
# coding: utf-8

# In[53]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Flatten, Reshape
high = 100000000
digits = 12 #len(str(high))
pad = 12

from num2words import num2words
import random

def index_list(pos):
    index_list = [0] * (pos)
    index_list.append(1)
    index_list += [0] * (10-pos-1)
    return index_list

def create_data(low, high, num):
    x_data = []
    y_data = []
    for i in range(num):
        a = random.randrange(low, high)
        b = a
        words = num2words(b)
        c = str(b).zfill(digits)  
        x_data.append(words.replace("-", " ").replace(",", "").replace(" and "," "))
        num_list = []
        for i in range(digits):
            num_list.append(index_list(int(c[i])))
        y_data.append(num_list)
    return x_data, np.array(y_data)

# appends some data to dataset
def append_data(x, y, x_data, y_data):
    x_data.append(x.replace("-", " ").replace(",", "").replace(" and "," "))
    c = str(y).zfill(digits) 
    num_list = [[]]
    for i in range(digits):
        num_list[0].append(index_list(int(c[i])))
    num_list = np.array(num_list)
    y_data = np.concatenate((y_data, num_list), axis=0)
    return x_data, y_data


# In[54]:


x_train, y_train = create_data(0, high, 600000)
x_test, y_test = create_data(0, high, 400000)

# change some "one thousand six hundred" to "sixteen hundred" etc.
for i in range(10):
    for j in range(1000, 1500, 100):
        x_train, y_train = append_data(num2words(j//100) + " hundred", j, x_train, y_train)


# In[55]:


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


# In[56]:


model = tf.keras.models.Sequential()
model.add(Embedding(vocab_size, pad, input_length=pad))
model.add(Flatten())
model.add(Dense(digits*10, activation=tf.nn.softmax))
model.add(Reshape((digits,10)))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=3)


# In[57]:


val_loss, val_acc = model.evaluate(x_test, y_test)
model.save('count2.model')


# In[58]:


new_model = tf.keras.models.load_model('count2.model')


# In[59]:


x = tokenizer.texts_to_sequences(["twelve"
                                 ,"thirteen"
                                 ,"one hundred twenty three"
                                 ,"four hundred seventy two thousand two hundred twenty two"
                                 ,"two hundred thirty seven thousand one hundred forty"
                                 ,"spongebob"
                                 ,"forty two million two hundred thousand one hundred thirteen"])
x = np.array([[0]*(pad-len(i)) + i for i in x])
predictions = new_model.predict(np.array(x))

for prediction in predictions:
    num = ""
    for i in prediction:
        num += str(i.argmax())
    print(int(num))


# In[ ]:





# In[ ]:




