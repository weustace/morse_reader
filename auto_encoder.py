import preprocess

import tensorflow as tf
from tensorflow import keras

import numpy as np
N_FREQ_BINS = 1251
N_TRAIN= 100
N_TEST = 75
BATCH_SIZE = 20



enc=keras.Sequential()
enc.add(keras.layers.Input(shape=(None,1251,1),ragged=True))



enc.add(keras.layers.TimeDistributed(keras.layers.Permute((2,1),input_shape=(1251,1)),input_shape=(None,1251,1))) # works
enc.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1,1251)),input_shape=(None,1,1251)))
enc.add(keras.layers.TimeDistributed(keras.layers.Dense(2,activation='relu',input_shape=(1,1251)),input_shape=(None,1,1251)))

enc.build()
print(enc.summary())




dec = keras.Sequential()
dec.add(keras.layers.Input(shape=(None,1,2),ragged=True))


dec.add(keras.layers.TimeDistributed(keras.layers.Dense(400,activation='relu',input_shape=(1,2)),input_shape=(None,1,2)))
dec.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1,400)),input_shape=(None,1,400)))
dec.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1,1251)),input_shape=(None,1,1251)))
dec.build()
print(dec.summary())
# dec_tdn.add()
# dec_tdn.add(keras.layers.Permute((2,1)))#return to original order
# dec.add(keras.layers.TimeDistributed(dec_tdn))


pipe = keras.Sequential()
pipe.add(keras.layers.Input(shape=(None,1251,1),ragged=True))
pipe.add(enc)
pipe.add(dec)
pipe.build()
print(pipe.summary())
pipe.compile(optimizer='rmsprop',loss = 'mse')


ai,text_list,_ = preprocess.get()
ai = tf.expand_dims(ai,-1) #add a 'features' dimension

ds = tf.data.Dataset.from_tensor_slices((ai,ai)) #we train an autoencoder here (or try to)
ds =ds.shuffle(buffer_size=20)
train_ds = ds.take(N_TRAIN)
test_ds = ds.take(N_TEST)

train_ds = train_ds.batch(BATCH_SIZE)



history = pipe.fit(ai,ai,epochs=2)

pipe.evaluate(test_ds,test_ds)
