import preprocess

import tensorflow as tf
from tensorflow import keras

import compress_pickle

import numpy as np
N_FREQ_BINS = 1251
N_TRAIN= 100
N_TEST = 75
BATCH_SIZE = 5



enc=keras.Sequential()
enc.add(keras.layers.Input(shape=(None,1251),ragged=True))



enc.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1251,)),input_shape=(None,1251)))
enc.add(keras.layers.TimeDistributed(keras.layers.Dense(1,activation='relu',input_shape=(1251,)),input_shape=(None,1251)))

enc.build()
print(enc.summary())




dec = keras.Sequential()

dec.add(keras.layers.Input(shape=(None,1),ragged=True))


dec.add(keras.layers.TimeDistributed(keras.layers.Dense(400,activation='relu',input_shape=(1,)),input_shape=(None,1)))
dec.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(400,)),input_shape=(None,400)))
dec.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1251,)),input_shape=(None,1251)))
dec.build()
print(dec.summary())
# dec_tdn.add()
# dec_tdn.add(keras.layers.Permute((2,1)))#return to original order
# dec.add(keras.layers.TimeDistributed(dec_tdn))


pipe = keras.Sequential()
pipe.add(keras.layers.Input(shape=(None,1251),ragged=True))
pipe.add(enc)
pipe.add(dec)
pipe.build()
print(pipe.summary())

@tf.function
def ragged_supporting_loss(target,predicted):
    diff = tf.math.subtract(target,predicted)
    diff = tf.math.square(diff)
    return tf.reduce_mean(diff)
pipe.compile(optimizer='adam',loss = ragged_supporting_loss)


ai,text_list,_ = preprocess.get()
# ai = tf.expand_dims(ai,-1) #add a 'features' dimension

ds = tf.data.Dataset.from_tensor_slices((ai,ai)) #we train an autoencoder here (or try to)
t = iter(ds)
for j in t:
    print(np.max(j[0]))
with tf.device('/cpu:0'):
    ds =ds.shuffle(buffer_size=175)
train_ds = ds.take(N_TRAIN)
test_ds = ds.take(N_TEST)

train_ds = train_ds.batch(2)
test_ds = test_ds.batch(5)



history = pipe.fit(train_ds,epochs=10)

pipe.evaluate(test_ds)

g_input = next(iter(test_ds))[0]
g_output = pipe.predict(g_input)

with open("experiment_results.pickled",'wb') as f:
    compress_pickle.dump((g_input,g_output),f,compression='gzip')

enc_layer = pipe.layers[0]
enc_layer.save("./encoder.saved_model")