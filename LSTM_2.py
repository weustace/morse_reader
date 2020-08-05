import preprocess
#instead of using RaggedTensors throughout, pad on a per-batch basis to enable the built-in CTC decoder to work properly

import tensorflow as tf
from tensorflow import keras

import compress_pickle

import numpy as np
N_FREQ_BINS = 1251
N_TIMESTEPS = 9930
N_TRAIN= 100
N_TEST = 75
BATCH_SIZE = 5



ai,text_list,_ = preprocess.get()
# ai = tf.expand_dims(ai,-1) #add a 'features' dimension
character_set = None
with open("characters_present.pickled","rb") as f:
    character_set = compress_pickle.load(f)
    
character_set = list(sorted(character_set))

#pad the audio data to constant length
with tf.device('/cpu:0'):
    ai = tf.convert_to_tensor([tf.pad(z,tf.constant([[0,N_TIMESTEPS-z.shape[0]],[0,0]])) for z in ai]) #painfully inefficient but quick enough...

ds = tf.data.Dataset.from_tensor_slices((ai,text_list)) 


with tf.device('/cpu:0'):
    ds =ds.shuffle(buffer_size=175)

    
      

pipe = keras.Sequential()

pipe.add(keras.layers.Input(shape=(N_TIMESTEPS,N_FREQ_BINS),ragged=True))



pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(N_FREQ_BINS,activation='relu',input_shape=(N_FREQ_BINS,)),input_shape=(N_TIMESTEPS,N_FREQ_BINS)))
pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(1,activation='relu',input_shape=(N_FREQ_BINS,)),input_shape=(N_TIMESTEPS,N_FREQ_BINS)))

pipe.add(keras.layers.LSTM(len(character_set)+2))#we hope that it'll work out separation of words and letters too...
pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(50,activation='relu',input_shape=(50,)),input_shape=(N_TIMESTEPS,50)))
pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(50,activation='softmax',input_shape=(50,)),input_shape=(N_TIMESTEPS,50)))
# pipe.add(keras.layers.Lambda(tf.keras.backend.ctc_decode())
pipe.build()
print(pipe.summary())



g_input = next(iter(test_ds))[0] #a single example
g_output = enc.predict(g_input)
with open("experiment_results.pickled",'wb') as f:
    compress_pickle.dump((g_input,g_output),f,compression='gzip')




    
