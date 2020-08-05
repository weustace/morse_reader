import preprocess

import tensorflow as tf
from tensorflow import keras

import compress_pickle

import numpy as np
N_FREQ_BINS = 1251
N_TRAIN= 100
N_TEST = 75
BATCH_SIZE = 5

@tf.function
def ragged_supporting_loss(target,predicted):
    diff = tf.math.subtract(target,predicted)
    diff = tf.math.square(diff)
    return tf.reduce_mean(diff)


enc=keras.Sequential()
enc.add(keras.layers.Input(shape=(None,1251),ragged=True))



enc.add(keras.layers.TimeDistributed(keras.layers.Dense(1251,activation='relu',input_shape=(1251,)),input_shape=(None,1251)))
enc.add(keras.layers.TimeDistributed(keras.layers.Dense(1,activation='relu',input_shape=(1251,)),input_shape=(None,1251)))

enc.build()
trained_enc = keras.models.load_model("encoder.saved_model")
for layer in enc.layers:
    layer.trainable=False #freeze the model
    print(enc.summary())

for i in range(len(trained_enc.layers)):
    enc.layers[i].set_weights(trained_enc.layers[i].get_weights()) #this is a bit ugly but loading models seems to break data formats somehow. To fix...


ai,text_list,_ = preprocess.get()
# ai = tf.expand_dims(ai,-1) #add a 'features' dimension
character_set = None
with open("characters_present.pickled","rb") as f:
    character_set = compress_pickle.load(f)
    
character_set = list(sorted(character_set))

ds = tf.data.Dataset.from_tensor_slices((ai,text_list)) 


with tf.device('/cpu:0'):
    ds =ds.shuffle(buffer_size=175)

def ctc_decode_wrapper(x):
    input_lengths = [x[i].shape[0]
    
        

pipe = keras.Sequential()
pipe.add(enc)
pipe.add(keras.layers.LSTM(len(character_set)+2))#we hope that it'll work out separation of words and letters too...
pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(50,activation='relu',input_shape=(50,)),input_shape=(None,50)))
pipe.add(keras.layers.TimeDistributed(keras.layers.Dense(50,activation='softmax',input_shape=(50,)),input_shape=(None,50)))
pipe.add(keras.layers.Lambda(tf.keras.backend.ctc_decode())
pipe.build()
print(pipe.summary())



g_input = next(iter(test_ds))[0] #a single example
g_output = enc.predict(g_input)
with open("experiment_results.pickled",'wb') as f:
    compress_pickle.dump((g_input,g_output),f,compression='gzip')




    
