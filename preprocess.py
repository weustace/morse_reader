import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm
import os
import scipy.signal
import simpleaudio as sa
import pickle

class AudioProcess():
    audio_list = [] #Python list of np.array for later conversion to tf.RaggedTensor
    text_list = []  #python list of str for later conversion to tf.RaggedTensor

    FIXED_SAMP_RATE = 44100

    def stack_ragged(self,tensors):#from https://stackoverflow.com/questions/57346556/creating-a-ragged-tensor-from-a-list-of-tensors
        values = tf.concat(tensors, axis=0)
        lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
        return tf.RaggedTensor.from_row_lengths(values, lens)


    def load_saved(self,save_path="./saved_file.pickled"):
        with open(self.save_file,'rb') as f:
            (ai,ti) = pickle.load(f)
        return (ai,ti)
    
    def import_and_save(self,search_directory="audio_files",save_path="./saved_file.pickled"):
        for entry in tqdm(os.scandir(search_directory)):
            if entry.is_file() and entry.name.split(".")[1] in ["mp3"]: #we assume that for each audio file there will be a corresponding text file
                y = AudioSegment.from_file(entry.path,frame_rate=44100)
                y = y.set_frame_rate(44100)
                y = np.frombuffer(y.raw_data,dtype=np.int16)
                f,t,y = scipy.signal.stft(y,fs=44100,nperseg=2500)
                y = np.transpose(np.abs(y))
                
                y = tf.constant(y)
                
                audio_list.append(y)
                with open(entry.path.split(".")[0]+".txt",errors='ignore') as f:#there are some non-unicode control characters in the ARRL texts. 
                    text_list.append(f.read().replace("\n"," "))



        #Stack the newly acquired data into TF tensors
        audio_input = self.stack_ragged(audio_list) #ragged tensor needed here
        text_input = tf.constant(text_list)

        with open(save_path,'wb') as f:
            pickle.dump((audio_input,text_input),f)

        return (audio_input,text_input)
