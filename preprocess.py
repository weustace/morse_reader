import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm
import os
import scipy.signal
# import simpleaudio as sa
import pickle

audio_list = [] #Python list of np.array for later conversion to tf.RaggedTensor
text_list = []  #python list of str for later conversion to tf.RaggedTensor

accepted_audio_file_type_list = ["mp3"]

FIXED_SAMP_RATE = 44100

def stack_ragged(tensors):#from https://stackoverflow.com/questions/57346556/creating-a-ragged-tensor-from-a-list-of-tensors
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

def load_saved(save_path="./saved_file.pickled"):
    with open(save_path,'rb') as f:
        (ai,ti,mp3_paths) = pickle.load(f)
    return (ai,ti,mp3_paths)

def import_and_save(search_directory="audio_files",save_path="./saved_file.pickled",*,audio_list = None, text_list = None, file_list=None):
    if audio_list is None:
        audio_list = 0
        audio_list = []
    if text_list is None:
        text_list = 0
        text_list = []
    if file_list is None:
        file_list = 0
        file_list = []
    
    for entry in tqdm(os.scandir(search_directory)):
        if entry.is_file() and entry.name.split(".")[1] in accepted_audio_file_type_list and entry.path not in file_list: #we assume that for each audio file there will be a corresponding text file
            y = AudioSegment.from_file(entry.path,frame_rate=44100)
            y = y.set_frame_rate(44100)
            y = np.frombuffer(y.raw_data,dtype=np.int16)
            f,t,y = scipy.signal.stft(y,fs=44100,nperseg=2500)
            y = np.transpose(np.abs(y))
            
            y = tf.constant(y)
            
            audio_list.append(y)
            with open(entry.path.split(".")[0]+".txt",errors='ignore') as f:#there are some non-unicode control characters in the ARRL texts. 
                text_list.append(f.read().replace("\n"," "))
            file_list.append(entry.path)


    #Stack the newly acquired data into TF tensors
    audio_input = stack_ragged(audio_list) #ragged tensor needed here
    text_input = tf.constant(text_list)

    with open(save_path,'wb') as f:
        pickle.dump((audio_input,text_input,file_list),f)

    return (audio_input,text_input,file_list)


def get(save_path="./saved_file.pickled",search_directory="audio_files"): 
    try:
        (audio,text,file_list) =  load_saved(save_path)
    except FileNotFoundError:
        print("Saved file not found; regenerating")
        return import_and_save(search_directory,save_path)
    
    for entry in os.scandir(search_directory):
        if entry.is_file() and entry.name.split(".")[1] in accepted_audio_file_type_list and entry.path not in file_list:
            print("Loaded file ({0}) not in saved dataset. Recalculate (y/n) or continue?".format(entry.path))
            if input().lower()=='y':
                return import_and_save(search_directory,save_path) 
            
    return (audio,text,file_list)

    
    