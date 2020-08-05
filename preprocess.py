import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm
import os
import scipy.signal
# import simpleaudio as sa
import compress_pickle
compression_format='gzip'

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
        (ai,ti,mp3_paths) = compress_pickle.load(f,compression=compression_format)
    return (ai,ti,mp3_paths)

def import_and_save(search_directory="audio_files",save_path="./saved_file.pickled",*,audio_list = None, text_list = None, file_list=None):
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0],True)
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
            print(entry.path)
            y = AudioSegment.from_file(entry.path,frame_rate=FIXED_SAMP_RATE)
            y = y.set_frame_rate(FIXED_SAMP_RATE)
            y = np.frombuffer(y.raw_data,dtype=np.int16)
            
            y = y.astype(np.float32)
            # f,t,y = scipy.signal.stft(y,fs=44100,nperseg=2500)
            # with tf.device('/cpu:0'):
            y += tf.random.normal(y.shape,mean=0,stddev=6000)
            y = tf.signal.stft(y,int(0.05*FIXED_SAMP_RATE),int(0.05*FIXED_SAMP_RATE),pad_end=True,fft_length=int(0.05*FIXED_SAMP_RATE))
            # y = np.transpose(np.abs(y))
            with tf.device('/cpu:0'): #Seems to be the only way to force TF to store y in RAM from hereon in
                #(the underlying numpy array is not modified again after this; because they are immutable, it is rewritten during this 
                #operation, so this represents an opportunity to copy it into RAM. Because tf.constant then captures a pointer to y, 
                #if we don't move it into RAM now then each tensor in audio_list lives in VRAM and we run out of VRAM rather quickly!)
                y = tf.abs(y)
            
            y = tf.constant(y)            
            audio_list.append(y)
            with open(entry.path.split(".")[0]+".txt",errors='ignore') as f:#there are some non-unicode control characters in the ARRL texts--0x1a = 'substitute' 
                text_list.append(f.read().replace("\n"," ").replace("\x1a",""))
            file_list.append(entry.path)


    #Stack the newly acquired data into TF tensors
    with tf.device('/cpu:0'):#we don't have enough GPU memory to handle the whole dataset at once.
        audio_input = stack_ragged(audio_list) #ragged tensor needed here
        text_input = tf.constant(text_list)

    with open(save_path,'wb') as f:
        compress_pickle.dump((audio_input,text_input,file_list),f,compression=compression_format,protocol=4)

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

    
    