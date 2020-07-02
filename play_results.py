import compress_pickle
import pydub 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pydub.playback import play

with open("experiment_results.pickled",'rb') as f:
    (ip,op) = compress_pickle.load(f,compression='gzip')


ax,(p1,p2) = plt.subplots(1,2,sharey=True,sharex=True)
p1.set_title("Input audio")
p2.set_title("Generated audio")
p1.set_xlabel("Frequency (a.u.)")
p2.set_xlabel("Frequency (a.u.)")
p1.set_ylabel("Time (a.u.)")
p1.set_xlim(40,50)
p2.set_xlim(40,50)
p1.set_ylim(5000,5050)
p2.set_ylim(5000,5050)
p1.imshow(ip[0])
p2.imshow(op[0])
plt.show()


