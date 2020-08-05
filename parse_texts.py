import preprocess
import os
import compress_pickle

character_set = set()
ai,text_list,_ = preprocess.get()
for text in text_list:
    character_set.update(set(str(text.numpy())))


print(len(character_set))

with open("characters_present.pickled","wb") as f:
    compress_pickle.dump(character_set,f,compression='pickle')