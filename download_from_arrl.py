import os
from urllib import request
import re

import time


LOG_FILE = "audio_files/ARRL_DOWNLOADED.csv"
domains = ["http://www.arrl.org/25-wpm-code-archive"]
already_read = set()

with open(LOG_FILE) as f:
    for line in f.read().split("\n")[:-1]:
        already_read.add(line.split(",")[1])
for domain in domains:
    page = request.urlopen(domain)
    str_page = str(page.read())
    for link_id in re.findall("http://www.arrl.org/files/file/Morse/Archive/.{0,40}\.mp3",str_page):
        if link_id not in already_read:
            file_name = link_id.split("/")[-1].replace("WPM","")
            os.system("wget -O audio_files/{0} {1}".format(file_name,link_id))
            text_link_id = link_id[:-len(file_name)-3] + file_name.split(".")[0] + ".txt"
            os.system("wget -O audio_files/{0} {1}".format(text_link_id.split("/")[-1],text_link_id))
            with open(LOG_FILE,'a') as f:
                f.write("{0},{1}\n".format(time.time(),link_id))
            already_read.add(link_id)
            time.sleep(2) #pause between requests to be nice to the server (wget is blocking)



