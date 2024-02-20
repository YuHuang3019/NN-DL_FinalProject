#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from collections import Counter
import os
import re
from collections import defaultdict
from tqdm import tqdm

def append_start_stop(sequence):
    append_begin= '<SOL>'
    append_end='<EOL>'
    line=[]
    seq = [s.lower() for s in sequence[1:]]
    line.append(append_begin)
    for word in seq:
        line.append(word)
    line.append(append_end)
    return line   

#Reference :https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase


def flatten_list(listOf_listOf_captions1):
    fl=[]
    flattened_list1 = []
    for x in listOf_listOf_captions1:
        #print(listOf_listOf_captions)
        for y in x:
            for z in y:
                fl.append(z)
    for i in fl:
        flattened_list1.append(i)
    flattened_list2 = set(flattened_list1)
    flattened_list = sorted(flattened_list2)
    return flattened_list


def generate_captions_list(file_captions, train_img_names):
    count=1
    captions=[]
    caption_list =[]
    #rm_ch=[ '!', '"', '#', '&', "'", "'n'", '(', ')', ',', '-','.']
    rm_ch=[ '!', '"', '#', '&', "'", '(', ')', ',', '-','.']
    count = 1
    for line in file_captions : 
        caption=[]
        append_tags=[]
        img_id, text = line.split()[0], line.split()[1:]
        if img_id.split('#')[0] in train_img_names :
            cap_num= int(img_id.split('#')[1])
            for word in text :
                if word not in rm_ch:
                    word = decontracted(word)
                    caption.append(word)
            captions.append(append_start_stop(caption))
            #print(captions)
            if cap_num == 4:
                caption_list.append(captions) 
                captions = []
    return caption_list


def generate_captions_dict(file_captions, train_img_names):
    count=0
    captions_dict=defaultdict(list)
    captions=[]
    #rm_ch=[ '!', '"', '#', '&', "'", "'n'", '(', ')', ',', '-','.']
    rm_ch=[ '!', '"', '#', '&', "'", '(', ')', ',', '-','.']
    for line in file_captions : 
        caption=[]
        append_tags=[]
        img_id, text = line.split()[0], line.split()[1:]
        img_name = img_id.split('#')[0]
        if img_name in train_img_names :
            cap_num= int(img_id.split('#')[1])   
            for word in text :
                if word not in rm_ch:
                    word = decontracted(word)
                    caption.append(word)
            captions.append(append_start_stop(caption))
            if cap_num == 4:
                captions_dict[img_name].append(captions) 
                #print(captions_dict[img_name])
                captions = []
    return captions_dict
        

def tokenize(flattened_list):
    map_num2token = {num:token for num,token in enumerate(flattened_list)}
    map_token2num = {token:num for num,token in enumerate(flattened_list)}
    return map_num2token,map_token2num

#Max length for the generator

def max_len(train_img_names,captions_dict):
    lengths = []
    for image_id in train_img_names :
        for captions in captions_dict[image_id]:
            for caption in captions:
                lengths.append(len(caption))

    return max(lengths)
    
   
