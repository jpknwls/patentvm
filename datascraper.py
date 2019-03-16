#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
autoencoder.py: AutoEncoder Model
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""

import numpy as np
import pandas as pd
import zipfile
import re
import random
import urllib as urllib2
from bs4 import BeautifulSoup
import wget, os
import time
import h5py


links = np.load('data.npy')


def get_links():
        """ Get all the links to zips
        """
    url = 'https://www.google.com/googlebooks/uspto-patents-grants-text.html'
    resp = urllib2.urlopen(url)
    soup = BeautifulSoup(resp.read())
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links[900:]
zips = get_links()


def extract_zip(z, zip_path, txt_path):
        """ extract a zip file to a certain path

        @param z (list): list of zip names
        @param zip_path (string): path to zip files
        @param txt_path (string): path to save file
        """
    zip_dir = zip_path + z[-10:]
    
    try:
        filename = wget.download(z, out = zip_dir)
    except:
        return None
    
    try:
        zip_ref = zipfile.ZipFile(zip_dir, 'r')
    except:
        return None
    
    zip_ref.extractall(txt_path)
    zip_ref.close() 
    return zip_dir

def get_txt_content(filepath, approved):
    """ returns raw text and relevant metadata

        @param filepath (string): path to txt file
        @param approved (list): list of approved string for beginning of lines
        """
    def approve_line(line, approved):
        if len(line) > 2 and not line[2].isupper(): return True
        for a in approved:
            if a in line:
                return True
        return False

    contents = []
    for line in open(filepath, 'rb'):
        if approve_line(str(line), approved): 
            contents.append(str(line.strip()))
    return contents


def preprocess_doc(docs):
    """ preprocesses the txt in each document

    @param docs (list): list of lists of strings (words in documents)
    """
    new_docs = []
    toktok = ToktokTokenizer()
    to_remove = [r'FIG. [1-9]', 'PAL',  'PNO', 'b\'', '\'']
    
    stopword_list = nltk.corpus.stopwords.words('english')
    
    stopword_list.append('said')
    stopword_list.append('b')
    stopword_list.remove('no')
    stopword_list.remove('not')
    
    for doc in docs:
        content = doc['content']
        for x in to_remove: content = re.sub(x, '', content)
        for x in to_remove: doc['id'] = re.sub(x, '', doc['id'])
        for x in to_remove: doc['title'] = re.sub(x, '', doc['title'])
        # tokenize
        tokens = re.split('\s+', content)
        words = [w for w in tokens if w.isalpha()]
        filtered_tokens = [str.lower(token) for token in words if token not in stopword_list]
        #append to new_docs
        doc['content'] = filtered_tokens
        new_docs.append(doc)
    return new_docs


def get_patents_from_txt(contents, ids): 
    """ Return relevant sections of patents from bodies of txt

    @param contents (string): list of strings
    @param ids (set): set of ids we want to keep 

    returns: list of documents
    """
    def replace_text(text, to_remove):
        for x in to_remove: text = re.sub(x, '', text)
        return str(text).strip()
        
    patents = [] 
    patent = dict()
    patent['citations'] = []
    curr_patent_txt = []
    for content in contents:
        
        if 'WKU' in content:
            if patent:
                patent['content'] = ' '.join(curr_patent_txt)
                #if 'id' in patent: print(patent['id'], len(patent['id']))
                if 'id' in patent and patent['id'] in ids: patents.append(patent)
                patent = dict()
                patent['citations'] = []
                curr_patent_txt = []
                #remove last number
            content = content.replace('WKU  ', '')
            content = [c for c in content.strip() if c.isdigit() or c.isupper()]
            if len(content) > 7: content = content[:8]
            if content[1] == '0': content.pop(1)
            if content[0] == '0': content.pop(0)
            #print(''.join(content))
            patent['id'] = ''.join(content)
            #patent['id'] = content.replace('WKU  ', '')
            continue
        if 'TTL' in content:
            patent['title'] = content.replace('TTL  ', '')
            
            continue
        if 'PNO' in content:
            #strip white space, strip letters, cut to max 7 (take off end if over 7)
            content = [c for c in content.strip() if c.isdigit()] 
            if not content: continue
            if content[0] == 0: content = content[1:]
            if len(content) > 7: content = content[:8]
            patent['citations'].append(''.join(content))
        else:
            curr_patent_txt.append(content)
    return patents



def get_relevant_docs(contents, min_size):
    """ Return docs with enough text for use

    @param contents (string): list of strings
    @param min_size (int): min number of words in file

    returns: list of documents
    """
    return [c for c in contents if len(c["content"]) > min_size]


ids = set()
for key, values in links[()].items():
    ids.add(str(key))
    for v in values: ids.add(str(v))


zip_path = 'patent_zips/'
txt_path = 'patent_txt/'
txt_cutoff = 55
approved = ['TTL', 'PAL', 'FIG', 'PNO', 'WKU']
completed_patents = []
min_size = 0
num = 0

start = time.time()
for z in zips:
    #get zip file from url
    zip_dir = extract_zip(z, zip_path, txt_path)
    if not zip_dir: continue
    os.remove(zip_dir)
    
    try:
        contents = get_txt_content(txt_path+os.listdir(txt_path)[0], approved)
    except:
        os.remove(txt_path+os.listdir(txt_path)[0])
        continue
    
    patents = get_patents_from_txt(contents, ids)
    _patents = get_relevant_docs(patents, min_size)
    processed_patents = preprocess_doc(_patents)
    
    os.remove(txt_path+os.listdir(txt_path)[0])
    
    np.save('../data/patent_fuller' + str(num),processed_patents)
    completed_patents.extend(processed_patents)
    num += 1
    
end = time.time()
print(end - start)

# collect small files

agg = []
for num in range(162):
    
    filename = '../data/patent_fuller' + str(num) + '.npy'
    temp = np.load(filename)
    agg.extend(temp)


np.save('../data/patent_large', agg[:20000])



