import io
import os
import sys
from urllib.parse import urlparse

import nltk
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import re
from nltk.stem import PorterStemmer
import json
import time
from sklearn.feature_extraction.text import HashingVectorizer

from pyexpat import features
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

indexDocuments = 0
unique = set()

index = defaultdict(list)    # key = token, value = postinglist(doc_id, #term freq)
corpus = {}  # dict of docid: document.text() in the collection
corpusurls = {} # key = docid, value = url
doc_lengths = {} # length of all documents

def get_term_frequency(posting_list, doc_id): # returns frequencies of terms within document when getting length of document
    for doc, freq in posting_list:
        if doc == doc_id:
            return freq
    return 0

def writePartialToDisk(fileNum): # writes partial indexes that are created to disk
    partialIndex = 'partial_index' + str(fileNum) + '.json'
    with open(partialIndex, 'w') as f:
        json.dump(index, f)

def combinePartials(): # merges all partial indexes into one
    combined_index = defaultdict(list)
    for file in os.listdir():
        if file.startswith('partial_index'):
            with open(file, 'r') as f:
                partial_index = json.load(f)
                for term, postings in partial_index.items():
                    combined_index[term].extend(postings)
    with open('inverted_index.json', 'w') as f: # creates json file of merged index
        json.dump(combined_index, f)
    with open("inverted_index.json", "r") as file: # reopens merged index file to sort
        data = json.load(file)
    sorted_data = {key: data[key] for key in sorted(data)} # sorts all data alphabetically
    with open("inverted_index.json", "w") as file: # rewrites index with sorted index
        json.dump(sorted_data, file)

    for doc_id in corpus: # gets document length and puts into dictionary for future use
        length = 0
        for term in corpus[doc_id]:
            w = calculate_weight(term)
            posting_list = get_posting_list(term)
            tf = get_term_frequency(posting_list, doc_id)
            length += w * w * tf
        doc_lengths[doc_id] = math.sqrt(length)

def invertedIndexer(dir):
    global index
    global indexDocuments
    global corpus
    global corpusurls
    global docs
    indexNum = 1
    docID = 0


    for root, dirs, files in os.walk(dir):
        for file in files:
            with io.open(os.path.join(root, file), 'r', encoding='UTF-8', errors='ignore') as f:
                t = f.read()
                soup = BeautifulSoup(t, 'html.parser')
                text = soup.get_text()
                tokens = tokenizer(text)
                term_freqs = defaultdict(int)

                try:
                    r = json.loads(t)
                    parsed = urlparse(r['url'])
                    defragLink = parsed._replace(fragment='').geturl()
                    for token in tokens:
                        term_freqs[token] += 1
                    for term, freq in term_freqs.items():
                        posting_list = index[term]
                        posting_list.append((str(docID), freq))
                        index[term] = posting_list
                    corpus[str(docID)] = text
                    indexDocuments += 1
                    corpusurls[str(docID)] = defragLink

                except:
                    continue
                docID += 1

        index_size_bytes = sys.getsizeof(index)
        index_size_mb = index_size_bytes / (1024 * 1024)
        if index_size_mb >= 10:
            writePartialToDisk(indexNum)
            indexNum += 1
            index = defaultdict(list)
    writePartialToDisk(indexNum)
    combinePartials()

    with open('corpus.json', 'w') as f:
        json.dump(corpus, f)

    with open('corpusurls.json', 'w') as f:
        json.dump(corpusurls, f)


def tokenizer(page): # tokenizes page to create index
    toks = re.findall(r'\w+', page.lower())
    stemmer = PorterStemmer()
    unstemTok = set()
    for token in toks:
        unstemTok.add(stemmer.stem(token))
        unique.add(stemmer.stem(token))
    return unstemTok


def tokenize(text): # new tokenizer for queries only
    text = text.lower()
    stemmer = PorterStemmer()
    unstemQue = []
    to = re.findall(r'\w+', text)
    for t in to:
        unstemQue.append(stemmer.stem(t))
    return unstemQue


def calculate_weight(term): # calculates tf-idf
    tf = 1
    idf = calculate_idf(term)
    return tf * idf

def calculate_idf(term): # calculates idf for tf-idf
    df = get_document_frequency(term)
    N = get_total_documents()
    if df == 0:
        return 0
    return math.log(N/df)

def get_posting_list(term): # returns posting list for index
    try:
        return index[term]
    except:
        return []

def get_document_frequency(term): # returns document frequency for a term
    try:
        return len(index[term])
    except:
        return 0

def get_total_documents(): # returns total number of documents within index
    return indexDocuments

def cosine_score(query_terms, k=5): # calculates cosine score for query
    scores = defaultdict(float)

    for term in query_terms:
        w = calculate_weight(term)
        posting_list = get_posting_list(term)
        for doc_id, tf in posting_list:
            scores[doc_id] += w * w * tf
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def searching(query):
    query = tokenize(query)
    idToScore = cosine_score(query, 5)
    ans = []
    for i in idToScore:
        ans.append(corpusurls[i[0]])
    return ans


if __name__ == '__main__':
    print('Indexing has begun, please wait....')
    indextimer = time.time()
    # invertedIndexer('/Users/David/Projects/Web Crawler/ANALYST')
    indextimerend = time.time()
    runtime = indextimerend - indextimer
    print(f"Finished Indexing in {runtime:.2f} s")

    while True:
        search = input('Enter query as -s \"query\" or \"-q\" to quit: ')
        keyword = search.split(' ')
        if keyword[0] == '-s':  # test cristina lopes
            timerStart = time.time()
            urlList = searching(search[3:]) # returns list of urls sorted by cosineScore
            for url in urlList[:5]:
                print(url)
            timeEnd = time.time()
            runtime = (timeEnd - timerStart) * 1000
            print(f"Finished query in {runtime:.4f} ms")
        elif keyword[0] == "-q":
            break
        else:
            print('Query entered incorrectly, please try again')