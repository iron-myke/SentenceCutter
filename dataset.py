import csv
import re
import json

from nltk.corpus import gutenberg, brown
from nltk.corpus.reader.plaintext import read_blankline_block, concat
from pandas import read_csv


with open("config.json", 'r') as config_file:
    config = json.load(config_file)

# writes a csv file with (x, y) pair for all potential end of sentence punctuation marks  
def write_dataset():
    sentences = load_sentences(config['gutenberg'])
    ngram_window = config['ngram_window']

    # get all potential end of sentence punctuation marks 
    # if some are neighbors, then consider the group as one
    candidates = re.finditer(re.compile(r'[\.:;?!]{1,}'), sentences)
    
    data_filename = config['dataset']
    with open(data_filename, 'w', newline='', encoding='utf-8') as data_file:
        fieldnames = ['X', 'Y']
        writer = csv.DictWriter(data_file, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            if c.end() + ngram_window <= len(sentences):
                # extract ngram and remove the linebreak character to avoid overfitting
                ngram = sentences[c.start() - ngram_window:c.end() + ngram_window].replace('\n', ' ')
                # if it is an end of sentence, y = 1.0
                y = 1.0 if sentences[c.end()] == '\n' else 0.0 
                writer.writerow({"X": ngram, "Y": y})


# loads and rebuilds sentences from the brown or gutenberg dataset.
# returns a string with all sentences separated by linebreaks
def load_sentences(gutenberg_sentences = False):
    # Gutenberg Dataset
    if gutenberg_sentences:
        sentences = sentences_from_corpus(gutenberg)
        sentences = [s.replace('\n', ' ') for s in sentences]
    
    # Brown Dataset
    else:
        sentences = brown.sents()
        sentences = [fuse_sentence(sentence) for sentence in sentences]
        
    return '\n'.join(sentences)

# from a list of words and punctiation marks, it returns a sentence
def fuse_sentence(list_sentence):
    sentence = ''
    for word in list_sentence:
        if word not in ['.', ',', ':', ';', '!', '?']:
            sentence += ' '
        sentence += word
    return sentence[1:]

# loads the dataset from the csv files
def load_dataset():
    return read_csv(config['dataset']) 

# build sentences from gutenberg corpus
# from https://stackoverflow.com/questions/30170556/retrieving-sentence-strings-from-nltk-corpus
def sentences_from_corpus(corpus, fileids = None):
    def read_sent_block(stream):
        sents = []
        for para in corpus._para_block_reader(stream):
            sents.extend([s.replace('\n', ' ') for s in corpus._sent_tokenizer.tokenize(para)])
        return sents

    return concat([corpus.CorpusView(path, read_sent_block, encoding=enc)
                   for (path, enc, fileid)
                   in corpus.abspaths(fileids, True, True)])


if __name__ == "__main__":
    write_dataset()