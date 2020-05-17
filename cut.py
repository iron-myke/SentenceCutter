import argparse
import json
import numpy as np
import re

from nltk.corpus import gutenberg

from train import load_model
with open("config.json", 'r') as config_file:
    config = json.load(config_file)

def cut(input_file, model_type):
    # preprocessing
    input_file = input_file.replace('\n', ' ')

    # extract potential end of sentence punctuation marks
    # remember their indexes to know where to cut
    candidates, indexes = extract_candidates(input_file)
    
    # load model and predict which punctuation marks end a sentence
    model = load_model(model_type)
    predictions = model.predict(candidates)
    
    # compute the cut indexes
    end_of_sentence = np.where(predictions == 1.0)
    cut_indexes = [0] + list(np.array(indexes)[end_of_sentence]) + [len(input_file)]

    # split the input text into sentences
    sentences = [
        input_file[cut_indexes[i]:cut_indexes[i+1]].strip().replace('\n', ' ') 
        for i in range(len(cut_indexes) - 1)
    ]

    return sentences

def extract_candidates(input_file):
    candidates = []
    indexes = []
    ngram_window = config['ngram_window']

    punctuation_marks_regexp = re.compile(r'[:\.;?!]{1,}')
    punctuation_marks = re.finditer(punctuation_marks_regexp, input_file)

    for p_m in punctuation_marks:
        ngram = input_file[p_m.start() - ngram_window:p_m.end() + ngram_window]
        candidates.append(ngram)
        indexes.append(p_m.end())
    return candidates, indexes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="samples/gutenberg_sample.txt", help="Input text file")
    parser.add_argument("--output", default="samples/result.txt", type=str, help="Output file path")
    parser.add_argument("--model", default='svm', type=str, choices=['svm', 'nb', 'dtc', 'sgd', 'rft'],
        help="Model used to cut the sentences (dtc, svm, sgd, nb or rft)")
    args = parser.parse_args()

    with open(args.input, 'r', encoding="utf8") as f:
        txt = f.read()
        sentences = cut(txt, args.model)
        with open(args.output, 'w', encoding="utf8") as output:
            for sentence in sentences:
                if sentence is not "":
                    output.write(sentence + "\n")


