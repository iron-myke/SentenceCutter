# Sentence Cutter
Here is a simple sentence cutter based on the sklearn and nltk libraries.

## Requirements
python 3.7

packages: sklearn, joblib, ntlk (Brown and Gutenberg corpuses), pandas, numpy

## Cut a text
The script **cut.py** basically takes as input a text file, extracts from it potential end of sentence punctuation 
marks with their indexes, and using one the trained models, it predicts which ones really end sentences. 
Then, using the indexes, it cuts the text into sentences and writes them in a file.

```bash
python3 cut.py --input TEXT_FILE_PATH --output OUTPUT_FILE_PATH --model MODEL 
```

* TEXT_FILE_PATH is the path to the file containing the text to be cut into sentences (default: **samples/wiki_sample_1.txt**)
* OUTPUT_FILE_PATH is the paths to the output file in which will be written the sentences (one per line) (default: **samples/result.txt**)
* MODEL is the model used to cut the text, it can be svm, rft, nb, sgd or dtc (default: **svm**)

In the folder samples, there are 3 text extracts that can be used to test this script. By default, 
the file **samples/result.txt** contains the sentences from **samples/wiki_sample_1.txt**

## Data/Training/Testing
Some pre-trained models are given in the folder **models**. Here are the scripts used to design and train them.

### Dataset
The Brown and Gutenberg corpuses (from the nltk package) can be used alternatively to build the dataset (cf. config file). 
The script **dataset.py** extracts all potential end of sentence punctuation marks (?;,.!) with their surrounding characters,
 and a label indicating if they truly end a sentence. It writes these (x, y) pairs in a csv file.

```bash
python3 dataset.py
```

### Models

I used the sklearn library to design several similar models. I have combined a TfidVectorizer with some different simple Binary Classifiers:
* rft: RandomForestClassifier
* svm: LinearSVC
* nb: MultinomialNB
* sgd: SGDClassifier
* dtc: DecisionTreeClassifier

### Training
The script **train.py** loads the dataset and cuts it into training and testing sets. Then it trains, tests, and displays metrics for the 5 models.
```bash
python3 train.py
```

### Config file

```
{
    "ngram_size": 7, # size of the bags of words used to build the dataset
    "tfid_max_features": 10000, # TfidVectorizer param
    "models": { # saved models file paths
        "nb": "models/nb.pkl",
        "dtc": "models/dtc.pkl",
        "rft": "models/rft.pkl",
        "svm": "models/svm.pkl",
        "sgd": "models/sgd.pkl"
    },
    "dataset": "dataset/dataset.csv", # dataset csv file path
    "gutenberg": false # use the gutenberg corpus instead of the brown corpus
}
```

### Results
The Brown corpus is a lot better for acronyms and numbers. But, it considers all ';' as end of sentence marks,
because of the way sentences are cut. It could be undone but as sentences were cut that way by those who have built this corpus, 
I let it as it is. The Gutenberg corpus is better for novel style sentences (long sentences with ';' and '"') 
but achieves poorly on acronyms and numbers.

Overall, on both datasets and without tuning, all models achieve similar performance on the testing set (f1_score and accuracy around 0.97-0.99).
LinearSVC (svm) is the best model, that is why it is the default model in the cutter script.


## Bonus
Here is how I would adapt the cut.py script to cut html formatted books
* use a html parser to extract only the content of the html tags, in a list, associated with their start/end indexes in the html book.
* extract the ngrams (potential end of sentence punctuation marks) from theses parts
* use a model to predict if they end a sentence
* Using the indexes from the html parser, insert sentence tags after each predicted end of sentence (an end tag and then a start tag)