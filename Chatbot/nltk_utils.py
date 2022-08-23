import numpy as np

import nltk

# nltk.download('punkt') ... run this only one time

from nltk.stem.porter import PorterStemmer # import stem function
stemmer = PorterStemmer() # 



def tokenize(sentence):
    return nltk.word_tokenize(sentence) # tokenize a sentence with this function

def stem(word): 
    return stemmer.stem(word.lower()) # stem the word.

def bag_of_words(tokenized_sentence, all_words):  #bag of words algorithm.
    tokenized_sentence = [stem(w) for w in tokenized_sentence] # stem the tokenized sentence
    bag = np.zeros(len(all_words),dtype = np.float32)  # make the bag. First we make a matrix full of zeroes.
    for idx, w, in enumerate(all_words): # this is bascially making the array of ones and zeroes...if w is in the "all_words" pool, bag it
        if w in tokenized_sentence:
            bag[idx] = 1.0
        #check if word is in token sentence


    return bag
# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)


words = ["organize", "organizes", "organizing"] # example: this is an array of words.
stemmed_words = [stem(w) for w in words] # stem "words" array and add each one to an array...this will contain the word "organ"
# print(stemmed_words) #