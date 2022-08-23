import json
import nntplib
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import tensorflow as tf


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet


with open('intents.json','r') as f: # import all the intents
    intents = json.load(f)


all_words = [] # import all words
tags = [] # import tags that define the intent
xy = []

for intent in intents['intents']: # for each intent, iterate through them with the iterating variable named "intent". Enhanced forloop.
    tag = intent['tag'] # tag = the tag of the intent
    tags.append(tag) # add the tag to tags
    for pattern in intent['patterns']: # for each item in patterns
        w = tokenize(pattern) # tokenize the sentence
        all_words.extend(w) # extend the word
        xy.append((w, tag)) # add the tag to xy...label + feature???. It's these things: tokenized sentence + tag.


ignore_words = ['?', '!', '.', ','] # ignore words
all_words=[stem(w) for w in all_words if w not in ignore_words]

# essentially, lowercase, tokenize + stem, at this point


all_words  = sorted(set(all_words)) # remove dupes + sort matrix all_words
tags = sorted(set(tags)) # remove dupes + sort matrix tags

print(tags)
# print(all_words)


X_train = [] # put bag of words in here
y_train = []

for(pattern_sentence, tag) in xy: # loop over xy array
    bag = bag_of_words(pattern_sentence, all_words) # create a bag of words using the tokenized pattern sentence, then all_words.
    X_train.append(bag) #add the bag_of_words to the xtrain

    label = tags.index(tag) # the label is the index of the tag.
    # pretty sure this is one hot encoding

    # ['delivery', 'funny', 'goodbye', 'greeting', 'items', 'payments', 'thanks']
    # if the label is 'funny', it'll be like [0, 1, 0, 0, 0, 0, 0]

    y_train.append(label) # bruh i have no idea, crossentropyloss


X_train = np.array(X_train)
y_train = np.array(y_train)

# create pytorch dataset



#hyperparameters
batch_size = 8
hidden_size = 65
output_size = len(tags) #number of classes/texts we have
input_size = len(X_train[0]) # length of the label
print(input_size, len(all_words)) 
print(output_size, tags) # prints out seven, because there are 7 texts in tags
learning_rate = 0.001
num_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) # number of samples
        self.x_data = X_train # get feature
        self.y_data = y_train # get labels

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index] # return the features and labels

    def __len__(self):
        return self.n_samples # get length of samples



dataset = ChatDataset() # dataset is the chat dataset
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True, num_workers=0)
    #heart of the pytorch data utility
    #represents python iterable over dataset
    # parameters:
    #   dataset = dataset from which to load the data
    #   batch_size = how many samples per batch to load. It means the number of training examples utilized in one iteration.
    #   shuffle = if you want to have the data reshuffled after every epoch
    #   num_workers: how many subprocesses are used for data loadining. 
        # In Python, a process is an instance of the Python interpreter that executes Python code.
        # Subprocesses run new codes by making new processes.
        # start new applications right from the program you're writing.
        # When num_workers>0, only these workers will retrieve data, main process won't.
        # So when num_workers=2 you have at most 2 workers simultaneously putting data into RAM, not 3.


print(train_loader)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# note that this code needs GPU runtime, this code checks if it's available. If it's not we simply just use cpu to run model.
# of course, cpu is not fast. Who cares.


model = NeuralNet(input_size, hidden_size, output_size).to(device) # push the NN model to the device



# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(num_epochs): #actual loop for training data, ask anish to walk me through it.
    for (words, labels) in train_loader:
        words = words.to(device) # i think we just need these so that we can run them through train code
       # print(words)

        labels = labels.to(device)

        # forward
        outputs = model(words) # train the data
        loss = criterion(outputs, labels)

        #backward prop, optimization
        optimizer.zero_grad() # training loop
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')


data = { # dictionary to save different things like model state
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth" # pth for pytorch
torch.save(data, FILE) # save to pickled file

print(f'training complete. file saved to {FILE}')
