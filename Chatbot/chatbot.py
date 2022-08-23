import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize # import tokenization and bag of words algorithm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if your device supports GPU runtime, do it

with open('intents.json', 'r') as f:
    intents = json.load(f) # load up the training data


FILE = "data.pth"
data = torch.load(FILE) # train the data 

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device) # 
model.load_state_dict(model_state) # knows our learned parameters
model.eval() # not training, evaluates


bot_name = "Sam"
print("Let's chat! type 'quit' to exit (greetings, farewells, thanks, items, payments, deliveries, jokes).")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    
    # first tokenize, then bag of words, lastly
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)

    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()] #greeting, etc

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]: # check if tag matches
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                intent['patterns'].append(sentence) # im trying to push the responses to the intents.json if the tag fits


    else:
        print(f"{bot_name}: I do not understand...")




