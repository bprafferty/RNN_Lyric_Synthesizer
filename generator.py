#Project: Artificial Led Zeppelin
#Authors: Brian Rafferty, Kaiyu Chen, Richard Estes

import numpy as npy
import torch
from torch import nn
import torch.nn.functional as nnF
import recurrentNN as RNN
import networkTrainer as nnT

#open & read file containing Led Zep Lyrics
with open('./ledZep.txt', 'r') as userFile:
    lyrics = userFile.read()

#create set of every character seen
characterSet = tuple(set(lyrics))

#encode characterSet into dictionary with int/char pairs
intDict = dict(enumerate(characterSet))

#encode characterSet into dictionary with char/int pairs
charDict = {currChar: ii for ii, currChar in intDict.items()}

#convert lyrics to their integer value
convertChars = npy.array([charDict[currChar] for currChar in lyrics])

#instantiate network paramaters                  
hiddenLayers = 2
hiddenStates = 512
recurrentNetwork = RNN.recurrentNN(characterSet, hiddenStates, hiddenLayers)

#training parameters
numSequences = 10
sequenceLength = 5
trainingSessions = 20

#begin training the network
nnT.teachNetwork(recurrentNetwork, convertChars, trainingSessions, numSequences, sequenceLength, learningRate=0.001, formatPrinter=50)

#standard procedure to save model for prediction 
modelDict = {'hiddenStates': recurrentNetwork.hiddenStates,
              'hiddenLayers': recurrentNetwork.hiddenLayers,
              'state_dict': recurrentNetwork.state_dict(),
              'tokens': recurrentNetwork.characterSet}

with open('ledZep.recurrentNetwork', 'wb') as f:
    torch.save(modelDict, f)
    
#predicts a character, returning it and the hidden state
def chooseNext(recurrentNetwork, char, curHidden=None, bestChar=None):
        #convert input char to one-hot and tensor
        inputChar = npy.array([[recurrentNetwork.charDict[char]]])
        inputChar = nnT.createOneHotVector(inputChar, len(recurrentNetwork.characterSet))
        tensorInputs = torch.from_numpy(inputChar)
        
        #select optimal hidden state
        curHidden = tuple([each.data for each in curHidden])

        #generate output
        generatedChar, curHidden = recurrentNetwork(tensorInputs, curHidden)

        #grab calculated probability using softmax
        charProbability = nnF.softmax(generatedChar, dim=1).data
        
        #choose optimal character for prediction
        if bestChar is None:
            optimalChar = npy.arange(len(recurrentNetwork.characterSet))
        else:
            charProbability, optimalChar = charProbability.topk(bestChar)
            optimalChar = optimalChar.numpy().squeeze()
        
        #use numpy.random to make the next character different from original
        charProbability = charProbability.numpy().squeeze()
        char = npy.random.choice(optimalChar, p=charProbability/charProbability.sum())
        
        # return the vector value of the predicted char and the hidden state
        return recurrentNetwork.intDict[char], curHidden
        
#method uses trained model to synthesize new lyrics
def generateLyrics(recurrentNetwork, outputLength, initialLyric='The', bestChar=None):
    
    recurrentNetwork.eval() 
    #iterate through the initialLyric characters
    characterSet = [currChar for currChar in initialLyric]
    curHidden = recurrentNetwork.init_hidden(1)
    for currChar in initialLyric:
        outputChar, curHidden = chooseNext(recurrentNetwork, currChar, curHidden, bestChar=bestChar)

    characterSet.append(outputChar)
    
    #join all predicted characters together
    for index in range(outputLength):
        outputChar, curHidden = chooseNext(recurrentNetwork, characterSet[-1], curHidden, bestChar=bestChar)
        characterSet.append(outputChar)

    return ''.join(characterSet)
    
#create synthetic lyrics and print results to console
print(generateLyrics(recurrentNetwork, 1000, initialLyric='The', bestChar=5))