#Project: Artificial Led Zeppelin
#Authors: Brian Rafferty, Kaiyu Chen, Richard Estes

import torch
from torch import nn

#create class for our RNN model to be used like an object
class recurrentNN(nn.Module):
    def __init__(self, characterSet, hiddenStates=256, hiddenLayers=2, dropProbability=0.5, learningRate=0.001):
        super(recurrentNN,self).__init__()
        self.dropProbability = dropProbability
        self.hiddenLayers = hiddenLayers
        self.hiddenStates = hiddenStates
        self.learningRate = learningRate

        #build same data structures for internal use
        self.characterSet = characterSet
        self.intDict = dict(enumerate(self.characterSet))
        self.charDict = {curChar: index for index, curChar in self.intDict.items()}
        
        #define the built in LSTM model with object parameters
        self.lstm = nn.LSTM(len(self.characterSet), hiddenStates, hiddenLayers, 
                            dropout=dropProbability, batch_first=True)
        
        #use built in method to get dropout layer
        self.dropout = nn.Dropout(dropProbability)
        
        #connect model
        self.fc = nn.Linear(hiddenStates, len(self.characterSet))

    def forward(self, x, hiddenVal):        
        #get the outputs and the new hiddenVal state from the lstm
        r_output, hiddenVal = self.lstm(x, hiddenVal)
        
        #pass through a dropout layer
        characterOutput = self.dropout(r_output)
        
        #view() used to collect LSTM outputs
        characterOutput = characterOutput.contiguous().view(-1, self.hiddenStates)
        
        #put the input character through the fully-connected layer
        characterOutput = self.fc(characterOutput)
        
        #return the final output and the hiddenVal state
        return characterOutput, hiddenVal
    
    
    def init_hidden(self, numSequences):
        #create tensors based upon dynamic sizes of hidden states/layers and number of sequences
        stateProbability = next(self.parameters()).data
        
        hiddenVal = (stateProbability.new(self.hiddenLayers, numSequences, self.hiddenStates).zero_(), stateProbability.new(self.hiddenLayers, numSequences, self.hiddenStates).zero_())
        
        #return LSTM output
        return hiddenVal