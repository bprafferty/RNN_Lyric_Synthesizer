#Project: Artificial Led Zeppelin
#Authors: Brian Rafferty, Kaiyu Chen, Richard Estes

import numpy as npy
import torch
from torch import nn

#method to encode vectors
def createOneHotVector(arr, n_labels):
    #initialize size
    nullVector = npy.zeros((npy.multiply(*arr.shape), n_labels), dtype=npy.float32)
    
    #put 1s in spot with number
    nullVector[npy.arange(nullVector.shape[0]), arr.flatten()] = 1.
    
    #reshape to match every other vector
    finalVectorShape = nullVector.reshape(list(arr.shape)+ [n_labels])
    return finalVectorShape

#compress input into groups for training
def determineBatchSize(arr, numSequences, sequenceLength):
    completeBatch = numSequences * sequenceLength
    
    #maximum groups
    iterationsPossible = len(arr)//completeBatch
    
    #remove excess characters for next iteration
    arr = arr[:iterationsPossible * completeBatch]

    #reshape the rows to equal lengths
    arr = arr.reshape((numSequences, -1))
    
    #iterate through the sequences
    for n in range(0, arr.shape[1], sequenceLength):

        #the input characters
        inputChar = arr[:, n:n+sequenceLength]

        #the target characters
        targetChar = npy.zeros_like(inputChar)
        
        try:
            targetChar[:, :-1], targetChar[:, -1] = inputChar[:, 1:], arr[:, n+sequenceLength]
        except IndexError:
            targetChar[:, :-1], targetChar[:, -1] = inputChar[:, 1:], arr[:, 0]

        yield inputChar, targetChar

#train the model
def teachNetwork(recurrentNetwork, data, trainingSessions=10, numSequences=10, sequenceLength=50, learningRate=0.001, gradientClipping=5, holdOutData=0.1, formatPrinter=10):
    #instantiate built in train method
    recurrentNetwork.train()
    
    #instantiate classification
    criterion = nn.CrossEntropyLoss()

    #use built in Adam algorithm to yield best character
    optimumChar = torch.optim.Adam(recurrentNetwork.parameters(), lr=learningRate)
    
    #collection of expected values
    valIndex = int(len(data)*(1-holdOutData))
    data, valData = data[:valIndex], data[valIndex:]
    
    counter = 0
    characterSetLength = len(recurrentNetwork.characterSet)
    for index in range(trainingSessions):

        #find current hidden state
        curHidden = recurrentNetwork.init_hidden(numSequences)
        
        for inputChar, targetChar in determineBatchSize(data, numSequences, sequenceLength):
            counter += 1
            
            #one-hot encode vectors and turn them into torch tensors
            inputChar = createOneHotVector(inputChar, characterSetLength)
            inputs, targets = torch.from_numpy(inputChar), torch.from_numpy(targetChar)
            
            #instantiate current hidden state by looking at all previous states
            curHidden = tuple([allPrevious.data for allPrevious in curHidden])

            #use built in zero accumulated gradients
            recurrentNetwork.zero_grad()
            
            #retrieve generated character and hidden state value
            generatedChar, curHidden = recurrentNetwork(inputs, curHidden)
            
            #determine loss and use built in backward() for backprop
            loss = criterion(generatedChar, targets.view(numSequences*sequenceLength).long())
            loss.backward()

            #clip_grad_norm squashes data between -1 and 1 to avoid exponential increases
            nn.utils.clip_grad_norm_(recurrentNetwork.parameters(), gradientClipping)
            optimumChar.step()
            
            #print loss statistics
            if (counter % formatPrinter == 0):
                hiddenVal = recurrentNetwork.init_hidden(numSequences)
                lossesArray = []
                recurrentNetwork.eval()
                for inputChar, targetChar in determineBatchSize(valData, numSequences, sequenceLength):
                    #one-hot encode vectors and turn them into torch tensors
                    inputChar = createOneHotVector(inputChar, characterSetLength)
                    inputChar, targetChar = torch.from_numpy(inputChar), torch.from_numpy(targetChar)
                    
                    #update values for hidden states
                    hiddenVal = tuple([allPrevious.data for allPrevious in hiddenVal])
                    
                    inputs, targets = inputChar, targetChar
                    
                    generatedChar, hiddenVal = recurrentNetwork(inputs, hiddenVal)
                    calculatedLoss = criterion(generatedChar, targets.view(numSequences*sequenceLength).long())
                
                    lossesArray.append(calculatedLoss.item())

                #continue training after current loss statistics are determined
                recurrentNetwork.train()
                
                print("Training Step: {}/{}".format(index+1, trainingSessions),
                      "Iteration Number: {}".format(counter),
                      "Val Loss: {:.4f}".format(npy.mean(lossesArray)))