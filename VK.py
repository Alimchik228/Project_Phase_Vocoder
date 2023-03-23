import wave
from scipy.fft import fft, ifft
from scipy.io import wavfile
from glob import glob
import numpy as np
import sys

test_mono = glob(sys.argv[1])
fn, x = wavfile.read(test_mono[0])
x = np.array(x)

def hann(winSize):
    M = winSize*2+1
    wn = list()
    for i in range(1,M):
        wn.append(0.5 - 0.5*np.cos(2*np.pi*i/(M-1)))
    return np.array(wn)

def createFrames(x, hop, windowSize):

    numberSlices = round((len(x)-windowSize)/hop)
    
    x = x[0:(numberSlices*hop+windowSize)]
    
    vectorFrames = np.ones((round(len(x)/hop),windowSize))
    
    for index in range(1, numberSlices+1):
    
        indexTimeStart = (index-1)*hop + 1
        indexTimeEnd = (index-1)*hop + windowSize+1
        
        vectorFrames[index] = x[indexTimeStart:indexTimeEnd]
    return np.array(vectorFrames), np.array(numberSlices)

def fusionFrames(framesMatrix, hop):

    sizeMatrix = framesMatrix.shape
    
    numberFrames = sizeMatrix[0]

    sizeFrames = sizeMatrix[1]

    vectorTime = np.zeros((numberFrames*hop-hop+sizeFrames,1))
    
    timeIndex = 0
    vectorTime = np.array(vectorTime)
    framesMatrix = np.array(framesMatrix)

    for index in range(numberFrames):
      
        vectorTime[timeIndex:timeIndex+sizeFrames] += framesMatrix[index].reshape(-1, 1)
        
        timeIndex = timeIndex + hop

    return vectorTime

def pitchShift(inputVector, windowSize, hopSize, alpha):
    x = inputVector
    winSize = windowSize
    hop = hopSize
    hopOut = round(alpha*hop)
    wn = hann(winSize)
    wn = np.array(range(0,len(wn),2)) 
    #x = np.array([np.ones((hop*3, 1)), x])
    y,numberFramesInput = createFrames(x,hop,winSize)
    numberFramesOutput = numberFramesInput
    outputy = np.ones((numberFramesOutput,winSize))
    phaseCumulative = 0
    previousPhase = 0
    for index in range(numberFramesInput):
        #Analysis
        currentFrame = y[index,:]
        
        currentFrameWindowed = currentFrame * wn.T / np.sqrt(((winSize/hop)/2))
        currentFrameWindowedFFT = fft(currentFrameWindowed)
        magFrame = np.absolute(currentFrameWindowedFFT)
        phaseFrame = np.angle(currentFrameWindowedFFT)
        #Processing
        deltaPhi = phaseFrame - previousPhase
        previousPhase = phaseFrame

        deltaPhiPrime = deltaPhi - hop * 2*np.pi*np.array(range(winSize))/winSize

        deltaPhiPrimeMod = (deltaPhiPrime+np.pi)//(2*np.pi) - np.pi

        trueFreq = 2*np.pi*np.array(range(winSize))/winSize + deltaPhiPrimeMod/hop
        ##hui
        phaseCumulative = phaseCumulative + hopOut * trueFreq
        
        ##Get the magnitude
        outputMag = magFrame
        
        ##Produce output frame
        mnimi =[complex(0, i) for i in phaseCumulative]
        outputFrame = ifft(outputMag * np.exp(mnimi)).real
        
        ##Save frame that has been processed
        outputy[index,:] = outputFrame * wn.T / np.sqrt(((winSize/hopOut)/2))
        #outputy[index * hopOut:index * hopOut +winSize] += outputFrame * wn.T / np.sqrt(((winSize/hopOut)/2))
        ##Finalize
    outputTimeStretched = fusionFrames(outputy,hopOut)
    return outputTimeStretched

y2 = pitchShift(x,1024,128,float(sys.argv[3]))
y2 = np.int16(y2*(32767/y2.max()))
wavfile.write(sys.argv[2], fn, y2)

