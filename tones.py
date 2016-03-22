#################################################################################################
# To Do
# 1. Fix blips between loops. just amke a long array and trim to size rather than loop.
#    sound = pygame.mixer.Sound(self.toneList[toneToPlay])
#    sound.play(loops = int(app.getColLen()[col])) #A better way to do this would be to create a long array and trim to length rather than loop.
###########################################################################################

#Imports
########
import pygame
from pygame.locals import *
import math
import numpy as np
from Tkinter import *
import matplotlib.pyplot as plt
from random import randint, shuffle

#For wavebender
import sys
import wave
import struct
#import random
import argparse
from itertools import count, islice
try:
    from itertools import zip_longest
except ImportError:
    from itertools import imap as map
    from itertools import izip as zip
    from itertools import izip_longest as zip_longest
try:
    stdout = sys.stdout.buffer
except AttributeError:
    stdout = sys.stdout


#Globals
########
numTones = 10
#freqList = [128, 256, 512, 1024, 2048]
#freqList = [440, 493.88, 523.25, 587.33, 659.25, ]
lengthList = [0.25, 0.5, 1.0]
reWriteToneFile = 0
graphWaveform=0


class App():

    def __init__(self,master):

        #Launch the frame
        self.frame = Frame(master)
        self.frame = Frame(master)
        self.row0Frame = Frame(self.frame) #Create a frame for its own grid
        self.row0Frame.grid(row=0, column=0, columnspan=15, sticky=W)
        #self.row2Frame = Frame(self.frame) #Create a frame for its own grid
        #self.row2Frame.grid(row=2, column=0, columnspan=15, sticky=W)
        self.frame.pack()     
        
        #Row 0
        btn = Button(self.row0Frame, text='Go', command= lambda a=1: self.go(a))
        btn.grid(row=0, column=0)
        l = Label(self.row0Frame, text=' ')
        l.grid(row=0,column=1)

        l = Label(self.row0Frame, text=' ')
        l.grid(row=0,column=len(lengthList)+2)
        btn = Button(self.row0Frame, text='Rand', command= lambda a=1: self.clkRandom(a))
        btn.grid(row=0, column=len(lengthList)+3)
        btn = Button(self.row0Frame, text='Reset', command= lambda a=1: self.clkReset(a))
        btn.grid(row=0, column=len(lengthList)+4)
        btn = Button(self.row0Frame, text='Play', command= lambda a=1: waveBender.playViolin(a))
        btn.grid(row=0, column=len(lengthList)+5)

        #Note Rows
        self.checkBoxList = []
        self.checkBoxVarList = []
        for i in range(len(toneGen.getNameFrequencies())):
            tempList = []
            tempVarList = []
            #print i),toneGen.getNameFrequencies()[i]
            l = Label(self.frame, text=toneGen.getNameFrequencies()[i])
            l.grid(row=i+1,column=0)
            for j in range(numTones):                
                v = IntVar()
                c = Checkbutton(self.frame, variable=v)#, command= lambda a=i, b=j: self.cb(a,b))
                c.var = v
                tempList.append(c)
                tempVarList.append(c.var)
                c.grid(row=i+1, column=j+1)
            self.checkBoxList.append(tempList)
            self.checkBoxVarList.append(tempVarList)

        #Up Down Rows
        self.colLen = np.zeros(numTones)
        self.colLenLabel = []
        for j in range(-1,numTones):
            if j>=0:
                self.v = StringVar()
                self.colLenLabel.append(self.v)
                #print str(self.colLen[j]+1)[0]
                self.colLenLabel[j].set(str(self.colLen[j]+1)[0])
                l = Label(self.frame, textvariable=self.colLenLabel[j])
                l.grid(row=len(toneGen.getNameFrequencies())+i+2, column=j+1)
            btn = Button(self.frame, text='^', command= lambda row=0, col=j: self.clkUpDown(row,col))
            btn.grid(row=len(toneGen.getNameFrequencies())+i+3, column=j+1)
            btn = Button(self.frame, text='v', command= lambda row=1, col=j: self.clkUpDown(row,col))
            btn.grid(row=len(toneGen.getNameFrequencies())+i+4, column=j+1)            

    def go(self,a):

        for i in range(numTones):
            #for j in range(5):
            #    self.activeList[i].set(j)
            #self.activeList[i].set('.')
            freqToPrint=[]
            #print self.checkBoxVarList
            for j in range(len(toneGen.getNumFrequencies())):
                if self.checkBoxVarList[j][i].get():
                    freqToPrint.append(toneGen.getNumFrequencies()[j])
            #print i, freqToPrint
            if len(freqToPrint)>0:
                toneGen.playFreq(freqToPrint,i)
        #print ''

##    def cb(self,i,j):
##        print i,j
##        self.go(1)

    def clkRandom(self,a):
        for i in range(numTones):
            numToCheck = randint(0,3)
            #print numToCheck
            shuffledList = range(len(toneGen.getNameFrequencies()))
            #print shuffledList
            shuffle(shuffledList)
            #print shuffledList[0:numToCheck]
            for j in shuffledList[0:numToCheck]:
                self.checkBoxVarList[j][i].set(1)

    def clkReset(self,a):
        for i in range(numTones):
            for j in range(len(toneGen.getNameFrequencies())):
                self.checkBoxVarList[j][i].set(0)

    def clkUpDown(self,row,col):
        #print row,col,self.colLen[col]
        if col == -1 and row == 0:
            for j in range(numTones):
                if self.colLen[j]<8:
                    self.colLen[j] += 1
                    self.colLenLabel[j].set(str(self.colLen[j]+1)[0])
        elif col == -1 and row == 1:
            for j in range(numTones):
                if self.colLen[j]>0:
                    self.colLen[j] -= 1
                    self.colLenLabel[j].set(str(self.colLen[j]+1)[0])
        elif row == 0 and self.colLen[col]<8:
            self.colLen[col] += 1
            self.colLenLabel[col].set(str(self.colLen[col]+1)[0])
        elif row == 1 and self.colLen[col]>0:
            self.colLen[col] -= 1
            self.colLenLabel[col].set(str(self.colLen[col]+1)[0])

    def getColLen(self):
        return self.colLen
            

class ToneGen():
    def __init__(self):

        #Setup Pygame
        #############
        bits = 16
        pygame.mixer.pre_init(44100, -bits, 2)
        pygame.mixer.init()

        #Create Tone File
        #################

        #Import notes.csv
        self.importFreqList = []
        self.freqNames = []
        self.freqNumbers = []
        for line in open('notes.csv', 'r'):
            if '#' in line:
                continue
            elif 'Note' in line:
                continue
            else:
                lineSplit = line.split(',')
                if float(lineSplit[1])>=440:
                    if float(lineSplit[1])<880:
                        self.importFreqList.append(float(lineSplit[1]))
                        self.freqNames.append(lineSplit[0])

        if reWriteToneFile:

            print '\nWriting new tone file'

            #List of tones
            self.toneList=[]
            outFile = ''

            #Setup binary to number conversion
            #print len(importFreqList), 2 ** len(importFreqList)-1
            num = 2 ** len(self.importFreqList)-1
            numCol=len(bin(num)[2:])
            #max_sample = 2**(bits - 1) - 1
            #print max_sample
            #Scale max sample by a smidge to prevent noise
            max_sample = int((2**(bits - 1) - 1) * .99)
            #print max_sample

            for i in range(1,num+1):
                tempFreqList = []
                tempBin = bin(i)[2:]
                for k in range(numCol-len(tempBin)):
                    tempBin = '0'+tempBin
                for j in range(numCol):
                    if tempBin[j] == '1':
                        tempFreqList.append(self.importFreqList[j])
                #print i, tempBin, tempFreqList

                sample_rate = 44100
                duration = 0.1
                n_samples = int(round(duration*sample_rate))

                #Print statement
                print 'Gen:',tempFreqList#,len(str(tempFreqList))
                if len(str(tempFreqList)) > len(outFile):
                    outFile = str(tempFreqList)
                    #print ' ',outFile

                #setup our numpy array to handle 16 bit ints, which is what we set our mixer to expect with "bits" up above
                buf = np.zeros((n_samples, 2), dtype = np.int64)

                for frequency in tempFreqList:

                    tempBuf = np.zeros((n_samples, 2), dtype = np.int64)
                    for s in range(n_samples):
                        t = float(s)/sample_rate    # time in seconds

                        #grab the x-coordinate of the sine wave at a given time, while constraining the sample to what our mixer is set to with "bits"
                        tempBuf[s][0] = int(round(max_sample*math.sin(2*math.pi*frequency*t)))
                    
                    buf += tempBuf #Create additive array
                    #sound = pygame.mixer.Sound(tempBuf.astype(np.int16)) #Play sound, need to pass 16 bit array
                    #sound.play(loops = 0)
                    #while pygame.mixer.get_busy(): #Wait until mixer.play is done
                    #    continue
                    #plt.plot(tempBuf[0:500])

                #Scale the additive array back to 16bit max values
                tempMax = buf.max()
                buf *= max_sample
                buf /= tempMax

                #Play sound, need to pass 16 bit array
                self.toneList.append(buf.astype(np.int16))
                
            #Save tone file
            dic = {str(i) : self.toneList[i] for i in range(len(self.toneList))} #Need to make a dictionary of numpy arrays to save the order
            #np.savez('tones/'+outFile, *self.toneList[:len(self.toneList)])
            np.savez('tones/'+outFile, **dic)
            del self.toneList

        #Load tone file
        ###############
        unOrderedList = []
        keyWordList = []
        outFile = '[440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]'
        npzFile = np.load('tones/'+outFile+'.npz')
        for n in npzFile:
            keyWordList.append(n)
            unOrderedList.append(npzFile[n])
        #Create an ordered list by sorted keywords
        self.toneList = [unOrderedList[keyWordList.index(str(i))] for i in range(len(keyWordList))]

    def playFreq(self,playFrequencyList,col):

        binary=''
        #print playFrequencyList
        for freq in toneGen.getNumFrequencies():
            if freq in playFrequencyList:
                binary += '1'
            else:
                binary += '0'
         
        toneToPlay = int(binary, 2)-1
        #print playFrequencyList, toneToPlay

        #Play sound, need to pass 16 bit array
        sound = pygame.mixer.Sound(self.toneList[toneToPlay])
        sound.play(loops = int(app.getColLen()[col])) #A better way to do this would be to create a long array and trim to length rather than loop.
        while pygame.mixer.get_busy(): #Wait until mixer.play is done
            continue
        if graphWaveform:
            plt.plot(self.toneList[toneToPlay][0:500,0])
            plt.show()

    def playFreqOld(self,frequencyList):

        #Vars
        #####
        bits = 16
        duration = 0.5          # in seconds
        #frequencyList = [256, 512]
        sample_rate = 44100
        n_samples = int(round(duration*sample_rate))

        #setup our numpy array to handle 16 bit ints, which is what we set our mixer to expect with "bits" up above
        buf = np.zeros((n_samples, 2), dtype = np.int64)
        max_sample = 2**(bits - 1) - 1

        for frequency in frequencyList:

            tempBuf = np.zeros((n_samples, 2), dtype = np.int64)
            for s in range(n_samples):
                t = float(s)/sample_rate    # time in seconds

                #grab the x-coordinate of the sine wave at a given time, while constraining the sample to what our mixer is set to with "bits"
                tempBuf[s][0] = int(round(max_sample*math.sin(2*math.pi*frequency*t)))
            
            buf += tempBuf #Create additive array
            #sound = pygame.mixer.Sound(tempBuf.astype(np.int16)) #Play sound, need to pass 16 bit array
            #sound.play(loops = 0)
            #while pygame.mixer.get_busy(): #Wait until mixer.play is done
            #    continue
            #plt.plot(tempBuf[0:500])

        #Scale the additive array back to 16bit max values
        tempMax = buf.max()
        buf *= max_sample
        buf /= tempMax

        #Play sound, need to pass 16 bit array
        sound = pygame.mixer.Sound(buf.astype(np.int16))
        sound.play(loops = 0)
        while pygame.mixer.get_busy(): #Wait until mixer.play is done
            continue
        #plt.plot(buf[0:500,0])
        #plt.show()

    def getNameFrequencies(self):
        return self.freqNames

    def getNumFrequencies(self):
        return self.importFreqList


class WaveBender():

    def grouper(self, n, iterable, fillvalue=None):
        "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    def sine_wave(self, frequency=440.0, framerate=44100, amplitude=0.5,
            skip_frame=0):
        '''
        Generate a sine wave at a given frequency of infinite length.
        '''
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        for i in count(skip_frame):
            sine = math.sin(2.0 * math.pi * float(frequency) * (float(i) / float(framerate)))
            yield float(amplitude) * sine

    def square_wave(self, frequency=440.0, framerate=44100, amplitude=0.5):
        for s in sine_wave(frequency, framerate, amplitude):
            if s > 0:
                yield amplitude
            elif s < 0:
                yield -amplitude
            else:
                yield 0.0

    def damped_wave(self, frequency=440.0, framerate=44100, amplitude=0.5, length=44100):
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        return (math.exp(-(float(i%length)/float(framerate))) * s for i, s in enumerate(self.sine_wave(frequency, framerate, amplitude)))

    def white_noise(self, amplitude=0.5):
        '''
        Generate random samples.
        '''
        return (float(amplitude) * random.uniform(-1, 1) for i in count(0))

    def compute_samples(self, channels, nsamples=None):
        '''
        create a generator which computes the samples.

        essentially it creates a sequence of the sum of each function in the channel
        at each sample in the file for each channel.
        '''
        return islice(zip(*(map(sum, zip(*channel)) for channel in channels)), nsamples)

    def violin(self, amplitude=0.1):
        # simulates a violin playing G.
        return (self.damped_wave(400.0,  amplitude=0.76*amplitude, length=44100 * 5),
                self.damped_wave(800.0,  amplitude=0.44*amplitude, length=44100 * 5),
                self.damped_wave(1200.0, amplitude=0.32*amplitude, length=44100 * 5),
                self.damped_wave(3400.0, amplitude=0.16*amplitude, length=44100 * 5),
                self.damped_wave(600.0,  amplitude=1.0*amplitude,  length=44100 * 5),
                self.damped_wave(1000.0, amplitude=0.44*amplitude, length=44100 * 5),
                self.damped_wave(1600.0, amplitude=0.32*amplitude, length=44100 * 5))

    def playViolin(self,placeHolder_NotUsed):
        #Make the calls to WaveBender
        channels = (self.violin(),)
        samples = self.compute_samples(channels, 44100 * 60 * 1)
        nchannels=1
        self.write_wavefile(stdout, samples, 44100 * 60 * 1, nchannels)

    def write_wavefile(f, samples, nframes=None, nchannels=2, sampwidth=2, framerate=44100, bufsize=2048):
        "Write samples to a wavefile."
        if nframes is None:
            nframes = 0

        w = wave.open(f, 'wb')
        w.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))

        max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

        # split the samples into chunks (to reduce memory consumption and improve performance)
        for chunk in waveBender.grouper(bufsize, samples):
            frames = b''.join(b''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
            w.writeframesraw(frames)
        
        w.close()

    def write_pcm(f, samples, sampwidth=2, framerate=44100, bufsize=2048):
        "Write samples as raw PCM data."
        max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

        # split the samples into chunks (to reduce memory consumption and improve performance)
        for chunk in grouper(bufsize, samples):
            frames = b''.join(b''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
            f.write(frames)

        f.close()

##    def main():
##        parser = argparse.ArgumentParser(prog="wavebender")
##        parser.add_argument('-c', '--channels', help="Number of channels to produce", default=2, type=int)
##        parser.add_argument('-b', '--bits', help="Number of bits in each sample", choices=(16,), default=16, type=int)
##        parser.add_argument('-r', '--rate', help="Sample rate in Hz", default=44100, type=int)
##        parser.add_argument('-t', '--time', help="Duration of the wave in seconds.", default=60, type=int)
##        parser.add_argument('-a', '--amplitude', help="Amplitude of the wave on a scale of 0.0-1.0.", default=0.5, type=float)
##        parser.add_argument('-f', '--frequency', help="Frequency of the wave in Hz", default=440.0, type=float)
##        parser.add_argument('filename', help="The file to generate.")
##        args = parser.parse_args()
##
##        # each channel is defined by infinite functions which are added to produce a sample.
##        channels = ((sine_wave(args.frequency, args.rate, args.amplitude),) for i in range(args.channels))
##
##        # convert the channel functions into waveforms
##        samples = compute_samples(channels, args.rate * args.time)
##
##        # write the samples to a file
##        if args.filename == '-':
##            filename = stdout
##        else:
##            filename = args.filename
##        write_wavefile(filename, samples, args.rate * args.time, args.channels, args.bits // 8, args.rate)
##
##    if __name__ == "__main__":
##        main()



######
#Main#
######

#Launch the frame
root = Tk()
root.title("Brickman's Tone Generator")
root.resizable(0,0) #Do not allow resizing

#Init Classes
toneGen = ToneGen()
waveBender = WaveBender()
app = App(root)

#Ready set go!
root.mainloop()
#app.go()
