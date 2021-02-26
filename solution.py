import gym
import numpy
import random
import math

import sys

#Remarque : il faut installer "pip install box2d box2d-kengz"
#Soft actor-critic avec tanh sur deux gaussiennes pour lunar-lander

env = gym.make('LunarLanderContinuous-v2')
print(env.observation_space)
print(env.action_space)

#2D en action, 8D pour les états

#sigmoid going from -1 to 1
def tanh(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue

def tanhDerivative(x):
    if x>100:
        returnValue=0
    elif x<-100:
        returnValue=0
    else:
        returnValue=(1+tanh(x))*(1-tanh(x))
    return returnValue

#sigmoid going from 0 to 1
def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=0
    else:
        returnValue=math.exp(x)/(1+math.exp(x))
    return returnValue

def sigmoidDerivative(x):
    if x>100:
        returnValue=0
    elif x<-100:
        returnValue=0
    else:
        temp=1+math.exp(x)
        returnValue=math.exp(x)/(temp*temp)
    return returnValue

def softPlus(x):
    if x>100:
        returnValue=x
    elif x<-100:
        returnValue=0
    else:
        returnValue=math.log(1+math.exp(x))
    return returnValue

def softPlusDerivative(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=0
    else:
        returnValue=1/(1+math.exp(-x))
    return returnValue

        
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutput,lastLayerActivationType):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer1=sizeHiddenLayer1
        self.sizeHiddenLayer2=sizeHiddenLayer2
        self.sizeOutput=sizeOutput
        self.LastLayerActivationType=lastLayerActivationType

        #below are the weights
        self.HiddenLayer1EntryWeights=numpy.zeros([sizeHiddenLayer1,sizeInput])
        self.HiddenLayer2EntryWeights=numpy.zeros([sizeHiddenLayer2,sizeHiddenLayer1])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer2])

        #below the storage for the weights update
        self.HiddenLayer1Storage=numpy.zeros([sizeHiddenLayer1,sizeInput])
        self.HiddenLayer2Storage=numpy.zeros([sizeHiddenLayer2,sizeHiddenLayer1])
        self.LastLayerStorage=numpy.zeros([sizeOutput,sizeHiddenLayer2])

        #pour stocker la dérivation d'après les entrées
        self.DerivativesHiddenLayer2=numpy.zeros([sizeInput,sizeHiddenLayer2])
        self.DerivativesHiddenLayer1=numpy.zeros([sizeInput,sizeHiddenLayer1])
        self.DerivativesLastLayer=numpy.zeros(sizeInput)
        
        a=0.001
        
        #random initialization
        for i in range(0,sizeHiddenLayer1):
            for j in range(0,sizeInput):
                self.HiddenLayer1EntryWeights[i,j]=random.uniform(-a,a)

        for i in range(0,sizeHiddenLayer2):
            for j in range(0,sizeHiddenLayer1):
                self.HiddenLayer2EntryWeights[i,j]=random.uniform(-a,a)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer2):
                self.LastLayerEntryWeights[i,j]=random.uniform(-a,a)

        self.HiddenLayer1EntryDeltas=numpy.zeros(sizeHiddenLayer1)
        self.HiddenLayer2EntryDeltas=numpy.zeros(sizeHiddenLayer2)
        self.LastLayerEntryDeltas=numpy.zeros(sizeOutput)

        self.HiddenLayer1Output=numpy.zeros(sizeHiddenLayer1)
        self.HiddenLayer2Output=numpy.zeros(sizeHiddenLayer2)
        self.LastLayerOutput=numpy.zeros(sizeOutput)

    def output(self,x):
        for i in range(0, self.sizeHiddenLayer1):
            self.HiddenLayer1Output[i]=tanh(numpy.dot(self.HiddenLayer1EntryWeights[i],x))
        for i in range(0, self.sizeHiddenLayer2):
           self.HiddenLayer2Output[i]=tanh(numpy.dot(self.HiddenLayer2EntryWeights[i],self.HiddenLayer1Output))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]=numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output)
            if self.LastLayerActivationType[i]=="tanh":
                self.LastLayerOutput[i]=tanh(self.LastLayerOutput[i])
            elif self.LastLayerActivationType[i]=="sigmoid":
                self.LastLayerOutput[i]=sigmoid(self.LastLayerOutput[i])
            elif self.LastLayerActivationType[i]=="softPlus":
            	self.LastLayerOutput[i]=softPlus(self.LastLayerOutput[i])
            #no activation

    #écrite pour suivre le gradient
    #storage indique si on cumule les gradients (True) ou si on fait une mise-à-jour (False)
    def retropropagation(self,x,differences,pas,storage):
        self.output(x)

        #deltas computation
        for i in range(0,self.sizeOutput):
            if self.LastLayerActivationType[i]=="tanh":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                tanhDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            elif self.LastLayerActivationType[i]=="sigmoid":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                sigmoidDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            elif self.LastLayerActivationType[i]=="softPlus":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                softPlusDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            else:
            	#no activation
                self.LastLayerEntryDeltas[i]=differences[i]
                
        for i in range(0,self.sizeHiddenLayer2):
            #here usually you need a sum
            self.HiddenLayer2EntryDeltas[i]=0
            for j in range(0,self.sizeOutput):
                self.HiddenLayer2EntryDeltas[i]+=self.LastLayerEntryDeltas[j]* \
                    (1+self.HiddenLayer2Output[i])*(1-self.HiddenLayer2Output[i])*self.LastLayerEntryWeights[j,i]

        for i in range(0,self.sizeHiddenLayer1):
            #here usually you need a sum
            self.HiddenLayer1EntryDeltas[i]=0
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer1EntryDeltas[i]+=self.HiddenLayer2EntryDeltas[j]* \
                    (1+self.HiddenLayer1Output[i])*(1-self.HiddenLayer1Output[i])*self.HiddenLayer2EntryWeights[j,i]
                    
        #weights update or storage
        for i in range(0,self.sizeHiddenLayer2):
            for j in range(0,self.sizeOutput):
                self.LastLayerStorage[j,i]+=pas*self.LastLayerEntryDeltas[j]* \
                    self.HiddenLayer2Output[i]

        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer2Storage[j,i]+=pas*self.HiddenLayer2EntryDeltas[j]* \
                    self.HiddenLayer1Output[i]
            
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeInput):
                self.HiddenLayer1Storage[i,j]+=pas*self.HiddenLayer1EntryDeltas[i]*x[j]

        if storage==False:
            for i in range(0,self.sizeHiddenLayer2):
                for j in range(0,self.sizeOutput):
                    self.LastLayerEntryWeights[j,i]+=self.LastLayerStorage[j,i]/batchSize
                    self.LastLayerStorage[j,i]=0
                    
            for i in range(0,self.sizeHiddenLayer1):
                for j in range(0,self.sizeHiddenLayer2):
                    self.HiddenLayer2EntryWeights[j,i]+=self.HiddenLayer2Storage[j,i]/batchSize
                    self.HiddenLayer2Storage[j,i]=0
                    
            for i in range(0,self.sizeHiddenLayer1):
                for j in range(0,self.sizeInput):
                    self.HiddenLayer1EntryWeights[i,j]+=self.HiddenLayer1Storage[i,j]/batchSize
                    self.HiddenLayer1Storage[i,j]=0

    #méthode de dérivation par rapport aux entrées
    def derivatives(self):
        for i in range(0,self.sizeInput):
            for j in range(0,self.sizeHiddenLayer1):
                self.DerivativesHiddenLayer1[i,j]=(1+self.HiddenLayer1Output[j])*(1-self.HiddenLayer1Output[j])*self.HiddenLayer1EntryWeights[j,i]

            for j in range(0,self.sizeHiddenLayer2):
                self.DerivativesHiddenLayer2[i,j]=0
                for k in range(0,self.sizeHiddenLayer1):
                    self.DerivativesHiddenLayer2[i,j]+=(1+self.HiddenLayer2Output[j])*(1-self.HiddenLayer2Output[j])*self.HiddenLayer2EntryWeights[j,k]*self.DerivativesHiddenLayer1[i,k]

            self.DerivativesLastLayer[i]=0
            for k in range(0,self.sizeHiddenLayer2):
                self.DerivativesLastLayer[i]+=self.LastLayerEntryWeights[0,k]*self.DerivativesHiddenLayer2[i,k]

    #copy self into NN
    def copy(self,NN):
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeInput):
                NN.HiddenLayer1EntryWeights[i][j]=self.HiddenLayer1EntryWeights[i][j]

        for i in range(0,self.sizeHiddenLayer2):
            for j in range(0,self.sizeHiddenLayer1):
                NN.HiddenLayer2EntryWeights[i][j]=self.HiddenLayer2EntryWeights[i][j]

        for i in range(0,self.sizeOutput):
            for j in range(0,self.sizeHiddenLayer2):
                NN.LastLayerEntryWeights[i][j]=self.LastLayerEntryWeights[i][j]
                
#Récapitulatif des réseaux de neurones présents :
#Q1, Q2, Qtarget1, Qtarget2, pi, 
sizeInputPi=8
sizeInputQ=10 #d'abord 8 pour l'état puis 2 pour l'action 
sizeHiddenLayer1=32
sizeHiddenLayer2=32
sizeOutputPi=4 #mu1, mu2, log sigma1, log sigma2
sizeOutputQ=1

gamma=0.99

piNN=NN(sizeInputPi,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputPi,["tanh","tanh","noActivation","noActivation"])
Q1NN=NN(sizeInputQ,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputQ,["noActivation"])
Q2NN=NN(sizeInputQ,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputQ,["noActivation"])
Q1TargetNN=NN(sizeInputQ,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputQ,["noActivation"])
Q2TargetNN=NN(sizeInputQ,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputQ,["noActivation"])
Q1NN.copy(Q1TargetNN)
Q2NN.copy(Q2TargetNN)

nbEpisodes=5000
action=[0,0]
actionPrime=[0,0]
actionTilde=[0,0]
stateAction=numpy.zeros(10)
differences=[0,0,0,0]
moyenne=0
pasMax=0.0005
pasMin=0.0001
periode=100
rho=0.995
pas=0.001
alpha=0.2

tailleFenetreGlissante=50
derniersScores=[-200]*tailleFenetreGlissante
onRaffine=False
fin=False
nbSteps=0
sigma1=1.0
sigma2=1.0
batchSize=4.0
warmingEpisodes=100
logmin=-20
logmax=2

first=True
for ep in range(0, nbEpisodes):
    if ep%100==1:
        print("episode: "+str(ep))
        print(moyenne/ep)
        print("########################################################")

    state=env.reset()
    
    firstState=state
    endOfEpisode = False

    #d'abord on simule
    scoreEpisode=0
    I=1
    length=0
    while not endOfEpisode:
        length+=1
        #calcul de la moyenne glissante
        moyenneGlissante=0
        for j in range(tailleFenetreGlissante-10,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=10

        if moyenneGlissante>-50:
            onRaffine=True
            env.render()
        else:
            onRaffine=False
            
        moyenneGlissante=0
        for j in range(0,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=tailleFenetreGlissante

        if moyenneGlissante>200:
            fin=True
            break

        piNN.output(state)
        [mu1,mu2,lsigma1,lsigma2]=piNN.LastLayerOutput

        if lsigma1<logmin:
            lsigma1=logmin
        elif lsigma1>logmax:
            lsigma1=logmax
        if lsigma2<logmin:
            lsigma2=logmin
        elif lsigma2>logmax:
            lsigma2=logmax

        action[0]=mu1+math.exp(lsigma1)*random.gauss(0,1)
        action[1]=mu2+math.exp(lsigma2)*random.gauss(0,1)

        if ep<warmingEpisodes:
            action = env.action_space.sample()

        next_state, reward, endOfEpisode, info = env.step(action) 

        if endOfEpisode==True:
            done=1
        else:
            done=0
        
        if ep%batchSize==0:
            if first==True:
                storage=False
                first=False
            else:
                storage=True
        else:
            storage=True
            first=True
        #storage=False
        
        
        #compute targets for the Q functions
        piNN.output(next_state)
        [mu1Prime,mu2Prime,lsigma1Prime,lsigma2Prime]=piNN.LastLayerOutput

        if lsigma1Prime<logmin:
            lsigma1Prime=logmin
        elif lsigma1Prime>logmax:
            lsigma1Prime=logmax
        if lsigma2Prime<logmin:
            lsigma2Prime=logmin
        elif lsigma2Prime>logmax:
            lsigma2Prime=logmax
        
        actionPrime[0]=mu1Prime+math.exp(lsigma1Prime)*random.gauss(0,1)
        actionPrime[1]=mu2Prime+math.exp(lsigma2Prime)*random.gauss(0,1)
#changer std par logstd
        logdensity=-math.log(math.sqrt(2*math.pi))-lsigma1Prime-math.log(math.sqrt(2*math.pi))-lsigma2Prime \
        -(actionPrime[0]-mu1Prime)**2*math.exp(-2*lsigma1Prime)/2-(actionPrime[1]-mu2Prime)**2*math.exp(-2*lsigma2Prime)/2 -2*(math.log(2)-actionPrime[0]-math.log(1+math.exp(-2*actionPrime[0]))+math.log(2)-actionPrime[1]-math.log(1+math.exp(-2*actionPrime[1])))

        for i in range(0,8):
            stateAction[i]=next_state[i]
        for i in range(8,10):
            stateAction[i]=actionPrime[i-8]

        Q1TargetNN.output(stateAction)
        Q1Output=Q1TargetNN.LastLayerOutput[0]
        Q2TargetNN.output(stateAction)
        Q2Output=Q2TargetNN.LastLayerOutput[0]
        
        target=reward+gamma*(1-done)*(min(Q1Output,Q2Output)-alpha*logdensity)
        
        #update Q-functions
        for i in range(0,8):
            stateAction[i]=state[i]
        for i in range(8,10):
            stateAction[i]=action[i-8]
            
        Q1NN.output(stateAction)
        currentStateActionValue=Q1NN.LastLayerOutput[0]
        differences[0]=2*(target-currentStateActionValue)
        Q1NN.retropropagation(stateAction,differences,pas,storage)   

        Q2NN.output(stateAction)
        currentStateActionValue=Q2NN.LastLayerOutput[0]
        differences[0]=2*(target-currentStateActionValue)
        Q2NN.retropropagation(stateAction,differences,pas,storage)  
                                   
        #update policy
        piNN.output(state)
        [mu1Tilde,mu2Tilde,lsigma1Tilde,lsigma2Tilde]=piNN.LastLayerOutput

        if lsigma1Tilde<logmin:
            lsigma1Tilde=logmin
        elif lsigma1Tilde>logmax:
            lsigma1Tilde=logmax
        if lsigma2Tilde<logmin:
            lsigma2Tilde=logmin
        elif lsigma2Tilde>logmax:
            lsigma2Tilde=logmax
        
        gauss1=random.gauss(0,1)
        gauss2=random.gauss(0,1)
        actionTilde[0]=(mu1Tilde+math.exp(lsigma1Tilde)*gauss1)
        actionTilde[1]=(mu2Tilde+math.exp(lsigma2Tilde)*gauss2)

        for i in range(0,8):
            stateAction[i]=state[i]
        for i in range(8,10):
            stateAction[i]=tanh(actionTilde[i-8])
        Q1NN.output(stateAction)
        qvaleur1=Q1NN.LastLayerOutput[0]
        Q2NN.output(stateAction)
        qvaleur2=Q2NN.LastLayerOutput[0]

        if qvaleur1<qvaleur2:
            argmin=0
        else:
            argmin=1

        alpha2=1.0
        alpha2=alpha
        differences[0]=-alpha2*(actionTilde[0]-mu1Tilde)*math.exp(-2*lsigma1Tilde)
        differences[1]=-alpha2*(actionTilde[1]-mu2Tilde)*math.exp(-2*lsigma2Tilde)
        differences[2]=-alpha2*(-1+(actionTilde[0]-mu1Tilde)**2*math.exp(-2*lsigma1Tilde))
        differences[3]=-alpha2*(-1+(actionTilde[1]-mu2Tilde)**2*math.exp(-2*lsigma2Tilde))
        '''
        for i in range(0,8):
            stateAction[i]=state[i]
        for i in range(8,10):
            stateAction[i]=actionTilde[i-8]
        '''

        if argmin==0:
            #Q1NN.output(stateAction)
            Q1NN.derivatives()

            differences[0]=differences[0]+Q1NN.DerivativesLastLayer[8]*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0]))) 
            differences[1]=differences[1]+Q1NN.DerivativesLastLayer[9]*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))
            differences[2]=differences[2]+Q1NN.DerivativesLastLayer[8]*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0])))*gauss1*math.exp(lsigma1Tilde)
            differences[3]=differences[3]+Q1NN.DerivativesLastLayer[9]*gauss2*math.exp(lsigma2Tilde)*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))*gauss2*math.exp(lsigma2Tilde)
        else:
           # Q2NN.output(stateAction)
            Q2NN.derivatives()

            differences[0]=differences[0]+Q2NN.DerivativesLastLayer[8]*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0])))
            differences[1]=differences[1]+Q2NN.DerivativesLastLayer[9]*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))
            differences[2]=differences[2]+Q2NN.DerivativesLastLayer[8]*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0])))*gauss1*math.exp(lsigma1Tilde)
            differences[3]=differences[3]+Q2NN.DerivativesLastLayer[9]*gauss2*math.exp(lsigma2Tilde)*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))*gauss2*math.exp(lsigma2Tilde)                                                 
        '''
        if argmin==0:
            Q1NN.output(stateAction)
            Q1NN.derivatives()

            differences[0]=differences[0]+Q1NN.DerivativesLastLayer[8]*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0]))) *tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))
            differences[1]=differences[1]+Q1NN.DerivativesLastLayer[9]*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1]))) *tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))
            differences[2]=differences[2]+Q1NN.DerivativesLastLayer[8]*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0])))*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))
            differences[3]=differences[3]+Q1NN.DerivativesLastLayer[9]*gauss2*math.exp(lsigma2Tilde)*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))*gauss2*math.exp(lsigma2Tilde)*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))
        else:
            Q2NN.output(stateAction)
            Q2NN.derivatives()

            differences[0]=differences[0]+Q2NN.DerivativesLastLayer[8]*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0]))) *tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))
            differences[1]=differences[1]+Q2NN.DerivativesLastLayer[9]*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1]))) *tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))
            differences[2]=differences[2]+Q2NN.DerivativesLastLayer[8]*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))- \
                alpha2*((-actionTilde[0]+mu1Tilde)*math.exp(-2*lsigma1Tilde)+2-4*math.exp(-2*actionTilde[0])/(1+math.exp(-2*actionTilde[0])))*gauss1*math.exp(lsigma1Tilde)*tanhDerivative((mu1Tilde+math.exp(lsigma1Tilde)*gauss1))
            differences[3]=differences[3]+Q2NN.DerivativesLastLayer[9]*gauss2*math.exp(lsigma2Tilde)*tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))- \
                alpha2*((-actionTilde[1]+mu2Tilde)*math.exp(-2*lsigma2Tilde)+2-4*math.exp(-2*actionTilde[1])/(1+math.exp(-2*actionTilde[1])))*gauss2*math.exp(lsigma2Tilde) *tanhDerivative((mu2Tilde+math.exp(lsigma2Tilde)*gauss2))                                                   
        '''
        
        piNN.retropropagation(state,differences,pas,storage)    
        #update target networks
   
        for i in range(0,sizeHiddenLayer1):
            for j in range(0,sizeInputQ):
                Q1TargetNN.HiddenLayer1EntryWeights[i][j]= \
                    rho*Q1TargetNN.HiddenLayer1EntryWeights[i][j]+(1-rho)*Q1NN.HiddenLayer1EntryWeights[i][j]
                Q2TargetNN.HiddenLayer1EntryWeights[i][j]= \
                    rho*Q2TargetNN.HiddenLayer1EntryWeights[i][j]+(1-rho)*Q2NN.HiddenLayer1EntryWeights[i][j]
                
        for i in range(0,sizeHiddenLayer2):
            for j in range(0,sizeHiddenLayer1):
                Q1TargetNN.HiddenLayer2EntryWeights[i][j]= \
                    rho*Q1TargetNN.HiddenLayer2EntryWeights[i][j]+(1-rho)*Q1NN.HiddenLayer2EntryWeights[i][j]
                Q2TargetNN.HiddenLayer2EntryWeights[i][j]= \
                    rho*Q2TargetNN.HiddenLayer2EntryWeights[i][j]+(1-rho)*Q2NN.HiddenLayer2EntryWeights[i][j]

        for i in range(0,sizeOutputQ):
            for j in range(0,sizeHiddenLayer2):
                Q1TargetNN.LastLayerEntryWeights[i][j]= \
                    rho*Q1TargetNN.LastLayerEntryWeights[i][j]+(1-rho)*Q1NN.LastLayerEntryWeights[i][j]
                Q2TargetNN.LastLayerEntryWeights[i][j]= \
                    rho*Q2TargetNN.LastLayerEntryWeights[i][j]+(1-rho)*Q2NN.LastLayerEntryWeights[i][j]

        state = next_state
        scoreEpisode+=reward

        nbSteps+=1

    if ep%100==1:
            print(str(sigma1)+" "+str(sigma2))
        

    if fin==True:
        break
        
    if ep<tailleFenetreGlissante:
        derniersScores[ep]=scoreEpisode
    else:
        derniersScores.pop(0)
        derniersScores.append(scoreEpisode)

    moyenne+=scoreEpisode
    print(str(ep)+': reward='+str(scoreEpisode)+' length='+str(length))


   


    
print("end of learning period")

env.close()