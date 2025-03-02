import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

################################# DATA ####################################
data=pd.read_csv(r'Data.csv')
data=data.sample(frac=1).reset_index(drop=True)

train_input_data=data.drop(['O'],axis=1)
train_input=np.matrix(train_input_data.values)
train_output_data=data.drop(['A','B','C','D'],axis=1)
train_output=np.matrix(train_output_data.values)
###########################################################################

################################### NN ######################################
def sigmoid(x):
    
    return 1/(1+np.exp(-x))

def mse(answers):
    
    error=[]
    for i in range(len(train_output)):
        mse=(train_output[i]-answers[i]) ** 2
        error.append(mse)

    return np.array(error)
    
def feed_forward(entity):
    
    input_weights=np.reshape(np.matrix(entity[0:12]),(4,3))
    hl1_weights=np.reshape(np.matrix(entity[12:18]),(3,2))
    hl2_weights=np.reshape(np.matrix(entity[18:20]),(2,1))
    
    answers=[]
    for i in range(len(train_input)):
        
        hl1_h=np.dot(train_input[i],input_weights)
        hl1_hA=sigmoid(hl1_h)
        
        hl2_h=np.dot(hl1_hA,hl1_weights)
        hl2_hA=sigmoid(hl2_h)
        
        o=np.dot(hl2_hA,hl2_weights)
        oA=sigmoid(o)
        
        answers.append(oA)
        
    return answers
####################################################################################

############################################ GA ####################################
plt.rcParams['figure.figsize']=[8,6]

target=0
size=1000

population=[]

def populate(size):
    
    initial=[None]*size
    
    for i in range(len(initial)):
        
        entity=2*np.random.random((20,))-1
        
        initial[i] = entity
    
    return initial

def fitness(population,target, size):
    print('fitness')
    scores=[]
    pop=[]
    cord=[]
    for index,entity in enumerate(population):
        #print('feedfwd')
        answers=feed_forward(entity)
      #  print('done')
        error=mse(answers)
        
        score=np.mean(error)-target
        
        scores.append((score,entity,index))
        cord.append((score,score))
        #pop.append((entity,index))
    print('done')
    #print('scores:',len(scores),type(scores))#,np.array(scores).shape)
    #top_score=sorted(scores)[:size]
    top_score=np.array(scores)[np.array(scores)[:,0].argsort()]
    #print('top_score:',len(top_score),type(top_score),top_score[0][0])
    for i in range(size):
        order=top_score[i][1]
        pop.append(order)
    
    return np.array(pop)[:int(size*0.05)],np.array(cord),top_score[0],answers
        

def mate(population,size):
    #print('mate')
    pop_new=[]
    
    for _ in range(int(size/2)):
        
        p1=population[np.random.randint(0,len(population))]
        p2=population[np.random.randint(0,len(population))]
        
        # print(p1,'\n\n',p2)
        # input()
        
        for i in range(2):
            
            if i == 0:
                child = np.concatenate(([p1[:10],p2[10:]]))
                # child = copy.deepcopy(p1)
                # if np.array_equal(child, childx): print(True) 
                # else: print(False)
                # input()
                
            if i == 1:
                child = np.concatenate(([p2[:10],p1[10:]]))
                # child = copy.deepcopy(p2)
            
            mutation= 0.5*np.random.random((20,))-0.25
            
            child=child+mutation
            
            pop_new.append(child)
            
    return np.array(pop_new)

#population=populate(size)

def cycle(population,target,size,generations):
    #print('cycle')
    gen=0
    population=populate(size)
    #print('populate:',len(population))
    for _ in range(generations):
        gen+=1
        population,cord,top_score,answers=fitness(population,target,size)
        #print('fitness:',len(population))
        
        print('Generation:',gen)
        x=float(np.round(top_score[0],20))
        print('fittest: ',x)
        
        # print(population)
        population=mate(population,size)
        #print('mate:',len(population))
        
        if x <= 0:
            break
        
    print('Final weights',top_score[1])
    
        
    
    return population,cord,answers

def draw(cord, target):
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.scatter(cord[:,0],cord[:,1],c='green',s=12)
    plt.scatter(target[0], target[1], c ='red', s = 60)
    

generations=50

#population=mate(population,size)
#print('before cycle:',len(population))
population,cord,answers=cycle(population,target,size,generations)
#print('after cycle:',len(population))
draw(cord,(target,target))
for i in range(len(answers)):
    print(np.round(answers[i],2),'===',train_output[i])
#population,cord=fitness(population,target,size)
######################################################################################
