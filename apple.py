import numpy as np
import random,copy,time
from datetime import datetime as dt

#set nn structure [input nodes,x hiddenlayers with x nodes....,output nodes, o for offset]
#function to generate random chromosomes
def chromosome_generetor(structure):
    x=[]
    for i in range(0,len(structure)-1):
        k=structure[i]*structure[i+1]
        x.append(k)
    x=sum(x)
    return 2*np.random.random((x,))-1

#function to arrange the weights for each layers 
def weights_classifier(structure,chromosome):
    weights_list=[]
    for i in range(0,len(structure)-2):
        if i == 0:
            a=0
            b=structure[i]*structure[i+1]
        else:
            a=b
            b=a+(structure[i]*structure[i+1])
        r=structure[i]
        c=structure[i+1]
        w=np.reshape(np.matrix(chromosome[a:b]),(r,c))
        weights_list.append(w)
    return weights_list

def sigmoid(x):
    return 1/(1+np.exp(-x))

def feed_forward(structure,chromosome,train_input):
    weights_list=weights_classifier(structure,chromosome)
    for i in range(len(weights_list)):
        if i==0:
            output=sigmoid(np.dot(train_input,weights_list[i]))
        else:
            output=sigmoid(np.dot(output,weights_list[i]))     
    return output
 
def playing(structure,chromosome,train_input):
    decision=feed_forward(structure,chromosome,train_input)
    return np.array(decision)

def visionApple(snake,apple):
    vision_apple=[0, 0, 0, 0, 0, 0, 0, 0]
    n=snake[0][1]
    s=snake[0][1]
    w=snake[0][0]
    e=snake[0][0]

    for i in range(10):

        n-=60
        s+=60
        w-=60
        e+=60

        if apple[1]==n and apple[0]==snake[0][0]:
            vision_apple[0]=1

        if apple[1]==s and apple[0]==snake[0][0]:
            vision_apple[1]=1  

        if apple[0]==w and apple[1]==snake[0][1]:
            vision_apple[2]=1      

        if apple[0]==e and apple[1]==snake[0][1]:
            vision_apple[3]=1        

        if apple[1]==n and apple[0]==w:
            vision_apple[4]=1    

        if apple[1]==s and apple[0]+60==w:
            vision_apple[5]=1    

        if apple[1]+60==n and apple[0]==e:
            vision_apple[6]=1 

        if apple[1]==s and apple[0]==e:
            vision_apple[7]=1
            
    return vision_apple

def visionBody(snake,apple):
    vision_body=[0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(1,len(snake)):
        
        n=snake[0][1]
        s=snake[0][1]
        w=snake[0][0]
        e=snake[0][0]
        
        for _ in range(10):

            n-=60
            s+=60
            w-=60
            e+=60

            if snake[i][1]==n and snake[i][0]==snake[0][0]:
                vision_body[0]=1

            if snake[i][1]==s and snake[i][0]==snake[0][0]:
                vision_body[1]=1  

            if snake[i][0]==w and snake[i][1]==snake[0][1]:
                vision_body[2]=1      

            if snake[i][0]==e and snake[i][1]==snake[0][1]:
                vision_body[3]=1        

            if snake[i][1]==n and snake[i][0]==w:
                vision_body[4]=1    

            if snake[i][1]==s and snake[i][0]+60==w:
                vision_body[5]=1    

            if snake[i][1]+60==n and snake[i][0]==e:
                vision_body[6]=1 

            if snake[i][1]==s and snake[i][0]==e:
                vision_body[7]=1
            
    return vision_body

def visionWall(snake):
    #            nw,sw,ww,ew
    vision_wall=[0, 0, 0, 0]
    
    vision_wall[0]=int((snake[0][1]/60))
    vision_wall[1]=int(((600-snake[0][1])/60)-1)
    vision_wall[2]=int((snake[0][0]/60))
    vision_wall[3]=int(((600-snake[0][0])/60)-1)
    
    return vision_wall

def move_d(decision):
    
    move=[0,0,0,0]
    j=np.amax(decision)
    k=np.where(decision==j)
    i=k[1][0]
    #print('\nj= ',j,' k= ',k,' i= ',i)
    move[i]=1
    
    return move

def game(structure,chromosome):
    
    frame_w=600
    frame_h=600
    
    #pg.init()
    #screen = pg.display.set_mode((frame_w,frame_h))

    white=[255,255,255]
    red=[255,0,0]
    black=[0,0,0]
    green=[0,255,0]
    
    running=True
    apple_present=False

    score=0
    steps=0
    apple_steps=200

    x=[]
    y=[]

    for i in range(0,frame_w,60):
        x.append(i)
        y.append(i)

    grid=[]
    for i in range(len(x)):
        for j in range(len(y)):
            grid.append((x[i],y[j]))
			
    
    snake_head=grid[55]
    snake=[snake_head]
	
    
    while running:
        
        #pg.event.get()
        
        grid_c=copy.deepcopy(grid) 

        if apple_present==False:
            time.sleep(0.001)
            for body in snake:
                if body in grid_c:
                    grid_c.remove(body)
            apple=random.choice(grid_c)
            apple_present=True

        vision_apple=visionApple(snake,apple)

        #print('\nApple= ',vision_apple)

        #print('snake= ',snake,' apple= ',apple,' Score= ',score,' steps= ',steps)

        if steps==0:
            vision_body=[0, 0, 0, 0, 0, 0, 0, 0]
            vision_wall=[0, 0, 0, 0]
            head_direction=[0, 0, 0, 0]
            tail_direction=[0, 0, 0, 0]

        nn_input=np.concatenate([vision_apple,vision_body,vision_wall,head_direction,tail_direction])

        ###call nn
        decision=playing(structure,chromosome,nn_input)
        #### call move
        move=move_d(decision)
        #print(decision)
        

        snake_c=copy.deepcopy(snake)

        if move ==[1,0,0,0]:
            snake[0]=(snake[0][0],snake[0][1]-60)
        if move ==[0,1,0,0]:
            snake[0]=(snake[0][0],snake[0][1]+60)
        if move ==[0,0,1,0]:
            snake[0]=(snake[0][0]-60,snake[0][1])
        if move ==[0,0,0,1]:
            snake[0]=(snake[0][0]+60,snake[0][1])
            
            
        for i in range(1,len(snake)):

            if snake[i][0]==snake[0][0] and snake[i][1]==snake[0][1]:
                #print('snake= ',snake,' apple= ',apple,' Score= ',score,' steps= ',steps)
                #print('Body Collision')
                running=False
                

        for i in range(1,len(snake)):
            snake[i]=snake_c[i-1]

        if snake[0]==apple:
            apple_present=False
            snake.append(snake_c[-1])
            #snake_c=copy.deepcopy(snake)
            score+=1
            apple_steps=200
            #print('Apple Gone')

        if len(snake)==1:
            vision_body=[0, 0, 0, 0, 0, 0, 0, 0]
        elif len(snake)>1:
            vision_body=visionBody(snake,apple)
        #print('Body= ',vision_body)

        if snake[0][0] < 0 or snake[0][0] > 540 or snake[0][1] < 0 or snake[0][1] > 540 :
            #print('Wall Collision')
            running=False


        if len(snake) != 1:        
            if (snake_c[-1][1]-snake[-1][1])<0:
                tail_direction=[0,1,0,0]
            if (snake_c[-1][1]-snake[-1][1])>0:
                tail_direction=[1,0,0,0]
            if (snake_c[-1][0]-snake[-1][0])<0:
                tail_direction=[0,0,0,1]
            if (snake_c[-1][0]-snake[-1][0])>0:
                tail_direction=[0,0,1,0]
        else:
            tail_direction=move

        head_direction=move

        vision_wall=visionWall(snake)
        
        steps+=1
        apple_steps -= 1
        if apple_steps == 0:
            running=False
        if score==99:
            running=False

    return score,steps

population=[]

def populate(size):
    
    initial_pop=[None]*size
    
    for i in range(size):
        chromosome=chromosome_generetor(structure)
        initial_pop[i]=chromosome
    
    return initial_pop

def fitness(population,size,structure,gen):
    
    if gen > 500:
        top_percentail=0.1
    else:
        top_percentail=0.05
    score=[None]*size
    apples=[None]*size
    for index,chromosome in enumerate(population):
        
        apple,steps=game(structure,chromosome)
        apples[index]=(apple)
        point=(steps+((2**apple)+500*(apple**2.1))-(0.25*(steps**1.3)*(apple**1.2)))*-1
        score[index]=point,chromosome,apple,steps,index
        
    top_score=np.array(score)[np.array(score)[:,0].argsort()]
    #print(top_score)
    order=[None]*size
    for i in range(size):
        order[i]=top_score[i][1]
    
    top_snakes=np.array(order)[:int(size*0.1)]
    warrior=top_score[0]
    weakest=top_score[99]
    #best_apple=np.amax(apples)
    #print(top_score[0][2],top_score[0][3],top_score[99][2],top_score[99][3])
    return top_snakes,warrior,weakest

def mutation(child,m_rate):
    
    loop=int(round((len(child)/100)*m_rate))
    #x=len(child)
    #child=np.reshape(child,(1,x))
    x=child.tolist()
    #print(len(x))
    for _ in range(loop):
        i=random.randrange(0,len(x),1)
        #print(i)
        child[i]=child[i]+(2*np.random.random()-1)
    
    #child=np.reshape(child,(x,1))
    return child

def mate(population,size,m_rate,pattern):
    new_pop=[]
    for _ in range(int(size/2)):
        
        #print(len(population))
        p1=population[np.random.randint(0,len(population))]
        p2=population[np.random.randint(0,len(population))]
        

        if pattern=='single_point':
            x=int(round(((len(p1)/100)*50)+0.1))
            for i in range(2):
            
                if i == 0:
                    child = np.concatenate(([p1[:x],p2[x:]]))

                if i == 1:
                    child = np.concatenate(([p2[:x],p1[x:]]))
                
                child=mutation(np.array(child),m_rate)
                new_pop.append(child)
        
        if pattern=='multi_point':
            child=[None]*len(p1)
            for i in range(0,len(child),2):
                if i == len(child)-1:
                    child[i]=p1[i]
                else:
                    child[i]=p1[i]
                    child[i+1]=p2[i+1]       
            child=mutation(np.array(child),m_rate)
            new_pop.append(child)

            child=[None]*len(p1)
            for i in range(0,len(child),2):
                if i == len(child)-1:
                    child[i]=p2[i]
                else:
                    child[i]=p2[i]
                    child[i+1]=p1[i+1]
            child=mutation(np.array(child),m_rate)
            new_pop.append(child)
            
    return new_pop

def cycle(generation,population,size,m_rate,structure,pattern,gen):
 
    if gen==0:
        population=populate(size)

    for _ in range(generation):
        start_time=dt.now()
        gen+=1
        population,warrior,weakest=fitness(population,size,structure,gen)
        population=mate(population,size,m_rate,pattern)
        path='weights_5/gen_'+str(gen)+'.txt'
        file=open(path,'w')
        file.write(str(warrior[1].tolist()))
        file.close()

        f=open('population_5/gen_'+str(gen)+'.txt','w')
        f.write(str(population))
        f.close()
        print('Generation: ',str(gen),'-----','Fittest: ',warrior[0],' Score= ',warrior[2],'->',weakest[2],' Steps= ',warrior[3],'->',weakest[3])
        print('Time Taken: ',dt.now()-start_time)
    return warrior

structure=[28,14,7,4,0]
size=500
generations=5000
m_rate=100
pattern='single_point'

interupted=input('Interupted?[y/n]: ')

if interupted=='y':
	gen=int(input('Last Gen: '))
	
	file=open('population_5/gen_'+str(gen)+'.txt','r')
	gen_x=file.read()
	file.close()

	population=[None]*size
	for j in range(0,len(population)):
		refined_w=gen_x.split(')')[j].split('(')[1].split(',')
		chromosome=[None]*len(refined_w)
		for i in range(0,len(refined_w)):
			if i == 0:
				chromosome[i]=float(gen_x.split(')')[j].split('(')[1].split(',')[i].split('[')[1])
			elif i == len(refined_w)-1:
				chromosome[i]=float(gen_x.split(')')[j].split('(')[1].split(',')[i].split(']')[0])
			else:
				chromosome[i]=float(gen_x.split(')')[j].split('(')[1].split(',')[i])
		population[j]=np.array(chromosome)
    
else:
	new_pop=[]
	gen=0
print('All files are loaded and learning has started')
warrior=cycle(generations,population,size,m_rate,structure,pattern,gen)
