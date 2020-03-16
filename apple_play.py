import numpy as np
import random,copy,time
import pygame as pg

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

    y=snake[0][1]
    x=snake[0][0]

    for i in range(1,len(snake)):
        #n
        if snake[i][1]+60==y and snake[i][0]==x:
            vision_body[0]=1
        #s
        if snake[i][1]-60==y and snake[i][0]==x:
            vision_body[1]=1  
        #w
        if snake[i][0]+60==x and snake[i][1]==y:
            vision_body[2]=1      
        #e
        if snake[i][0]-60==x and snake[i][1]==y:
            vision_body[3]=1        
        #nw
        if snake[i][1]+60==y and snake[i][0]+60==x:
            vision_body[4]=1
        #sw
        if snake[i][1]-60==y and snake[i][0]+60==x:
            vision_body[5]=1    
        #ne
        if snake[i][1]+60==y and snake[i][0]-60==x:
            vision_body[6]=1 
        #se
        if snake[i][1]-60==y and snake[i][0]-60==x:
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

def game(structure,chromosome,speed):
    
    weights_list=weights_classifier(structure,chromosome)
    
    snake_frame_w=600
    snake_frame_h=600
    
    nn_frame_w=800
    nn_frame_h=1000
    
    pg.init()
    screen = pg.display.set_mode((snake_frame_w+nn_frame_w,nn_frame_h))

    white=[255,255,255]
    red=[255,0,0]
    black=[0,0,0]
    green=[0,255,0]
    desert=[115,92,64]
    light_desert=[232,186,129]
    grey=[64,64,64]
    
    running=True
    apple_present=False

    score=0
    steps=0
    apple_steps=500

    x=[]
    y=[]

    for i in range(0,snake_frame_w,60):
        x.append(i)
        y.append(i)

    grid=[]
    for i in range(len(x)):
        for j in range(len(y)):
            grid.append((x[i],y[j]))

    snake_head=grid[55]
    snake=[snake_head]
    
    while running:
        
        pg.event.get()
        
        grid_c=copy.deepcopy(grid)

        if apple_present==False:
            for body in snake:
                if body in grid:
                    grid_c.remove(body)
            apple=random.choice(grid_c)
            apple_present=True

        vision_apple=visionApple(snake,apple)
        if np.amax(vision_apple) > 0:
            line=vision_apple.index(int(np.amax(vision_apple)))
            colors=[red,red,red,red,red,red,red,red]
            colors[line]=green
        else:
            colors=[red,red,red,red,red,red,red,red]
            
        if steps==0:
            vision_body=[0, 0, 0, 0, 0, 0, 0, 0]
            vision_apple=[0, 0, 0, 0, 0, 0, 0, 0]
            vision_wall=[0, 0, 0, 0]
            head_direction=[0, 0, 0, 0]
            tail_direction=[0, 0, 0, 0]
            
   #     print('\nApple= ',vision_apple)
  #      print('Body= ',vision_body)
 #       print('head_direction= ',head_direction,' tail_direction= ',tail_direction, ' wall_dis= ',vision_wall)
#        print(' Score= ',score,' steps= ',steps)
        
        nn_input=np.concatenate([vision_apple,vision_body,vision_wall,head_direction,tail_direction])

        ###call nn
        decision=playing(structure,chromosome,nn_input)
        #print(decision)
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
                print('Body Collision')
                running=False
                

        for i in range(1,len(snake)):
            snake[i]=snake_c[i-1]

        if snake[0]==apple:
            apple_present=False
            snake.append(snake_c[-1])
            #snake_c=copy.deepcopy(snake)
            score+=1
            apple_steps=500
            #print('Apple Gone')
        
        
        
        if len(snake)==1:
            vision_body=[0, 0, 0, 0, 0, 0, 0, 0]
        elif len(snake)>1:
            vision_body=visionBody(snake,apple)
        

        if snake[0][0] < 0 or snake[0][0] > 540 or snake[0][1] < 0 or snake[0][1] > 540 :
            print('Wall Collision')
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
            print('Too much steps')
            
        screen.fill(white)
        
        wall=pg.Rect(0,0,snake_frame_w,snake_frame_h)
        background=pg.Rect(0,0,snake_frame_w,snake_frame_h)
        pg.draw.rect(screen,desert,wall)
        pg.draw.rect(screen,black,wall,2)
        
        stat_border=pg.Rect(0,snake_frame_h,snake_frame_w,snake_frame_h)
        stat_background=pg.Rect(0,0,snake_frame_w,nn_frame_h)
        pg.draw.rect(screen,light_desert,stat_background)
        pg.draw.rect(screen,black,stat_border,2)
        
        nn_background=pg.Rect(snake_frame_w,0,nn_frame_w,nn_frame_h)
        nn_board=pg.Rect(snake_frame_w,0,nn_frame_w,nn_frame_h)
        pg.draw.rect(screen,grey,nn_background)
        pg.draw.rect(screen,black,nn_board,1)

        #for i in range(len(grid_c)):
            #g_grid=pg.Rect(grid_c[i][0],grid_c[i][1],60,60)
            #pg.draw.rect(screen,black,g_grid,1)
            
       # ln_l=pg.draw.line(screen,colors[0],(snake[0][0],snake[0][1]),(snake[0][0],0),2)
      #  s_l=pg.draw.line(screen,colors[1],(snake[0][0],snake[0][1]),(snake[0][0],600),2)
     #   w_l=pg.draw.line(screen,colors[2],(snake[0][0],snake[0][1]),(0,snake[0][1]),2)
    #    e_l=pg.draw.line(screen,colors[3],(snake[0][0],snake[0][1]),(600,snake[0][1]),2)
   #     nw_l=pg.draw.line(screen,colors[4],(snake[0][0],snake[0][1]),(snake[0][0]-600,snake[0][1]-600),2)
  #      sw_l=pg.draw.line(screen,colors[5],(snake[0][0],snake[0][1]),(snake[0][0]-600,snake[0][1]+600),2)
 #       ne_l=pg.draw.line(screen,colors[6],(snake[0][0],snake[0][1]),(snake[0][0]+600,snake[0][1]-600),2)
#        ne_l=pg.draw.line(screen,colors[7],(snake[0][0],snake[0][1]),(snake[0][0]+600,snake[0][1]+600),2)
        
        g_apple=pg.Rect(apple[0]+15,apple[1]+15,30,30)
        pg.draw.rect(screen,green,g_apple)

        for i in range(len(snake)):
            if i==0:
                color=red
            else:
                color=black
            g_snake=pg.Rect(snake[i][0],snake[i][1],60,60)
            b_snake=pg.Rect(snake[i][0],snake[i][1],60,60)
            pg.draw.rect(screen,color,g_snake)
            pg.draw.rect(screen,white,b_snake,1)
            
        outputs=[0]
        for i in range(len(weights_list)):
            if i==0:
                output=sigmoid(np.dot(nn_input,weights_list[i]))
            else:
                output=sigmoid(np.dot(output,weights_list[i])) 
            outputs.append(output)
        outputs

        node_cords=[]
        for i in range(len(structure)-1):
            if i !=0:
                strongest_nodes=np.where(outputs[i]==np.amax(outputs[i]))[1].tolist()
            if i == 0:
                y_cor=snake_frame_w+10+15
            else:
                y_cor=(y_cor+int((nn_frame_h/30)/(len(structure)-1))*30)

            start_x=int(((nn_frame_h/30)-structure[i])/2) * 30

            nodes_center_cor=[]
            for x in range(structure[i]):
                if x==0:
                    x_cor=(start_x-30)+45
                else:
                    x_cor=x_cor+30

                nodes_center_cor.append((x_cor,y_cor))       

                if i == 0:
                    if nn_input[x] != 0:
                        pg.draw.circle(screen, green, (y_cor,x_cor) , 13)
                    else:
                        pg.draw.circle(screen, white, (y_cor,x_cor) , 10)

                elif i != 0:
                    if x in strongest_nodes:
                        #print('---------------->',x,strongest_line)
                        pg.draw.circle(screen, green, (y_cor,x_cor) , 13)
                    else:
                        pg.draw.circle(screen, white, (y_cor,x_cor) , 10)

                else:
                    #print('-->',x)
                    pg.draw.circle(screen, white, (y_cor,x_cor) , 10)

            node_cords.append(nodes_center_cor)
            #node_cords.append(0)

        for i in range(len(node_cords)):
            if i < len(node_cords)-1:
                for from_cor in node_cords[i]:
                    #print(from_cor[0],from_cor[1]+10)
                    for to_cor in node_cords[i+1]:
                        #print(from_cor[0],from_cor[1]+10,to_cor[0],to_cor[1]-10)
                        r=random.randint(0,255)
                        g=random.randint(0,255)
                        b=random.randint(0,255)
                        l_color=[r,g,b]
                        line=pg.draw.line(screen,l_color,(from_cor[1]+13,from_cor[0]),(to_cor[1]-13,to_cor[0]),1)

        pg.display.flip()
        
        time.sleep(speed)

    return score,steps

structure=[28,14,7,4,0]
start=int(input('From= '))
end=int(input('To= '))
speed=float(input('Frame Speed (s)= '))
for j in range(start,end):
    while True:
        try:
                file=open('weights_5/gen_'+str(j)+'.txt','r')
                break
        except:
                file=open('weights_5/gen_'+str(j-1)+'.txt','r')
                weights=file.read()
                file.close()
                refined_w=weights.split(',')
                chromosome=[None]*len(refined_w)

                for i in range(len(refined_w)):
                        if i==0:
                                chromosome[i]=(float(refined_w[i].split('[')[1]))
                        elif i==len(refined_w)-1:
                                chromosome[i]=(float(refined_w[i].split(']')[0]))
                        else:
                                chromosome[i]=(float(refined_w[i]))
                score,steps=game(structure,chromosome,speed)
                print('Gen= ',j,'-----Score= ',score)

    weights=file.read()
    file.close()

    refined_w=weights.split(',')
    chromosome=[None]*len(refined_w)
    
    for i in range(len(refined_w)):
        if i==0:
            chromosome[i]=(float(refined_w[i].split('[')[1]))
        elif i==len(refined_w)-1:
            chromosome[i]=(float(refined_w[i].split(']')[0]))
        else:
            chromosome[i]=(float(refined_w[i]))
            
    score,steps=game(structure,chromosome,speed)  
    print('Gen= ',j,'-----Score= ',score)
