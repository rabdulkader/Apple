import numpy as np
import subprocess,os
import random,copy,time

while True:
    try:
        fread_1=open('result_1.txt','r')
        break
    except:
        time.sleep(0.0001)
new_pop1=fread_1.read()
time.sleep(2)
fread_1.close()

result_1=[None]*125
for j in range(0,125):
    print(new_pop1)
    if j==0:
        refined_point=new_pop1.split('---')[j].split(',')[0].split('(')[1]
    else: 
        refined_point=new_pop1.split('---')[j].split(',')[1].split('(')[1]
        
    refined_apple=new_pop1.split('---')[j].split(')')[-1].split(',')[1]
    refined_weight=new_pop1.split('---')[j].split('(')[2].split(')')[0].split('[')[1].split(']')[0].split(',')

    chromosome=[None]*len(refined_weight)
    for i in range(len(chromosome)):
        chromosome[i]=float(refined_weight[i])
    result_1[j]=refined_point,np.array(chromosome),refined_apple
print('Done')