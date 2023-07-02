#Encoder
#Importing needed modules
import sys
import os
import numpy as np
import random
import math

# #Reading arguments for program from shell call
gridpath = "grid10.txt"

#storing the file as strings line by line
griddata=[]

#Saving arm true means to the array-band (indices indicates arms)
grid = open(str(gridpath), "r")
for x in grid:
    griddata.append(x)
grid.close()
#Closing mdp file

#Grid Representation
grid=[]
for i in range(len(griddata)):    
    row=[]
    for word in griddata[i].split():
        try:
            row.append(int(word))
        except (ValueError, IndexError):
            pass
    grid.append(row)

n=len(grid)

allstates=[]
nonterm=[]
validstates=[]
endstates=[]
startstate=[]

for i in range(n):
    for j in range(n):
        
        allstates.append((i,j))
        if(grid[i][j]!=1):
            validstates.append((i,j))
        if(grid[i][j]!=1 and grid[i][j]!=3):
            nonterm.append((i,j))
        if(grid[i][j]==2):
            startstate.append((i,j))
        if(grid[i][j]==3):
            endstates.append((i,j))

#Useful lists
#allstates,validstates,endstates,startstate
numS=len(validstates)
numT=len(nonterm)
numSta=len(startstate)
numEnd=len(endstates)

#to get the state number corresponding to the coordinate of a valis state
def ctostates(x,y):
    s=0
    for i in range(numS):
        if(validstates[i][0]==x and validstates[i][1]==y):
            break
        s=s+1
    return(s)

print('numStates'+' '+str(numS))
print('numActions'+' '+'4')
print('Start'+' '+str(ctostates((startstate[0][0]),(startstate[0][1]))))
end=[]
for i in range(numEnd): 
    end.append(ctostates(endstates[i][0],endstates[i][1]))
print('end'+" "+ str(*end))

#Transitions
#all valid states
for i in range(numT):
    
    #Valid actions
    for j in range(4):
        
        a=nonterm[i][0]
        b=nonterm[i][1]
        s=(a,b)
        p=10
        n=-0.5

        system_certainty = 0.75
        stochasticity = [-1, 0, +1]
        for ss in stochasticity:
                
            if(ss == 0):
                tpr = system_certainty
            else:
                tpr= (1-system_certainty)/2
            
            #North
            if(j==0):
                ap=a-1
                if(ap<1):
                    ap = 1
                bp=b-ss

                if(bp<1):
                    bp = 1

                for k in range(numS):
                    if(validstates[k][0]==ap and validstates[k][1]==bp):
                        sp=(ap,bp)
                        break
                    else:
                        sp=(a,b)
                for kk in range(len(endstates)):
                    if(sp[0]==endstates[kk][0] and sp[1]==endstates[kk][1]):
                        r=p
                        break
                    else:
                        r=n

                print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+str(tpr))
            
            #South
            if(j==1):
                ap=a+1
                if(ap<1):
                    ap = 1
                bp=b-ss
                if(bp<1):
                    bp = 1
                for k in range(numS):
                    if(validstates[k][0]==ap and validstates[k][1]==bp):
                        sp=(ap,bp)
                        break
                    else:
                        sp=(a,b)
                for kk in range(len(endstates)):
                    if(sp[0]==endstates[kk][0] and sp[1]==endstates[kk][1]):
                        r=p
                        break
                    else:
                        r=n
                print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+str(tpr))
            
            #East
            if(j==2):
                ap=a-ss
                if(ap<1):
                    ap = 1
                bp=b+1
                if(bp<1):
                    bp = 1
                for k in range(numS):
                    if(validstates[k][0]==ap and validstates[k][1]==bp):
                        sp=(ap,bp)
                        break
                    else:
                        sp=(a,b)
                for kk in range(len(endstates)):
                    if(sp[0]==endstates[kk][0] and sp[1]==endstates[kk][1]):
                        r=p
                        break
                    else:
                        r=n
                print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+str(tpr))
            
            #West
            if(j==3):
                ap=a-ss
                if(ap<1):
                    ap = 1
                bp=b-1
                if(bp<1):
                    bp = 1
                for k in range(numS):
                    if(validstates[k][0]==ap and validstates[k][1]==bp):
                        sp=(ap,bp)
                        break
                    else:
                        sp=(a,b)
                for kk in range(len(endstates)):
                    if(sp[0]==endstates[kk][0] and sp[1]==endstates[kk][1]):
                        r=p
                        break
                    else:
                        r=n
                print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+str(tpr)) 

print('mdptype'+' '+'episodic')
print('discount'+' '+'1')
