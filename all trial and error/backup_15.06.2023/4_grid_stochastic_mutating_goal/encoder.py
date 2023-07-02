#Encoder
#Importing needed modules
import sys
import os
import numpy as np
import random
import math

# #Reading arguments for program from shell call
gridpath = sys.argv[1]

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
r = -1e-3
#Transitions
#all valid states
for i in range(numT):
    
    #Valid actions
    for j in range(4):
        a=nonterm[i][0]
        b=nonterm[i][1]
        s=(a,b)
        p=10
        n=-1e-3
        #North
        if(j==0):
            ap=a-1
            bp=b
            for k in range(numS):
                if(validstates[k][0]==ap and validstates[k][1]==bp):
                    sp=(ap,bp)
                    break
                else:
                    sp=(a,b)
            print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+'1')
        #South
        if(j==1):
            ap=a+1
            bp=b
            for k in range(numS):
                if(validstates[k][0]==ap and validstates[k][1]==bp):
                    sp=(ap,bp)
                    break
                else:
                    sp=(a,b)
            print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+'1')
        #East
        if(j==2):
            ap=a
            bp=b+1
            for k in range(numS):
                if(validstates[k][0]==ap and validstates[k][1]==bp):
                    sp=(ap,bp)
                    break
                else:
                    sp=(a,b)
            print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+'1')
                #North
        if(j==3):
            ap=a
            bp=b-1
            for k in range(numS):
                if(validstates[k][0]==ap and validstates[k][1]==bp):
                    sp=(ap,bp)
                    break
                else:
                    sp=(a,b)
            print('transition'+' '+str(ctostates(s[0],s[1]))+' '+str(j)+' '+str(ctostates(sp[0],sp[1]))+' '+str(r)+' '+'1') 

print('mdptype'+' '+'episodic')
print('discount'+' '+'1')
