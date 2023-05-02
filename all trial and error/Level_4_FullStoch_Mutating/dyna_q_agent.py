# Math functions
import numpy as np
import random as random

# Environment
from wgwt_environment_first import Environment as EnvUp
from wgwt_environment_second import Environment as EnvDown
from wgwt_environment_first import StartandGoal,StatetoCordinates,ctostates

seedloops=10 #trials
seedloops=10
p_trials=3
def dyna_q_agent_trail(time_steps=10000,switch_time=11000):
    
    def Env_agent(action,ts):
        global state
        if(ts<switch_time):
            Environment=EnvUp
        else:
            Environment=EnvDown
        [xp,yp,reward,new_state]=Environment(state,action)
        state=new_state
        return(xp,yp,reward)

    start_state,goal_state=StartandGoal()
    [start_x,start_y]=StatetoCordinates(start_state)
    [goal_x,goal_y]=StatetoCordinates(goal_state)

    start_x,start_y,goal_x,goal_y
    kings_moves=[0,1,2,3,4,5,6,7]
    start_state=ctostates(start_x,start_y)
    epsilon=0.1; #exploration rate
    alpha=0.5; #alpha in update-rule
    gamma=1; #gamma in target discount rate

    numX=9
    numY=12
    numA=8

    #Initialsing the model M(x,y,a,x',y',reward)
    model = np.nan*np.zeros((numX,numY,numA,3))

    episodes=np.zeros((p_trials,seedloops,time_steps))

    for pt in range(3):
        if(pt==0):
            replay_par=0
        if(pt==1):
            replay_par=5
        if(pt==2):
            replay_par=10

        for sl in range(seedloops):

            #Changing random seeds
            rseed=sl;
            random.seed(rseed)
            np.random.seed(rseed)
            Q=np.zeros((numX,numY,numA));

            global state
            state=start_state
            x=start_x
            y=start_y
            action=random.randint(0,numA-1);

            epi=0

            for ts in range(time_steps):

                #Exploration-rate for e-greedy
                exploration_rate=random.uniform(0, 1)

                #Fetching next-state reward from envrionment-function
                [xp,yp,reward]=Env_agent(action,ts)

                #Random-action
                if(exploration_rate > epsilon):
                    new_action=np.argmax(Q[xp-1,yp-1,:])
                #greedy-action
                else:
                    new_action=random.randint(0,numA-1) #random-action

                #print(state,action)
                #Updating Q (Q-learning off policy)
                Target= reward + gamma*np.max(Q[xp-1,yp-1,:])
                Q[x-1,y-1,action]=Q[x-1,y-1,action] + alpha*(Target-Q[x-1,y-1,action]) # update Q value

                #model_update
                model[x,y,action]=[xp,yp,reward]

                #starting from the new-state
                action=new_action
                x=xp
                y=yp

                #Restarting episode when goal is reached
                if(reward==1):
                    #incrementing episode number
                    state=start_state
                    x=start_x
                    y=start_y
                    action=random.randint(0,numA-1)
                    epi+=1

                episodes[pt][sl][ts]=epi

                for j in range(replay_par):
                    candidates = np.array(np.where(~np.isnan(model[:,:,:,0]))).T
                    idx = np.random.choice(len(candidates))
                    #Obtain the randomly selected state and action values from the candidates
                    xi,yi,actioni = candidates[idx]
                    #Obtain the expected reward and next state from the model
                    xpi,ypi,rewardi = model[xi,yi,actioni]        
                    #Updating Q (Q-learning off policy)
                    Target= rewardi + gamma*np.max(Q[int(xpi)-1,int(ypi)-1,:])
                    Q[xi-1,yi-1,actioni]=Q[xi-1,yi-1,actioni] + alpha*(Target-Q[xi-1,yi-1,actioni]) # update Q value
                    
    return(episodes)