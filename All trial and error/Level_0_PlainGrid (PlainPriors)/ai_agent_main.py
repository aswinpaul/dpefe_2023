# Active Inference agent
from math_functions import log_stable
from math_functions import softmax
from math_functions import onehot
from math_functions import kl_div
from math_functions import obj_array,obj_array_zeros
from true_parameters import trueA,trueB,det_dyn,det_A

import numpy as np
import math
from scipy.stats import dirichlet

# Environment
from wgwt_environment_first import Environment as EnvUp
# from wgwt_environment_second import Environment as EnvDown
from wgwt_environment_first import StartandGoal,StatetoCordinates,ctostates

# Agent functions
from ai_agent_planner import action_dist
from infer_state import infer_state

def ai_agent_main_trails(T,trials,sm_pars,time_steps,switch_time,a_known,grid_prior):
    
    EPS_VAL = 1e-16 #negligibleconstant

    global state
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

    #Number of states and number of controls/action
    num_states=[70]
    num_controls=[8]
    num_obs=[9,12]
    num_factors=len(num_states)
    num_control_factors=len(num_controls)
    num_modalities = len(num_obs) 

    numS=num_states[0]
    numA=num_controls[0]
    numO1=num_obs[0]
    numO2=num_obs[1]

    #Equipping agent with a deterministic prior/zero(flat)-prior about likelihood and dynamics

    a_prior=det_A()
    if(grid_prior==1):
        b_prior=det_dyn()
    else:
        b_prior=np.zeros((numS,numS,numA))

    #prior preferences in terms of observations
    C = obj_array_zeros(num_obs)
    C[0][goal_x]=1
    C[1][goal_y]=1

    #prior at t=0
    D = obj_array(num_factors)
    D[0]=np.zeros((num_states[0]))
    D[0][start_state]=1
    prior=D[0]

    #True Dynamics and liklihood for model learning evaluation
    Btrue_1=np.zeros((num_states[0], num_states[0], num_controls[0]))
    # Btrue_2=np.zeros((num_states[0], num_states[0], num_controls[0]))
    Btrue_1=trueB()
    A_true=obj_array(num_modalities)
    A_true=trueA()

    #MAIN
    smpt=len(sm_pars)
    episodes=np.zeros((smpt,trials,time_steps))
    modeldeviation_t_1=np.zeros((smpt,trials,time_steps))
    obsenoisedev_t_1=np.zeros((smpt,trials,time_steps))

    for sp in range(smpt):
        print('Gamma',sp)
        
        for ii in range(trials):
            print('Trial',ii)
            epi=0
            tau=0

            #Learning Transition dynamics
            #Dirichlet distribution (Priors)
            b=b_prior+EPS_VAL #Hidden-states
            a1=a_prior[0]+EPS_VAL #Mod-1
            a2=a_prior[1]+EPS_VAL #Mod-2

            Blearned=obj_array(num_factors)
            Alearned=obj_array(num_modalities)
            Blearned[0]=np.zeros((numS,numS,numA))
            Alearned[0]=np.zeros((numO1,numS))
            Alearned[1]=np.zeros((numO2,numS))

            for i in range(numS):
                Alearned[0][:,i]=dirichlet.mean(a1[:,i])
                Alearned[1][:,i]=dirichlet.mean(a2[:,i])
                for j in range(numA):
                    Blearned[0][:,i,j]=dirichlet.mean(b[:,i,j])
           
            
            #Planning using available A,B,C,T
            if(a_known==1):
                Qactions=action_dist(A_true,Blearned,C,T,sm_pars[sp])
            else:
                Qactions=action_dist(Alearned,Blearned,C,T,sm_pars[sp])
                
            for ts in range(time_steps):
                if(tau==0):
                    #New episode-start
                    prior=D[0]
                    global state
                    state=start_state
                    obs_idx=[start_x,start_y]

                kingsmoves=[0,1,2,3,4,5,6,7]
                #Perception
                if(a_known==1):
                    q_s=infer_state(prior,A_true,obs_idx)
                else:
                    q_s=infer_state(prior,Alearned,obs_idx)

                a1+=np.kron(q_s,onehot(obs_idx[0],numO1).reshape((-1,1)))
                a2+=np.kron(q_s,onehot(obs_idx[1],numO2).reshape((-1,1)))

                action_dist_qs=Qactions[tau,:,:].dot(q_s)
                action=np.random.choice(kingsmoves,p=action_dist_qs)
                [xp,yp,reward]=Env_agent(action,ts)
                obs_idx=[xp,yp]

                if(tau>1):
                    b[:,:,ac_mo]+=Qactions[tau,ac_mo,:].dot(q_s_mo)*np.kron(q_s_mo,q_s.reshape((-1,1)))

                tau+=1
                #End of trial conditions
                if(tau==T-1):
                    tau=0              
                    for i in range(numS):
                        Alearned[0][:,i]=dirichlet.mean(a1[:,i])
                        Alearned[1][:,i]=dirichlet.mean(a2[:,i])
                        for j in range(numA):
                            Blearned[0][:,i,j]=dirichlet.mean(b[:,i,j])
                    #Replanning with new_evidence
                    if(a_known==1):
                        Qactions=action_dist(A_true,Blearned,C,T,sm_pars[sp])
                    else:
                        Qactions=action_dist(Alearned,Blearned,C,T,sm_pars[sp])

                if(reward==1):
                    epi+=1
                    tau=0
                #Updating episodes completed
                episodes[sp,ii,ts]=epi

                Blearneddummy=np.zeros((numS,numS,numA))
                Alearneddummy=np.zeros((numO1,numA))
                Blearneddummy=np.array((Blearned[0]))
                Alearneddummy=np.array((Alearned[0]))
                for i in range(numS):
                    obsenoisedev_t_1[sp,ii,ts]+=kl_div(A_true[0][:,i],Alearneddummy[:,i])
                    for j in range(numA):
                        modeldeviation_t_1[sp,ii,ts]+=kl_div(Btrue_1[:,i,j],Blearneddummy[:,i,j])

                #Setting-up priors for next time_step
                prior=Blearned[0][:,:,action].dot(q_s)
                q_s_mo=q_s
                ac_mo=action
                
    return(episodes,modeldeviation_t_1,obsenoisedev_t_1)