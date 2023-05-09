#Importing needed modules
import numpy as np

class grid_environment():
    def __init__(self, epi_length_limit = 10000):
        mdppath = './GridEnv/mdp_small.txt'
        
        #storing the file as strings line by line
        mdpdata=[]
        
        #Saving arm true means to the array-band (indices indicates arms)
        mdp = open(str(mdppath), "r")
        for x in mdp:
            mdpdata.append(x)
        mdp.close()
        #Closing mdp file.
        
        #Determining the nature of mdp
        mdptype1=[]
        for word in mdpdata[-2].split():
            try:
                mdptype1.append(str(word))
            except (ValueError, IndexError):
                pass
        mdptype=mdptype1[1]
        
        #Discount factor
        gamma1=[]
        for word in mdpdata[-1].split():
            try:
                gamma1.append(float(word))
            except (ValueError, IndexError):
                pass
        
        self.gamma=float(gamma1[0])
        
        #Number of states
        states=[]
        for word in mdpdata[0].split():
            try:
                states.append(int(word))
            except (ValueError, IndexError):
                pass
        self.numS=int(states[0])
        #Number of actions
        actions=[]
        for word in mdpdata[1].split():
            try:
                actions.append(int(word))
            except (ValueError, IndexError):
                pass
        self.numA=int(actions[0])
        #Start state
        start=[]
        for word in mdpdata[2].split():
            try:
                start.append(int(word))
            except (ValueError, IndexError):
                pass
            
        self.startS=int(start[0])
        #Terminal states for episodic mdps
        if(mdptype=='episodic'):
            terminal=[]
            for word in mdpdata[3].split():
                try:
                    terminal.append(int(word))
                except (ValueError, IndexError):
                    pass
            self.no_of_termS = len(terminal)
            self.terS = terminal
        
        #T-matrix dimensions numS*numS*numA
        #R-matrix dimensions numS*numA*numS
        
        self.T = np.zeros((self.numS,self.numA,self.numS))
        self.R = np.zeros((self.numS,self.numA,self.numS))
        if(mdptype=='episodic'):
            for i in range(len(self.terS)):
                self.T[self.terS[i],:,self.terS[i]]=1
        
        for i in range(4,len(mdpdata)-2):
            trans=[]
            for word in mdpdata[i].split():
                try:
                    trans.append(float(word))
                except (ValueError, IndexError):
                    pass
            trans
            s1=int(trans[0])
            ac=int(trans[1])
            s2=int(trans[2])
            r=float(trans[3])
            p=float(trans[4])
            self.T[s1,ac,s2]=p
            self.R[s1,ac,s2]=r
        
        #useful variables
        #numS,numA,startS,terS,mdptype,gamma,T,R
        
        self.current_state = self.startS
        self.end_state = self.terS
        self.info = None
        self.tau = 0
        self.tau_limit = epi_length_limit
        self.termination = False
        self.truncation = False
        
    def reset(self, seed = 10):
        self.current_state = self.startS
        self.termination = False
        self.truncation = False
        self.tau = 0
        return self.current_state, self.info
        
    def step(self, action):
        self.tau += 1
        n_s = np.argmax(self.T[self.current_state, action, :])
        reward = self.R[self.current_state, action, n_s]
        if(reward == 10):
            self.termination = True
        if(self.tau > self.tau_limit):
            self.truncation = True
        self.current_state = n_s
        return n_s, reward, self.termination, self.truncation, self.info
        
    def get_trueB(self):
        true_B = np.zeros((self.numS, self.numS, self.numA))
        for i in range(self.numS):
            for j in range(self.numA):
                true_B[:,i,j] = self.T[i,j,:]
        return true_B
