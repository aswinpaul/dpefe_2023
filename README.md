# On efficient computation in active inference

Despite its general regard as a neurobiologically plausible model for intelligent behaviour, 
active inference faces computational challenges in scaling up to commonly studied environments. 
Recent developments, including deep active inference, use environment-specific (trained) neural networks to 
approximate environment parameters like transition dynamics. 
This paper proposes a modified planning algorithm for finite temporal horizons that 
do not rely upon action trajectories. Observing that the cost function used for planning is Bellman-optimal, 
the planning scheme uses the well-known dynamic programming algorithm for reduced computational complexity. 
The algorithm recursively evaluates the expected free energy for actions backwards in time (not action sequences). 
The improved computational efficiency enables the algorithm's execution without approximations (like approximating model 
parameters and functionals using neural networks), which helps us precisely compute and study planning and model learning 
under uncertainty. The proposed method enables planning and demonstrates optimal behaviour even when the final goal's target 
distribution is strictly defined (i.e., uninformed). This improvement opens new opportunities for all kinds of applications. 
For example, in a navigation/search task, the preference makes sense only regarding the goal position, whereas, in soccer, 
it is the final score at 90 minutes. It is straightforward to specify a target distribution from a goal state instead of 
the much more complicated task of defining a temporally informed target distribution. We also study the possibility of 
learning the prior preference distribution inspired by the Z-learning algorithm dependent on a similar desirability function. 
We benchmark the proposed solutions through simulations in a standard grid world task.

<!-- 
<p align="center">
  <img src="learning_prior_preference.gif" width="50%" height="50%"/>
</p> -->
