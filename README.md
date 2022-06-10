# An efficient non-approximate algorithm for active inference agents and applications to stochastic control
Despite its general regard as a biologically plausible model for intelligent behaviour, active inference
faced computational challenges in scaling up to commonly studied environments. Recent developments, including
deep active inference, use environment-specific (trained) neural networks to approximate environment parameters
like the transition dynamics. This paper proposes a modified planning algorithm for finite temporal horizons that
do not rely upon action trajectories. Observing that the cost function used for planning is Bellman optimal, the
planning scheme implements the well known dynamic programming algorithm and recursively evaluates the
expected free energy (a functional, evaluated by parameters that optimise surprise via variational free energy) for
actions in time-steps (not action sequences), backwards in time. The improved computational efficiency enables the
execution of the algorithm without approximations in larger scales, which helps us to precisely compute and study
planning and model-learning under uncertainty. The proposed method also enables planning and demonstrates
optimal behaviour even when the final goal’s target distribution is strictly defined (uninformed). This improvement
opens new opportunities for all kinds of applications. It is straightforward to specify a target distribution from a goal
state instead of the much more complicated task of defining a temporally informed target distribution. Additionally,
in our proposed architecture, we study the effect of agents learning the internal generative model parameters by
interacting with the surrounding environment. The results demonstrate the capability of the algorithm to tackle
environments where encoding these environmental functions is complicated. We build upon this work to assess
the utility of active inference for a stochastic control setting. We simulate the classic windy grid-world task with
additional complexities, namely: 1) environment transition stochasticity; 2) environment mutation, and 3) partial
observability. Our results demonstrate the advantage of using active inference to model intelligent behaviour and
control.
