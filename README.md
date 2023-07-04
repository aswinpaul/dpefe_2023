# On efficient computation in active inference

Despite being recognized as neurobiologically plausible, active inference faces difficulties when employed to simulate intelligent behaviour in complex environments due to its computational cost and the difficulty of specifying an appropriate target distribution for the agent. This paper introduces two solutions that work in concert to address these limitations. First, we present a novel planning algorithm for finite temporal horizons with drastically lower computational complexity. Second, inspired by Z-learning from control theory literature, we simplify the process of setting an appropriate target distribution for new and existing active inference planning schemes. 
Our first approach leverages the dynamic programming algorithm, known for its computational efficiency, to minimize the cost function used in planning through the Bellman-optimality principle.
Accordingly, our algorithm recursively assesses the expected free energy of actions in the reverse temporal order. This improves computational efficiency by orders of magnitude and allows precise model learning and planning, even under uncertain conditions. Our method simplifies the planning process and shows meaningful behaviour even when specifying only the agent's final goal state. 
The proposed solutions make defining a target distribution from a goal state straightforward compared to the more complicated task of defining a temporally informed target distribution. The effectiveness of these methods is tested and demonstrated through simulations in standard grid-world tasks. These advances create new opportunities for various applications.

Paper link: https://arxiv.org/abs/2307.00504

## Planning in DPEFE vs SI

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/dpefevssi.png" width="75%" height="75%" />
</p>

Graphics to compare and contrast the differences between the sophisticated inference and DPEFE (Dynamic
programming in expected free energy) algorithm planning schemes. A: Sophisticated inference algorithm uses an
extensive tree search, going forward in time, to accumulate free energy of the future paths. So, an agent’s preference
for observations, when matched with future predictions, will inform an optimal state-action trajectory, as shown in
the tree search. Light-purple states represent the preferred observations at that given time step, and light-blue actions
are the optimal actions inferred through the tree search. As noted in Friston et al. [2021], an agent can significantly
reduce the tree search complexity by terminating the search when the action probability falls below a certain threshold.
However, this approximation does not guarantee optimal policy as the agent might miss preferred observations deeper
in the tree search. B: In the DPEFE algorithm, an agent starts planning backwards from a fixed planning horizon. Here,
the EFE of future states informs EFE of state-action pairs one step backward in time. Hence, the planning complexity
of tree search is avoided, but the preference for future states propagates to influence decisions at previous time steps.
Since the agent needs to evaluate only a table (of EFE) at every planning step, this planning algorithm is linear in time,
number of states, and number of actions.

## Navigation problems

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/grid.png" width="75%" height="75%" />
</p>

A: A standard grid world of 100 states with 50 valid states. B: A grid of 400 states with 204 valid states.
C: A grid of 900 states with 497 valid states. These three grids are used for evaluating the performance of various
schemes.

## Learning of prior preference

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/learning_prior_preference.gif" width="50%" height="50%" />
</p>

## Sparse vs informed prior preference

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/prior.png" width="75%" height="75%" />
</p>

A: The sparsely defined preference distribution used by the DPEFE agent in simulations, B: The learned prefer-
ence distribution by AIF (T=1) agent over 50 episodes. Lighter colours imply a higher preference for the corresponding
states.
