# On efficient computation in active inference

Despite being recognized as neurobiologically plausible, active inference faces difficulties when employed to simulate intelligent behaviour in complex environments due to its computational cost and the difficulty of specifying an appropriate target distribution for the agent. This paper introduces two solutions that work in concert to address these limitations. First, we present a novel planning algorithm for finite temporal horizons with drastically lower computational complexity. Second, inspired by Z-learning from control theory literature, we simplify the process of setting an appropriate target distribution for new and existing active inference planning schemes. 
Our first approach leverages the dynamic programming algorithm, known for its computational efficiency, to minimize the cost function used in planning through the Bellman-optimality principle.
Accordingly, our algorithm recursively assesses the expected free energy of actions in the reverse temporal order. This improves computational efficiency by orders of magnitude and allows precise model learning and planning, even under uncertain conditions. Our method simplifies the planning process and shows meaningful behaviour even when specifying only the agent's final goal state. 
The proposed solutions make defining a target distribution from a goal state straightforward compared to the more complicated task of defining a temporally informed target distribution. The effectiveness of these methods is tested and demonstrated through simulations in standard grid-world tasks. These advances create new opportunities for various applications.

## Planning in DPEFE vs SI

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/dpefevssi.png" width="75%" height="75%" />
</p>

## Navigation problems

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/grid.png" width="75%" height="75%" />
</p>

## Learning of prior preference

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/learning_prior_preference.gif" width="50%" height="50%" />
</p>

## Sparse vs informed prior preference

<p align="center">
  <img src = "https://github.com/aswinpaul/dpefe_2023/blob/main/git_images/prior.png" width="75%" height="75%" />
</p>
