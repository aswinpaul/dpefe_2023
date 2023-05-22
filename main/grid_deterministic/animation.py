import imageio

# Figure figure when rendered in environment
# plt.savefig(f'./img/img_{self.tau}.png')

frames = []
for t in range(100):
    image = imageio.v2.imread(f'./img/img_{t}.png')
    frames.append(image)

imageio.mimsave('./prior_preference.gif', # output gif
            frames,          # array of input frames
            fps = 3)         # optional: frames per second
