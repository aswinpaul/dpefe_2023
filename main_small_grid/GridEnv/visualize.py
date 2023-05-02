import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import numpy as np

def showPNG(grid):
	plt.figure(figsize=(10, 5))
	plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest')
	plt.xticks([]), plt.yticks([])
	#plt.show()
	plt.savefig("gridFile.png")

grid = np.loadtxt(sys.argv[1], dtype=int)

if sys.argv[1] != sys.argv[-1]:
	nr = len(grid)
	nc = len(grid[0])
	x,y = 0,0
	for i in range(nr):
		for j in range(nc):
			if grid[i][j] == 2:
				x,y = i,j
				grid[i][j] = 3
				break

	path = []
	with open(sys.argv[2]) as f:
		for line in f:
			path = deepcopy(line.split())

	# path = [int(z) for z in path]
	for i in path:
		if i == 'N':
			x -= 1
		elif i == 'E':
			y += 1
		elif i == 'W':
			y -= 1
		else:
			x += 1
		grid[x][y] = 2
	grid[x][y] = 4
	showPNG(grid)

else:
	nr = len(grid)
	nc = len(grid[0])
	x,y = 0,0
	for i in range(nr):
		for j in range(nc):
			if grid[i][j] == 2:
				grid[i][j] = 3
			elif grid[i][j] == 3:
				grid[i][j] = 4
	showPNG(grid)