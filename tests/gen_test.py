from numpy.random import default_rng
import sys
n = int(sys.argv[1])
rng = default_rng()
grid = rng.integers(-20, 21, (n, 2*n))
# print(grid)
with open(sys.argv[2], 'w') as file:
    for i in range(n):
        for j in range(2*n):
            file.write(str(grid[i][j]) + ",")
        file.write("\n")