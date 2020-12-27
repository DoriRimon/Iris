import numpy as np
import argparse
import pandas
import mykmeanssp as mk

# TODO - work with numpy arrays

parser = argparse.ArgumentParser()
parser.add_argument("K", type=int)
parser.add_argument("N", type=int)
parser.add_argument("d", type=int)
parser.add_argument("MAX_ITER", type=int)
parser.add_argument("filename", type=str)

args = parser.parse_args()
K, N, d, MAX_ITER, filename = args.K, args.N, args.d, args.MAX_ITER, args.filename

assert(K > 0 and N > 0 and d > 0 and MAX_ITER > 0 and K < N)

observations = []
clusters = []
centroids = []

df = pandas.read_csv(filename, header=None)
observations = df.to_numpy(dtype=np.float_)


def norm_func(x, u):
	return np.power(np.linalg.norm(x - u, axis=0), 2)


def k_means_pp():
	np.random.seed(0)
	nums = [i for i in range(N)]
	rand = np.random.choice(nums, 1)
	centroids.append(int(rand[0]))
	min_arr = [norm_func(x, observations[centroids[-1]]) for x in observations]

	for j in range(1, K):
		latest_centroid = observations[centroids[-1]]

		new_dist = np.power(observations - latest_centroid, 2).sum(axis=1)
		for i in range(N):
			temp = min(min_arr[i], new_dist[i])
			min_arr[i] = temp

		s = sum(min_arr)
		probs = [m / s for m in min_arr]
		u = np.random.choice(nums, 1, p=probs)
		centroids.append(int(u[0]))


def print_centroids():
	print(','.join(list(map(str, centroids))), flush=True)


def main():
	k_means_pp()
	print_centroids()

	mk.kmeans([observations.tolist(), centroids, K, N, d, MAX_ITER])


main()
