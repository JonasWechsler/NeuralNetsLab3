import hopfield
import plot
import numpy as np

def reformat_data(data):
	return data.reshape((32, 32))

def sequential():
	with open("data/pict.dat") as f:
		data_list = list(map(int, f.read().split(",")))
		data_array = np.array(data_list).reshape((11, 1024))
	X0 = data_array[:3]
	W = hopfield.weights(X0)
	X = [data_array[9], data_array[10]]
	# for x in X0:
	# 	plot.plot_heatmap(reformat_data(x))
	# 	plot.plot_heatmap(reformat_data(hopfield.recall_until_stable(W, x)))
	for x in X:
		plot.plot_heatmap(reformat_data(x))
		plot.plot_heatmap(reformat_data(hopfield.recall_until_stable(W, x)))
		res_array = hopfield.recall_sequentially(W, x)
		for r in res_array:
			plot.plot_heatmap(reformat_data(np.array(r)))

def calculate_error(data, expected):
	return 100*sum(1 for i, j in zip(data, expected) if i != j)/len(data)

def add_noise(data, choices):
	new_data = list(data)
	for c in choices:
		new_data[c] *= -1
	return new_data

def noise():
	with open("data/pict.dat") as f:
		data_list = list(map(int, f.read().split(",")))
		data_array = np.array(data_list).reshape((11, 1024))
	X0 = data_array[:3]
	W = hopfield.weights(X0)
	for x in X0:
		error = []
		for i in range(0, 101, 10):
			choices = np.random.choice(len(x), size=int(i*len(x)/100), replace=False)
			x_noise = add_noise(x, choices)
			error.append(calculate_error(x, hopfield.recall_until_stable(W, x_noise)))
		print(error)
		plot.plot_points(error)



if __name__ == "__main__":
	#sequential()
	noise()