import hopfield
import plot
import numpy as np

def pict_data():
	with open("data/pict.dat") as f:
		data_list = list(map(int, f.read().split(",")))
		return np.array(data_list).reshape((11, 1024))

def reformat_data(data):
	return data.reshape((32, 32))

def sequential():
	data_array = pict_data()
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
	data_array = pict_data()
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

def test_capacity(start, end):
	data_array = pict_data()
	errors = []
	for n in range(start, end, 1):
		print(n)
		X0 = data_array[:n]
		W = hopfield.weights(X0)
		error = []
		for x in X0:
			choices = np.random.choice(len(x), size=int(20*len(x)/100), replace=False)
			x_noise = add_noise(x, choices)
			error.append(calculate_error(x, hopfield.recall_until_stable(W, x_noise)))
		errors.append(sum(error)/len(error))
	
	print(errors)

	plot.plot(range(start, end), errors)

def test_capacity_random(start, end):
	data_array = pict_data()
	errors = []
	for n in range(start, end, 1):
		X0 = list(data_array[:3])
		for _ in range(n-3):
			X0.append(generate_random())
		print(len(X0))
		W = hopfield.weights(X0)
		error = []
		for x in X0:
			choices = np.random.choice(len(x), size=int(20*len(x)/100), replace=False)
			x_noise = add_noise(x, choices)
			error.append(calculate_error(x, hopfield.recall_until_stable(W, x_noise)))
		errors.append(sum(error)/len(error))
	
	print(errors)

	plot.plot(range(start, end), errors)

def test_random(N=100, should_add_noise=False, should_remove_self_conn=False):
	data_array = [generate_random(100) for _ in range(300)]
	errors = []
	for n in range(1, N):
		print(n)
		X0 = data_array[:n]
		if should_remove_self_conn:
			W = hopfield.weights(X0, True)
		else:
			W = hopfield.weights(X0)
		error = []
		for x in X0:
			x_noise = x
			if should_add_noise:
				choices = np.random.choice(len(x), size=int(20*len(x)/100), replace=False)
				x_noise = add_noise(x, choices)
			error.append(1 if (x == hopfield.recall_until_stable(W, x_noise)).all() else 0)
		errors.append(sum(error))

	print(errors)

	plot.plot(range(1, N), errors)

def generate_random(N=1024):
	return np.random.choice([-1, 1], size=N)

def generate_random_biased(N=1024, ):
	return np.random.choice([-1, 1], size=N,  p=[1./3, 2./3])



if __name__ == "__main__":
	#sequential()
	#noise()
	#test_capacity(4, 8)
	#test_capacity_random(4, 20)
	test_random(25)
	test_random(25, should_remove_self_conn=True)
	#test_random(True)