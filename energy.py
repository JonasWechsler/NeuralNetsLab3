import hopfield
import attractors

def energy_at_attractors():
	W = attractors.get_weights()
	_, A = attractors.get_attractors()
	energy = [hopfield.energy(W, x) for x in A]
	for a, b in zip(energy, A):
		print(a, b)

if __name__ == "__main__":
	energy_at_attractors()