from train import main


def score():
	time = 5
	names = ["cora", "citeseer"]
	models = ['GCN', 'GraphSage', 'GAT']
	scores = {name: {model: None for model in models} for name in names}
	for name in names:
		for model in models:
			mean_acc = 0
			for i in range(time):
				acc = main(name, model)
				print(acc)
				mean_acc += acc / time
			scores[name][model] = mean_acc

	print(scores)


if __name__ == '__main__':
	score()
