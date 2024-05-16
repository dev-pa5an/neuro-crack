from dataset import Dataset

dataset = Dataset(train=True)
samples = [10,100]
for sample in samples:
    plt.imshow(dataset[sample][0])
    plt.xlabel("y="+str(dataset[sample][1].item()))
    plt.title("training data, sample {}".format(int(sample)))
    plt.show()