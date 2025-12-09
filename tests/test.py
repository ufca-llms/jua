from jua.dataset import Dataset

ds = Dataset("data/jurisprudencia-selecionada.csv", sample_size=1000)

train, test = ds.split_dataset()

# show train sorted by BM25_RANK
print(train.sort_values(by="BM25_RANK", ascending=False).head())

# show test sorted by BM25_RANK
print(test.sort_values(by="BM25_RANK", ascending=False).head())


