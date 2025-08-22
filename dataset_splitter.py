from datasets import load_dataset

dataset_dir = "datasets/all"
dataset = load_dataset(dataset_dir, split="train")

dataset = dataset.train_test_split(train_size=0.6, seed=42)
dataset['train'].to_csv(f"{dataset_dir}/train.csv")
dataset = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['test'].to_csv(f"{dataset_dir}/test.csv")
dataset['train'].to_csv(f"{dataset_dir}/val.csv")