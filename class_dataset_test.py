from modules.datasets.class_dataset import ClassDataset

data = ClassDataset('FacebookAI/roberta-large')

data.load('datasets/public/', split="train")
print(data.ds.__getitem__(0))
print(data.ori_data[0])