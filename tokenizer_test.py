from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('datasets/test_real_courses/', split="test")
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large')
tokenizer.padding_side = "right"
tokenized_ds = tokenizer(
    dataset['text'], 
    padding=True,
    return_tensors='pt',
    truncation=True,
    max_length=512,
)
print(tokenized_ds['input_ids'].size())