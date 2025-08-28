from transformers import RobertaTokenizerFast, RobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from modules.datasets.class_dataset import ClassDataset

def compute_similarity(text1, text2):
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
    model = RobertaModel.from_pretrained("FacebookAI/roberta-large")
    
    input_ids1 = tokenizer(text1)['input_ids']
    input_ids2 = tokenizer(text2)['input_ids']
    
    input_ids1 = torch.tensor(input_ids1).unsqueeze(0)
    input_ids2 = torch.tensor(input_ids2).unsqueeze(0)
    
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token
    
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    
    return similarity_score

if __name__ == "__main__":
    data = ClassDataset('FacebookAI/roberta-large')
    data.load('datasets/testing_dataset/', split="train")
    
    for i, example in enumerate(data.ori_data):
        score = compute_similarity(example['course'], example['target'])
        print(f"Comparison index {i}:")
        print("Text 1:", example['course'])
        print("Text 2:", example['target'])
        print("Score:", score)
    
    # text1 = "Amrozi accused his brother, whom he called ""the witness"", of deliberately distorting his evidence."
    # text2 = "Referring to him as only ""the witness"", Amrozi accused his brother of deliberately distorting his evidence."
    
    # score = compute_similarity(text1, text2)
    # print(score)