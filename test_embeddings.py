from transformers import RobertaTokenizerFast, RobertaModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ["Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.", "Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998."]

# Load model from HuggingFace Hub
tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
model = RobertaModel.from_pretrained("FacebookAI/roberta-large")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

print(cosine_similarity(sentence_embeddings[0], sentence_embeddings[1]))
