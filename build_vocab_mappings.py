from modules.token_alignment.vocab_mapping import get_vocab_mappings
from huggingface_hub import login
from dotenv import load_dotenv
import os

llm_pretrained_path = "meta-llama/Llama-2-7b-hf"
slm_pretrained_path = "FacebookAI/roberta-large"

slm_to_llm_vocab_mapping_path = "vocab_mappings/roberta_to_llama.json"
llm_to_slm_vocab_mapping_path = "vocab_mappings/llama_to_roberta.json"

def build_vocab_mappings():
    load_dotenv()

    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

    login(token=HF_ACCESS_TOKEN)

    # Building vocab mappings
    _ = get_vocab_mappings(slm_pretrained_path, llm_pretrained_path, slm_to_llm_vocab_mapping_path, num_processors=16)
    _ = get_vocab_mappings(llm_pretrained_path, slm_pretrained_path, llm_to_slm_vocab_mapping_path, num_processors=16)

if __name__ == "__main__":
    build_vocab_mappings()