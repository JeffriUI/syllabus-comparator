from datasets import load_dataset
from fate.ml.nn.dataset.base import Dataset
from transformers import AutoTokenizer

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

class ClassDataset(Dataset):
    """
    Custom dataset class, refitted from FATE-LLM built-in Dataset class for Sequence Classification
    """
    def __init__(
            self,
            tokenizer_name_or_path,
            text_max_length=512,
            truncation=True,
            padding=True,
            padding_side="right",
            pad_token=None):
        self.ori_data = None
        self.ds = None
        self.tokenizer = None
        self.sample_ids = None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        if 'llama' in tokenizer_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path, add_eos_token=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path)
        self.padding = padding
        self.truncation = truncation
        self.max_length = text_max_length
        self.tokenizer.padding_side = padding_side
        if pad_token is not None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
        
    def load(self, dir, split):
        tokenizer = self.tokenizer
        data = load_dataset(dir, split=split)
        
        def preprocess_text(examples):
            """
            A function to be used in Dataset.map(). Apply preprocessing in the form of 3-steps 
            consisting: 1) stopwords removal, 2) punctuation removal, and 3) lemmatization.
            """
            out = {}
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger_eng')
            stop_words = set(stopwords.words('english'))
            for c in ['course', 'target']:
                words = []
                for i in range(len(examples[c])):
                    tokens = word_tokenize(examples[c][i].lower())
                    no_stopwords = [word for word in tokens if word not in stop_words]
                    no_punctuation = [re.sub(r'[^\w\s]', '', token) 
                                       for token in no_stopwords if re.sub(r'[^\w\s]', '', token)]
                    lemmatizer = WordNetLemmatizer()
                    tagged = pos_tag(no_punctuation)
                    lemmatized = [lemmatizer.lemmatize(
                        word, pos='v' if tag.startswith('V') else 'n') for word, tag in tagged]
                    lemmatized = " ".join(lemmatized)
                    words.append(lemmatized)
                out[c] = words
            return out
        
        def tokenize_text(examples):
            """
            A function to be used in Dataset.map(). Apply tokenization to text pairs of course-target.
            Results in a unified BatchEncoding of the two texts combined into one sequence.
            """
            out = {}
            out = tokenizer(
                examples['course'],
                examples['target'],
                padding=self.padding,
                return_tensors='pt',
                truncation=self.truncation,
                max_length=self.max_length,
            )
            out['labels'] = examples['label']
            return out
        
        preprocessed_data = data.map(preprocess_text, batched=True)
        self.ds = preprocessed_data.map(
            tokenize_text, batched=True, remove_columns=preprocessed_data.column_names)
        self.ds.set_format('torch')
        self.ori_data = preprocessed_data

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return self.ds[idx]

    def __repr__(self):
        return self.tokenizer.__repr__()