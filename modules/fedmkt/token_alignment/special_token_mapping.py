# A copy of special_token_mapping.py from FATE-LLM
# updated with support for RoBERTa

import transformers


TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: '▁',
    transformers.LlamaTokenizerFast: '▁',
    transformers.GPTNeoXTokenizerFast: 'Ġ',
    transformers.GPT2TokenizerFast: 'Ġ',
    transformers.GPT2Tokenizer: 'Ġ',
    transformers.BloomTokenizerFast: 'Ġ',
    transformers.RobertaTokenizer: 'Ġ',
    transformers.RobertaTokenizerFast: 'Ġ',
}