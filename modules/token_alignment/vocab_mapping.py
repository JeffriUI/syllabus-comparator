# A copy of vocab_mapping.py from FATE-LLM
# updated with support for RoBERTa

import json
import editdistance
import tqdm
import multiprocessing
import logging

from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
from modules.token_alignment.special_token_mapping import TOKENIZER_TO_SPECIAL_TOKEN

logger = logging.getLogger(__name__)


def find_best_mapping(x, base_tokens, blending_model_special_token, base_model_special_token, best_one=True):
    """code refer to https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/vocab_mapping.py#L82"""
    tmp_x = x.replace(blending_model_special_token, base_model_special_token)
    if tmp_x in base_tokens:
        return tmp_x, tmp_x
    else:
        if best_one:
            return tmp_x, min([(y, editdistance.eval(tmp_x, y)) for y in base_tokens], key=lambda d: d[1])[0]
        else:
            token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
            min_distance = min(item[1] for item in token_and_distance)
            shortest_distance_tokens = [item[0] for item in token_and_distance if item[1] == min_distance]
            return tmp_x, shortest_distance_tokens


def get_vocab_mappings(model_name_or_path, candidate_model_name_or_path, vocab_mapping_save_path, num_processors=8):
    ori_tokenizer = get_tokenizer(model_name_or_path)
    candidate_tokenizer = get_tokenizer(candidate_model_name_or_path)

    ori_special_tok = TOKENIZER_TO_SPECIAL_TOKEN[ori_tokenizer.__class__]
    candidate_special_tok = TOKENIZER_TO_SPECIAL_TOKEN[candidate_tokenizer.__class__]

    candidate_tokens = list(candidate_tokenizer.get_vocab().keys())

    with multiprocessing.Pool(num_processors) as process_pool:
        func_args = [(tok, candidate_tokens, ori_special_tok, candidate_special_tok) for tok in ori_tokenizer.get_vocab()]

        vocab_mappings = dict(tqdm.tqdm(process_pool.starmap(find_best_mapping, func_args)),
                              total=len(ori_tokenizer.get_vocab()))

    with open(vocab_mapping_save_path, "w") as fout:
        json.dump(vocab_mappings, fout)

    return vocab_mappings