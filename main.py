import os
import json
# from subprocess import _InputString
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from huggingface_hub import login
from fate.arch import Context
from fate.arch.launchers.multiprocess_launcher import launch
from fate.ml.nn.homo.fedavg import FedAVGArguments
from modules.fedmkt.fedmkt import FedMKTTrainingArguments, FedMKTLLM, FedMKTSLM
from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
from fate_llm.model_zoo.pellm.roberta import Roberta
from peft import LoraConfig, TaskType
from transformers import AutoConfig, DataCollatorWithPadding, RobertaForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, ConfusionMatrixDisplay

# Local modules for adapted classes and functions
from modules.datasets.class_dataset import ClassDataset
from modules.models.class_llama import ClassLLaMa
from modules.token_alignment.vocab_mapping import get_vocab_mappings

llm_pretrained_path = "meta-llama/Llama-2-7b-hf"
slm_pretrained_path = "FacebookAI/roberta-large"

slm_to_llm_vocab_mapping_path = "vocab_mappings/roberta_to_llama.json"
llm_to_slm_vocab_mapping_path = "vocab_mappings/llama_to_roberta.json"

global_epochs = 5
batch_size = 4
llm_lr = 3e-5
slm_lr = 3e-5

dataset_directory = "datasets"

llm_model_saved_directory = "models/llm"
slm_models_saved_directory = ["models/slm_1",
                              "models/slm_2"]

def login_hf():
    load_dotenv()
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
    login(token=HF_ACCESS_TOKEN)

# Building vocab mappings
def build_vocab_mappings():
    # Building vocab mappings
    _ = get_vocab_mappings(slm_pretrained_path, llm_pretrained_path, slm_to_llm_vocab_mapping_path, num_processors=16)
    _ = get_vocab_mappings(llm_pretrained_path, slm_pretrained_path, llm_to_slm_vocab_mapping_path, num_processors=16)

def train_llm(ctx, pub_data_dir):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    
    pub_data_train = ClassDataset(llm_pretrained_path)
    pub_data_train.load(pub_data_dir, split="train")
    pub_data_val = ClassDataset(llm_pretrained_path)
    pub_data_val.load(pub_data_dir, split="validation")

    model = ClassLLaMa(
        pretrained_path=llm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="bfloat16"
    )

    training_args = FedMKTTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=llm_lr,
        output_dir="outputs/",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.008,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(llm_pretrained_path).vocab_size,
        post_fedavg=True,
        evaluation_strategy="epoch",
    )

    fed_args = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    with open(slm_to_llm_vocab_mapping_path, "r") as fin:
        vocab_mapping = json.loads(fin.read())

    slm_tokenizers = get_tokenizer(slm_pretrained_path)
    tokenizer = get_tokenizer(llm_pretrained_path)

    trainer = FedMKTLLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        fed_args=fed_args,
        train_set=pub_data_train.ds,
        val_set=pub_data_val.ds,
        tokenizer=tokenizer,
        slm_tokenizers=slm_tokenizers,
        slm_to_llm_vocab_mappings=vocab_mapping,
        save_trainable_weights_only=True,
    )

    trainer.train()
    trainer.save_model(llm_model_saved_directory)

def train_slm(ctx, pub_data_dir, priv_data_dir):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = Roberta(
        pretrained_path=slm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="bfloat16"
    )
    
    pub_data_train = ClassDataset(llm_pretrained_path)
    pub_data_train.load(pub_data_dir, split="train")
    
    priv_data_train = ClassDataset(slm_pretrained_path)
    priv_data_train.load(priv_data_dir, split="train")
    
    priv_data_val = ClassDataset(slm_pretrained_path)
    priv_data_val.load(priv_data_dir, split="validation")
    
    slm_index = int(priv_data_dir[-1]) - 1

    training_args = FedMKTTrainingArguments(
        global_epochs=global_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=slm_lr,
        output_dir="outputs/",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.008,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(slm_pretrained_path).vocab_size,
        post_fedavg=True,
        evaluation_strategy="epoch",
    )

    fed_args = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    with open(llm_to_slm_vocab_mapping_path, "r") as fin:
        vocab_mapping = json.loads(fin.read())

    tokenizer = get_tokenizer(slm_pretrained_path)
    llm_tokenizer = get_tokenizer(llm_pretrained_path)

    trainer = FedMKTSLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        fed_args=fed_args,
        pub_train_set=pub_data_train.ds,
        priv_train_set=priv_data_train.ds,
        val_set=priv_data_val.ds,
        tokenizer=tokenizer,
        save_trainable_weights_only=True,
        llm_tokenizer=llm_tokenizer,
        llm_to_slm_vocab_mapping=vocab_mapping,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    trainer.save_model(slm_models_saved_directory[slm_index])
    
def train_direct(data_dir):
    from transformers import Trainer
    from modules.trainer.trainer import TrainingArguments
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = Roberta(
        pretrained_path=slm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="bfloat16"
    )
    
    data_train = ClassDataset(slm_pretrained_path)
    data_train.load(data_dir, split="train")
    data_val = ClassDataset(slm_pretrained_path)
    data_val.load(data_dir, split="validation")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=slm_lr,
        output_dir="outputs/",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.008,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=False,
        evaluation_strategy="epoch",
    )

    tokenizer = get_tokenizer(slm_pretrained_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=data_train.ds,
        eval_dataset=data_val.ds,
        args=training_args,
    )

    trainer.train()
    trainer.save_model("models/direct")

def test(data_dir, model_dir):
    direct_model = RobertaForSequenceClassification.from_pretrained(f'{model_dir}/direct')
    fed_1_model = RobertaForSequenceClassification.from_pretrained(f'{model_dir}/slm_1')
    fed_2_model = RobertaForSequenceClassification.from_pretrained(f'{model_dir}/slm_2')
    
    models = {
        'Direct': direct_model,
        'FedClient1': fed_1_model,
        'FedClient2': fed_2_model
    }
    
    f1_scores = {}
    conf_matrix = {}
    
    test_data = ClassDataset(slm_pretrained_path)
    test_data.load(data_dir, split="test")
    
    y_true = test_data.ds['labels'].to_list()
    inputs = test_data.ds.remove_columns('labels')

    for model in models.keys():
        with torch.no_grad():
            logits = models[model](**inputs).logits

        y_pred = logits.argmax().tolist()
        f1_scores[model] = f1_score(y_true, y_pred, average='weighted')
        conf_matrix[model] = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    
    # Visualize AUC ROC Curve
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Two Models')
    plt.legend()
    plt.savefig("graphs/auc_roc_curve.png")
    plt.show()
    
    # Visualize F1-scores distribution
    bars = plt.bar(f1_scores.keys(), f1_scores.values(), bottom=np.zeros(3))
    plt.title('F1-scores Distribution')
    plt.bar_label(bars, fmt='%.4f')
    plt.savefig("graphs/f1_scores.png")
    plt.show()
    
    # Visualize Confusion Matrix
    fig, axs = plt.subplots(1, 3)
    for i, model in enumerate(models.keys()):
        axs[i] = ConfusionMatrixDisplay(confusion_matrix=conf_matrix[model])
        axs[i].plot(cmap=plt.cm.Blues)
        axs[i].set_title(model + 'Model')
    fig.suptitle('Confusion Matrix Across Models', va='bottom')
    plt.savefig("graphs/confusion_matrix.png")
    plt.show()

def run(ctx: Context):
    # Command line to build vocab mappings
    # Or, run build_vocab_mappings.py separately to save time
    # build_vocab_mappings()
    
    pub_data_dir = f'{dataset_directory}/public'
    priv_data_dir_1 = f'{dataset_directory}/private_1'
    priv_data_dir_2 = f'{dataset_directory}/private_2'
    
    if ctx.is_on_arbiter:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        train_llm(ctx, pub_data_dir)
    elif ctx.is_on_guest:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        train_slm(ctx, pub_data_dir, priv_data_dir_1)
    else:
        if ctx.local.party[1] == "9999":
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        #     slm_idx = 1
        # elif ctx.local.party[1] == "10000":
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #     slm_idx = 2
        # elif ctx.local.party[1] == "10001":
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        #     slm_idx = 3
        else:
            raise ValueError(f"party_id={ctx.local.party[1]} is illegal")

        train_slm(ctx, pub_data_dir, priv_data_dir_2)

if __name__ == "__main__":
    login_hf()
    launch(run)
    data_dir = f"{dataset_directory}/all"
    model_dir = "models"
    train_direct(data_dir)
    test(data_dir, model_dir)