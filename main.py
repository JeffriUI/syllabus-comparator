import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
from huggingface_hub import login
from fate.arch import Context
from fate.arch.launchers.multiprocess_launcher import launch
from fate.ml.nn.homo.fedavg import FedAVGArguments
from fate_llm.data.tokenizers.cust_tokenizer import get_tokenizer
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoConfig, DataCollatorWithPadding, RobertaForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_curve, auc, ConfusionMatrixDisplay

# Local modules for adapted classes and functions
from modules.datasets.seq_cls_dataset import SeqClsDataset
from modules.models.roberta import Roberta
from modules.fedmkt.fedmkt import FedMKTTrainingArguments, FedMKTLLM, FedMKTSLM
from modules.fedmkt.token_alignment.vocab_mapping import get_vocab_mappings

llm_pretrained_path = "FacebookAI/roberta-large"
slm_pretrained_path = "FacebookAI/roberta-base"

slm_to_llm_vocab_mapping_path = "vocab_mappings/roberta_small_to_roberta.json"
llm_to_slm_vocab_mapping_path = "vocab_mappings/roberta_to_roberta_small.json"

global_epochs = 5
batch_size = 4
llm_lr = 3e-3
slm_lr = 3e-3

dataset_directory = "datasets"

llm_model_saved_directory = "models/llm"
slm_models_saved_directory = ["models/slm_1",
                              "models/slm_2"]

logs = {}

def login_hf():
    load_dotenv()
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
    login(token=HF_ACCESS_TOKEN)

# Building vocab mappings
def build_vocab_mappings():
    _ = get_vocab_mappings(slm_pretrained_path, llm_pretrained_path, slm_to_llm_vocab_mapping_path, num_processors=16)
    _ = get_vocab_mappings(llm_pretrained_path, slm_pretrained_path, llm_to_slm_vocab_mapping_path, num_processors=16)

def train_llm(ctx, pub_data_dir):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = Roberta(
        pretrained_path=llm_pretrained_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        torch_dtype="bfloat16"
    )
    
    pub_data = SeqClsDataset(llm_pretrained_path)
    pub_data.load(pub_data_dir, split="train")

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
        logging_strategy="epoch",
        save_strategy="epoch",
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(llm_pretrained_path).vocab_size,
        post_fedavg=True,
        top_k_logits_keep=model.config.num_labels,
        skip_align=True,
    )

    fed_args = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    with open(slm_to_llm_vocab_mapping_path, "r") as fin:
        vocab_mapping = json.loads(fin.read())

    slm_tokenizers = [get_tokenizer(slm_pretrained_path)] * 2
    tokenizer = get_tokenizer(llm_pretrained_path)

    trainer = FedMKTLLM(
        ctx=ctx,
        model=model,
        training_args=training_args,
        fed_args=fed_args,
        train_set=pub_data.ds,
        tokenizer=tokenizer,
        slm_tokenizers=slm_tokenizers,
        slm_to_llm_vocab_mappings=vocab_mapping,
        save_trainable_weights_only=True,
    )

    log = trainer.train()
    trainer.save_model(llm_model_saved_directory)
    
    return log

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
    
    pub_data = SeqClsDataset(llm_pretrained_path)
    pub_data.load(pub_data_dir, split="train")
    
    priv_data = SeqClsDataset(slm_pretrained_path)
    priv_data.load(priv_data_dir, split="train")
    
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
        logging_strategy="epoch",
        save_strategy="epoch",
        use_cpu=False,
        vocab_size=AutoConfig.from_pretrained(slm_pretrained_path).vocab_size,
        post_fedavg=True,
        top_k_logits_keep=model.config.num_labels,
        skip_align=True,
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
        pub_train_set=pub_data.ds,
        priv_train_set=priv_data.ds,
        tokenizer=tokenizer,
        save_trainable_weights_only=True,
        llm_tokenizer=llm_tokenizer,
        llm_to_slm_vocab_mapping=vocab_mapping,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    priv_log, fedmkt_log = trainer.train()
    trainer.save_model(slm_models_saved_directory[slm_index])

    return priv_log, fedmkt_log

def train_direct(data_dir):
    from modules.trainer.trainer import TrainingArguments, Trainer
    
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
    
    data = SeqClsDataset(slm_pretrained_path)
    data.load(data_dir, split="train")
    
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
        num_train_epochs=global_epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        use_cpu=False,
    )

    tokenizer = get_tokenizer(slm_pretrained_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=data.ds,
        args=training_args,
        save_trainable_weights_only=True
    )

    log = trainer.train()
    trainer.save_model("models/direct")

    return log

def test(data_dir, model_dir, logs_dir):
    robertaModel = RobertaForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=slm_pretrained_path,
        torch_dtype=torch.bfloat16
    )
    
    direct_model = PeftModel.from_pretrained(
        model=robertaModel,
        model_id=f'{model_dir}/direct'
    )
    
    fed_1_model = PeftModel.from_pretrained(
        model=robertaModel,
        model_id=f'{model_dir}/slm_1'
    )
    
    fed_2_model = PeftModel.from_pretrained(
        model=robertaModel,
        model_id=f'{model_dir}/slm_2'
    )
    
    models = {
        'Direct': direct_model,
        'FedClient1': fed_1_model,
        'FedClient2': fed_2_model
    }
    
    acc_scores = {}
    f1_scores = {}
    conf_matrix = {}
    
    test_data = SeqClsDataset(slm_pretrained_path)
    test_data.load(data_dir, split="test")
    
    y_true = test_data.ds['labels'].tolist()
    inputs = test_data.ds.remove_columns('labels').data.to_pydict()
    for key in inputs.keys():
        inputs[key] = torch.LongTensor(inputs[key])

    for model in models.keys():
        with torch.no_grad():
            logits = models[model](**inputs).logits

        probs = logits.to(torch.float16).softmax(dim=-1).detach()
        y_pred = probs.argmax(dim=-1).numpy().tolist()
        y_score = probs.numpy()[:,1].tolist()
        acc_scores[model] = accuracy_score(y_true, y_pred)
        f1_scores[model] = f1_score(y_true, y_pred, average='weighted')
        conf_matrix[model] = confusion_matrix(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        prec_rec_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model} (AUC = {prec_rec_auc:.2f})')
    
    # Visualize Precision-Recall Curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (All RoBERTa-125M)')
    plt.legend()
    plt.savefig("./graphs/prec_rec_curve.png")
    
    # Visualize Accuracy-scores distribution
    plt.clf()
    bars = plt.bar(acc_scores.keys(), acc_scores.values(), bottom=np.zeros(3))
    plt.title('Accuracy-scores Distribution')
    plt.bar_label(bars, fmt='%.4f')
    plt.savefig("./graphs/acc_scores.png")
    
    # Visualize F1-scores distribution
    plt.clf()
    bars = plt.bar(f1_scores.keys(), f1_scores.values(), bottom=np.zeros(3))
    plt.title('F1-scores Distribution')
    plt.bar_label(bars, fmt='%.4f')
    plt.savefig("./graphs/f1_scores.png")
    
    # Visualize Confusion Matrix
    for model in models.keys():
        plt.clf()
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix[model], 
                                            display_labels=["Non-equivalent", "Equivalent"])
        cm_display.plot(cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.savefig(f"./graphs/confusion_matrix_{model}.png")
    
    log_keys_to_label = {
        "server": "Server (RoBERTa-355M)",
        "client_1": "Client 1 (RoBERTa-125M)",
        "client_2": "Client 2 (RoBERTa-125M)",
        "control": "Control (RoBERTa-125M)"
    }
    
    filenames = ["server", "client_1", "client_2", "control"]
    logs = {}
    for fname in filenames:
        with open(f"{logs_dir}/{fname}_logs.json", "r") as fin:
            temp = json.loads(fin.read())
        logs[fname] = temp
    
    # Visualize each model's loss progression by epoch
    loss = {
        "server": [],
        "client_1": [],
        "client_2": [],
        "control": []
    }
    
    client_1_priv_loss = []
    client_1_fedmkt_loss = []
    client_2_priv_loss = []
    client_2_fedmkt_loss = []
    
    for i in range(global_epochs):
        loss["server"].append(logs["server"][i][0]["loss"])
        client_1_priv_loss.append(logs["client_1"]["priv"][i][0]["loss"])
        client_1_fedmkt_loss.append(logs["client_1"]["fedmkt"][i][0]["loss"])
        client_2_priv_loss.append(logs["client_2"]["priv"][i][0]["loss"])
        client_2_fedmkt_loss.append(logs["client_2"]["fedmkt"][i][0]["loss"])
        loss["control"].append(logs["control"][i]["loss"])
    
    # Averaging client loss values, due to being separated to a private trainer and FedMKT trainer
    for i in range(global_epochs):
        loss["client_1"].append((client_1_priv_loss[i] + client_1_fedmkt_loss[i]) / 2)
        loss["client_2"].append((client_2_priv_loss[i] + client_2_fedmkt_loss[i]) / 2)
    
    
    plt.clf()
    plt.figure()
    for key in loss.keys():
        x = range(1, global_epochs+1)
        y = loss[key]
        
        plt.plot(x, y, marker='o', label=f'{log_keys_to_label[key]}')
        
        for (xi, yi) in zip(x, y):
            plt.annotate(f'{yi:.4f}', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title('Loss Chart by Model by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./graphs/loss.png")
    
    # Visualize each model's runtime accumulative by epoch
    runtime = {}
    
    server_runtime = []
    client_1_priv_runtime = []
    client_1_fedmkt_runtime = []
    client_2_priv_runtime = []
    client_2_fedmkt_runtime = []
    
    # Storing runtime to an array by epoch, due to how FedMKT works,
    # separating each epoch into a different training instance
    # resulting in a runtime for each epoch 
    # rather than a total runtime at the end of the epoch
    # For clients, due to being separated to a private trainer and FedMKT trainer,
    # plus a delay from waiting server to finish training and send their public logits,
    # values by epoch needs to be summed first
    for i in range(global_epochs):
        server_runtime.append(logs["server"][i][1]["train_runtime"])
        client_1_priv_runtime.append(logs["client_1"]["priv"][i][1]["train_runtime"])
        client_1_fedmkt_runtime.append(logs["client_1"]["fedmkt"][i][1]["train_runtime"])
        client_2_priv_runtime.append(logs["client_2"]["priv"][i][1]["train_runtime"])
        client_2_fedmkt_runtime.append(logs["client_2"]["fedmkt"][i][1]["train_runtime"])
    
    runtime["control"] = logs["control"][global_epochs]["train_runtime"]
    
    # Summing all runtime to get the total runtime
    for i in range(global_epochs):
        runtime["server"] = sum(server_runtime)
        runtime["client_1_priv"] = sum(client_1_priv_runtime)
        runtime["client_1_fedmkt"] = sum(client_1_fedmkt_runtime)
        runtime["client_2_priv"] = sum(client_2_priv_runtime)
        runtime["client_2_fedmkt"] = sum(client_2_fedmkt_runtime)
    
    labels = ["Control", "Server", "Client 1", "Client 2"]
    runtime_values = {
        'Private Trainer': np.array([runtime["control"], 0, runtime["client_1_priv"], runtime["client_2_priv"]]),
        'FedMKT Trainer': np.array([0, runtime["server"], runtime["client_1_fedmkt"], runtime["client_2_fedmkt"]])
    }
    
    plt.clf()
    bottom = np.zeros(4)
    for trainer, runtime in runtime_values.items():
        bars = plt.bar(labels, runtime, label=trainer, bottom=bottom)
        bottom += runtime
        max_limit = (int(max(bottom) / 5000) + 1) * 5000
        plt.ylim(top=max_limit)
        plt.bar_label(bars, fmt='%.0f', label_type='center')
    
    plt.title('Total Runtime Chart by Model')
    plt.ylabel('Runtime (in Seconds)')
    plt.legend()
    plt.savefig("./graphs/runtime.png")

def run(ctx: Context):
    pub_data_dir = f'{dataset_directory}/public'
    priv_data_dir_1 = f'{dataset_directory}/private_1'
    priv_data_dir_2 = f'{dataset_directory}/private_2'
    
    if ctx.is_on_arbiter:
        server_logs = {}
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        server_logs = train_llm(ctx, pub_data_dir)
        with open("logs/server_logs.json", "w") as fout:
            json.dump(server_logs, fout)
    elif ctx.is_on_guest:
        client_1_logs = {}
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        client_1_logs["priv"], client_1_logs["fedmkt"] = train_slm(ctx, pub_data_dir, priv_data_dir_1)
        with open("logs/client_1_logs.json", "w") as fout:
            json.dump(client_1_logs, fout)
    else:
        client_2_logs = {}
        if ctx.local.party[1] == "9999":
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        else:
            raise ValueError(f"party_id={ctx.local.party[1]} is illegal")

        client_2_logs["priv"], client_2_logs["fedmkt"] = train_slm(ctx, pub_data_dir, priv_data_dir_2)
        with open("logs/client_2_logs.json", "w") as fout:
            json.dump(client_2_logs, fout)
    
    data_dir = f"{dataset_directory}/all"
    model_dir = "models"
    logs_dir = "logs"
    
    if ctx.is_on_guest:
        control_logs = {}
        control_logs = train_direct(data_dir)
        with open("logs/control_logs.json", "w") as fout:
            json.dump(control_logs, fout)
        test(data_dir, model_dir, logs_dir)

if __name__ == "__main__":
    login_hf()
    
    # Command line to build vocab mappings
    # Doesn't need to be run if build_vocab_mappings.py was run separately
    # build_vocab_mappings()
    
    launch(run)