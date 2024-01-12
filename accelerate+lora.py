import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from argparse import ArgumentParser
import pandas as pd
import bitsandbytes as bnb
import re
from tqdm import tqdm
from peft import PeftModel, PeftConfig

import loralib as lora
# accelerate==0.20.3
# transformers==4.30.2
# Set up the environment
accelerator = Accelerator()


class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, max_length):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        if args.do_train:
            self.data = pd.read_parquet(args.input_file+'/train.parquet')
        else:
            self.data = pd.read_parquet(args.input_file+'/test.parquet')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]
        tokenizer.pad_token = tokenizer.eos_token
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }    

# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch['input_ids'].to(args.device)
#             attention_mask = batch['attention_mask'].to(args.device)
#             labels = batch['labels'].to(args.device)

#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.item()

#     avg_loss = total_loss / len(loader)
#     return avg_loss

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        optimizer.zero_grad()

        # inputs, targets = batch
        # inputs = inputs.to(args.device)
        # targets = targets.to(args.device)
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()


        accelerator.backward(loss)
        scheduler.step()
        optimizer.step()

    accelerator.wait_for_everyone()

    peft_model_id = args.output_file
    model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    return total_loss / len(loader)
    
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

if __name__ == "__main__":
    parser = ArgumentParser()
 
    parser.add_argument("--input_file", default="formatted_data", type=str)
    parser.add_argument("--output_file", default="saved_model", type=str)
    parser.add_argument("--save_path", default="lora_checkpoint", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--do_train", default=False, type=bool)  # vocab_size = 0 means auto (for char-level LM and .txt data)
    parser.add_argument("--max_length", default=400, type=int)
    parser.add_argument("--lora_rank", default=2, type=int)
    parser.add_argument("--lora_alpha", default=1, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=int)

    args = parser.parse_args()

    # Prepare the dataset
    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                            trust_remote_code = True)
    max_length = args.max_length

    dataset = MyCustomDataset(args, tokenizer, max_length)
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the model architecture
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                load_in_4bit = True,
                                                use_safetensors = False,
                                                torch_dtype=torch.float16,
                                                trust_remote_code = True)
    
    '''除了lm_head以外的全部线性层都加'''
    # m = find_all_linear_names(model)
    # print(model)
    pattern = r'\((\w+)\): Linear'
    linear_layers = re.findall(pattern, str(model.modules))
    target_modules = list(set(linear_layers))
    target_modules.remove('lm_head')
    print('target_modules:',target_modules)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    target_modules = ['q_proj','k_proj']

    peft_config = LoraConfig(
            peft_type="LORA",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    
    model = get_peft_model(model, peft_config)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable lora parameters : {total_trainable_params}")
    # Configure the training process
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    # Create a data loader
    model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)

    num_epochs = 1
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, data_loader, optimizer, criterion)

        train_loss = accelerator.gather(tensor=train_loss)

        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}")

