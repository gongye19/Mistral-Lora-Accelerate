from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


model_name_or_path = "/home/featurize/work/微调/accelerate+lora/saved_lora_adapters"

device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code = True)
model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path,load_in_4bit = True,trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code = True)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, pad_token_id=model.config.eos_token_id, max_new_tokens=100, do_sample=True)
ans = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]
# tokenizer解码
print(ans)