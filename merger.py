import torch
from transformers import AutoModelForCausalLM

# Load model dari safetensors shards
model = AutoModelForCausalLM.from_pretrained("/app/trainer/tuned_model", use_safetensors=True)

# Simpan model secara langsung menggunakan PyTorch tanpa sharding
torch.save(model.state_dict(), "/app/pytorch_model_singlefile.bin")