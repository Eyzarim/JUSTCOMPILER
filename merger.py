from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM

# Load model dari .safetensors
# Load model dari safetensors shards
model = AutoModelForCausalLM.from_pretrained("/app/trainer/tuned_model", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/app/trainer/tuned_model")

# Simpan model dalam format PyTorch (.bin)
model.save_pretrained("/app/hasil-akhir", max_shard_size="50GB")
tokenizer.save_pretrained("/app/hasil-akhir")
# Simpan model secara langsung menggunakan PyTorch tanpa sharding
torch.save(model.state_dict(), "/app/hasil-akhir/pytorch_model_singlefile.bin")