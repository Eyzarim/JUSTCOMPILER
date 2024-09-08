from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model dari .safetensors
model = AutoModelForCausalLM.from_pretrained("/app/hasil-akhir", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/app/hasil-akhir")

# Simpan model dalam format PyTorch (.bin)
model.save_pretrained("/app/hasil-akhir/pytorch_model")
tokenizer.save_pretrained("/app/hasil-akhir/pytorch_model")
