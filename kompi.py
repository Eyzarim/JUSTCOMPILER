from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model dari safetensors (shards)
model = AutoModelForCausalLM.from_pretrained("/app/tuned_model", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/app/tuned_model")

# Simpan model dalam format PyTorch (.bin)
model.save_pretrained("/app/hasil-akhir/llama3_combined")
tokenizer.save_pretrained("/app/hasil-akhir/llama3_combined")
