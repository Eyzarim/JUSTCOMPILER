from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model dari safetensors (shards)
model = AutoModelForCausalLM.from_pretrained("/app/tuned_model")
tokenizer = AutoTokenizer.from_pretrained("/app/tuned_model")

# Simpan model dalam format PyTorch (.bin)
model.save_pretrained("/app/hasil-akhir/llama3_combined",max_shard_size="0GB")
tokenizer.save_pretrained("/app/hasil-akhir/llama3_combined",max_shard_size="0GB")
