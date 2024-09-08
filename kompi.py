from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model dan tokenizer dari safetensors
model = AutoModelForCausalLM.from_pretrained("/app/tuned_model", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/app/tuned_model")

# Simpan model ke direktori, bukan ke satu file
model.save_pretrained("/app/hasil-akhir/llama3")  # Menyimpan model ke direktori llama3
tokenizer.save_pretrained("/app/hasil-akhir/llama3_token")  # Menyimpan tokenizer ke direktori llama3_token