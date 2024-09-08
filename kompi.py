from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model dari file safetensors
model = AutoModelForCausalLM.from_pretrained("/app/tuned_model", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/app/tuned_model")

# Simpan kembali dalam format `.bin` atau format lain yang didukung untuk konversi
model.save_pretrained("/app/hasil-akhir/llama3.bin")
tokenizer.save_pretrained("/app/hasil-akhir/llama3_token.bin")