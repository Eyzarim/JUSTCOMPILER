import nltk
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
import subprocess
import json

# Memastikan NLTK siap untuk digunakan (download tokenizer)
nltk.download('punkt_tab')

# Fungsi untuk melakukan inferensi menggunakan model yang sudah di-load di Ollama
def get_predictions(input_text):
    # Jalankan perintah ollama untuk melakukan inferensi dengan model GGUF
    command = f"ollama run llama_kesehatan '{input_text}'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    
    # Ambil output dari hasil inferensi
    output = result.stdout.decode('utf-8')
    return output.strip()

# Fungsi untuk menghitung BLEU Score antara prediksi dan referensi
def calculate_bleu_score(reference, prediction):
    reference_tokenized = nltk.word_tokenize(reference)
    prediction_tokenized = nltk.word_tokenize(prediction)
    return sentence_bleu([reference_tokenized], prediction_tokenized)

# Memuat dataset ChatDoctor-HealthCareMagic
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

# Misalnya kolom 'question' adalah input, dan 'answer' adalah referensi
input_texts = dataset['input']  # Gantilah dengan nama kolom yang sesuai
references = dataset['output']  # Jawaban dokter yang benar (referensi)

# Menyimpan hasil prediksi dan BLEU score
predictions = []
bleu_scores = []

# Loop untuk melakukan inferensi dan menghitung BLEU score
for input_text, reference in zip(input_texts, references):
    prediction = get_predictions(input_text)
    predictions.append(prediction)
    
    # Hitung BLEU score antara referensi dan prediksi
    bleu_score = calculate_bleu_score(reference, prediction)
    bleu_scores.append(bleu_score)
    print(f"Input: {input_text}")
    print(f"Prediction: {prediction}")
    print(f"Reference: {reference}")
    print(f"BLEU Score: {bleu_score}")
    print('-' * 50)

# Menyimpan hasil prediksi dan BLEU score ke dalam file
with open("predictions_and_bleu_scores.json", "w") as f:
    json.dump({"predictions": predictions, "bleu_scores": bleu_scores}, f)

print("Hasil prediksi dan BLEU score sudah disimpan di 'predictions_and_bleu_scores.json'")
