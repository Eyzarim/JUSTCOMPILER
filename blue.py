import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from datasets import load_dataset
import subprocess
import json
import os

# Memastikan NLTK siap untuk digunakan
nltk.download('punkt')

# Membuat folder log_latihan jika belum ada
if not os.path.exists("log_latihan"):
    os.makedirs("log_latihan")

# Fungsi untuk melakukan inferensi menggunakan model yang sudah di-load di Ollama
def get_predictions(input_text):
    # Jalankan perintah ollama untuk melakukan inferensi dengan model GGUF
    command = f"ollama run llama_kesehatan '{input_text}'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    
    # Ambil output dari hasil inferensi
    output = result.stdout.decode('utf-8')
    return output.strip()

# Fungsi untuk menghitung BLEU Score
def calculate_bleu_score(reference, prediction):
    reference_tokenized = nltk.word_tokenize(reference)
    prediction_tokenized = nltk.word_tokenize(prediction)
    return sentence_bleu([reference_tokenized], prediction_tokenized)

# Fungsi untuk menghitung METEOR Score
def calculate_meteor_score(reference, prediction):
    return meteor_score([reference], prediction)

# Fungsi untuk menghitung ROUGE Score
def calculate_rouge_scores(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)

# Memuat dataset ChatDoctor-HealthCareMagic
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

# Hanya mengambil 200 data
dataset = dataset.select(range(200))

# Misalnya kolom 'question' adalah input, dan 'answer' adalah referensi
input_texts = dataset['input']  # Gantilah dengan nama kolom yang sesuai
references = dataset['output']  # Jawaban dokter yang benar (referensi)

# Menyimpan hasil prediksi dan berbagai score
predictions = []
bleu_scores = []
meteor_scores = []
rouge_scores = []

# Loop untuk melakukan inferensi dan menghitung berbagai score
for input_text, reference in zip(input_texts, references):
    prediction = get_predictions(input_text)
    predictions.append(prediction)
    
    # Hitung BLEU score antara referensi dan prediksi
    bleu_score = calculate_bleu_score(reference, prediction)
    bleu_scores.append(bleu_score)
    
    # Hitung METEOR score antara referensi dan prediksi
    meteor = calculate_meteor_score(reference, prediction)
    meteor_scores.append(meteor)
    
    # Hitung ROUGE score antara referensi dan prediksi
    rouge = calculate_rouge_scores(reference, prediction)
    rouge_scores.append(rouge)
    
    # Output setiap hasil
    print(f"Input: {input_text}")
    print(f"Prediction: {prediction}")
    print(f"Reference: {reference}")
    print(f"BLEU Score: {bleu_score}")
    print(f"METEOR Score: {meteor}")
    print(f"ROUGE Scores: {rouge}")
    print('-' * 50)

# Menyimpan hasil ke file JSON dalam folder log_latihan
with open("log_latihan/predictions_and_bleu_scores.json", "w") as f_bleu:
    json.dump({"predictions": predictions, "bleu_scores": bleu_scores}, f_bleu)

with open("log_latihan/predictions_and_meteor_scores.json", "w") as f_meteor:
    json.dump({"predictions": predictions, "meteor_scores": meteor_scores}, f_meteor)

with open("log_latihan/predictions_and_rouge_scores.json", "w") as f_rouge:
    json.dump({"predictions": predictions, "rouge_scores": [str(rouge) for rouge in rouge_scores]}, f_rouge)

print("Hasil prediksi dan score sudah disimpan dalam folder 'log_latihan'.")