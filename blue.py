import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from datasets import load_dataset
import subprocess
import json
import os

# Memastikan NLTK siap untuk digunakan
nltk.download('punkt')
nltk.download('wordnet')

# Membuat folder log_latihan jika belum ada
if not os.path.exists("log_latihan"):
    os.makedirs("log_latihan")

# Fungsi untuk melakukan inferensi menggunakan model yang sudah di-load di Ollama
def get_predictions(input_text):
    # Jalankan perintah ollama untuk melakukan inferensi dengan model GGUF
    command = f"ollama run llama-opsi '{input_text}'"
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        output = result.stdout.decode('utf-8').strip()
        if result.returncode != 0:
            print(f"Error: {result.stderr.decode('utf-8')}")
        return output
    except subprocess.TimeoutExpired:
        print("Timeout: Model tidak bisa menyelesaikan inferensi dalam waktu yang ditentukan.")
        return "Error: Timeout during inference"

# Fungsi untuk menghitung BLEU Score dengan smoothing
def calculate_bleu_score(reference, prediction):
    reference_tokenized = nltk.word_tokenize(reference)
    prediction_tokenized = nltk.word_tokenize(prediction)
    smoothie = SmoothingFunction().method4  # Menggunakan smoothing agar skor BLEU tidak terlalu rendah
    return sentence_bleu([reference_tokenized], prediction_tokenized, smoothing_function=smoothie)

# Fungsi untuk menghitung METEOR Score
def calculate_meteor_score(reference, prediction):
    reference_tokenized = nltk.word_tokenize(reference)
    prediction_tokenized = nltk.word_tokenize(prediction)
    return meteor_score([reference_tokenized], prediction_tokenized)

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

    # Simpan hasil ke file JSON secara bertahap setelah setiap inferensi
    with open("log_latihan2/predictions_and_scores.json", "w") as f:
        json.dump({
            "predictions": predictions,
            "bleu_scores": bleu_scores,
            "meteor_scores": meteor_scores,
            "rouge_scores": [{key: value.fmeasure for key, value in rouge.items()} for rouge in rouge_scores]
        }, f)

print("Hasil prediksi dan score sudah disimpan dalam folder 'log_latihan'.")
