from datasets import load_dataset

# Memuat dataset (gantilah nama dataset sesuai kebutuhan)
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
print(dataset[1])