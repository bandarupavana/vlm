import matplotlib.pyplot as plt

combos = [
    "Transformer+CLIP",
    "Transformer+R2Gen",
    "Enc-Dec+CLIP",
    "Enc-Dec+R2Gen",
    "Attention+CLIP",
    "Attention+R2Gen"
]

rouge_l = [0.1818, 0.3, 0.2581, 0.1379, 0.3333, 0.2143]

plt.figure(figsize=(10,5))
plt.bar(combos, rouge_l)
plt.xticks(rotation=30)
plt.ylabel("ROUGE-L Score")
plt.title("Comparison of VLM Combinations (ROUGE-L)")

plt.show()