import matplotlib.pyplot as plt

combos = [
    "Transformer+CLIP",
    "Transformer+R2Gen",
    "Enc-Dec+CLIP",
    "Enc-Dec+R2Gen",
    "Attention+CLIP",
    "Attention+R2Gen"
]

rouge_l = [0.24, 0.1935, 0.1081, 0.1538, 0.25, 0.1951]  # example

plt.figure(figsize=(10,5))
plt.plot(combos, rouge_l, marker="o")
plt.xticks(rotation=30)
plt.ylabel("ROUGE-L Score")
plt.title("Comparison of VLM Combinations")
plt.grid(True)
plt.show()
