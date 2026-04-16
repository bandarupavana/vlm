import matplotlib.pyplot as plt

classes = ["Adeno", "Large Cell", "Normal", "Squamous"]
f1_scores = [0.91, 0.88, 0.95, 0.90]  # example

plt.bar(classes, f1_scores)
plt.ylim(0, 1)
plt.ylabel("F1-score")
plt.title("Class-wise Performance of CNN")
plt.show()
