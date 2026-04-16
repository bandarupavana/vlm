import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Example values (replace with your actual y_true & y_pred)
cm = np.array([
    [28, 1, 0, 1],
    [2, 25, 1, 0],
    [0, 1, 30, 0],
    [1, 0, 2, 27]
])

class_names = [
    "Adenocarcinoma",
    "Large Cell",
    "Normal",
    "Squamous Cell"
]

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.show()
