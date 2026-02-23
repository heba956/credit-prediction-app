import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample true labels and predicted labels
true_labels = [0, 1, 1, 0, 2, 2, 1, 0]
predicted_labels = [0, 1, 0, 0, 2, 1, 1, 0]

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1, 2],
            yticklabels=[0, 1, 2])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()