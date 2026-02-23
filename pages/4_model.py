import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Sample confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a heatmap
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add annotations using numeric indices for x and y coordinates
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]}',
             horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()