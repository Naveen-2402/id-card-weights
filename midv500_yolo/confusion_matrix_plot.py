# confusion_matrix_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load CSV file
df = pd.read_csv('results.csv')

# Check columns
assert 'true_label' in df.columns and 'pred_label' in df.columns, \
    "CSV must contain 'true_label' and 'pred_label' columns."

# Extract true and predicted labels
y_true = df['true_label']
y_pred = df['pred_label']

# Create confusion matrix
labels = sorted(list(set(y_true) | set(y_pred)))  # union of all classes
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # Save to file
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels))
