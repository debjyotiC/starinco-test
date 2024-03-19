import numpy as np
from os import listdir
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load('test-results/george_test_task.npz', allow_pickle=True)
predicted_label, actual_label = data['out_x'], data['out_y']

classes_values = listdir("dataset/george_test_task")


label_predicted = np.argmax(predicted_label, axis=1)
label_actual = np.argmax(actual_label, axis=1)

conf_matrix = confusion_matrix(label_actual, label_predicted).ravel()
acc = accuracy_score(label_actual, label_predicted)
report = classification_report(label_actual, label_predicted, output_dict=True)
results = confusion_matrix(label_actual, label_predicted)
precision = report['weighted avg']['precision']
sensitivity = report['weighted avg']['recall']


print(f"Accuracy Score: {acc}")
print(f"F1: {report['weighted avg']['f1-score']}")
print(f"Precision: {precision}")
print(f"Sensitivity: {sensitivity}")

ax = plt.subplot()
sns.heatmap(results, annot=True, annot_kws={"size": 20}, ax=ax, fmt='g')
ax.set_xticks(range(len(classes_values)))
ax.set_yticks(range(len(classes_values)))
# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=12)
ax.set_ylabel('True labels', fontsize=12)
ax.set_title(f'Confusion Matrix for thermal image classifier with accuracy {round(acc, 2)}')
ax.xaxis.set_ticklabels(classes_values, fontsize=10, rotation=45)
ax.yaxis.set_ticklabels(classes_values, fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png')
plt.show()
