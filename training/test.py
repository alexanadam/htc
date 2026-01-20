import configs.config as config
from utils.helper import get_hierarchy
from sklearn.preprocessing import MultiLabelBinarizer
from training.metrics import hierarchical_evaluation


hierarchy = get_hierarchy(config.FORC_TAXONOMY)

target = [
    ["animal", "cat"],  # Second instance: True label is 'animal' and 'cat'
    ["animal", "dog"],  # Fourth instance: True label is 'animal' and 'siamese'
]

predict = [
    ["animal", "cat"],  # Prediction is correct
    ["dog"],  # Prediction is slightly wrong (we expected 'bulldog' here)
]

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

all_classes = [
    "animal",
    "dog",
    "cat",
]
mlb.fit([all_classes])
target_one_hot = mlb.transform(target)
predict_one_hot = mlb.transform(predict)

prec, rec, f1 = hierarchical_evaluation(
    target_one_hot, predict_one_hot, mlb, hierarchy, average="micro"
)

print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
