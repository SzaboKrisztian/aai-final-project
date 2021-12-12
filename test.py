from sklearn.metrics import classification_report
from itertools import product

classes = ['A', 'B', 'C']
truth = ['C', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'A', 'A']
predictions = ['A', 'C', 'A', 'B', 'A', 'B', 'C', 'A', 'B', 'A']

conf_matrix = {}
for index in range(len(predictions)):
    pred = predictions[index]
    act = truth[index]
    score = conf_matrix.get((pred, act), 0)
    conf_matrix[(pred, act)] = score + 1

accs = []
for country in classes:
    tp = conf_matrix.get((country, country), 0)
    fn = sum(list(map(lambda item : conf_matrix.get((item, country), 0) if country != item else 0, classes)))
    fp = sum(list(map(lambda item : conf_matrix.get((country, item), 0) if country != item else 0, classes)))
    tn = sum(list(map(lambda item : conf_matrix.get((item[0], item[1]), 0) if item[0] != country and item[1] != country else 0, product(classes, classes))))

    acc = 0 if tp + tn + fn + fp == 0 else (tp + tn) / (tp + tn + fn + fp)
    accs.append(acc)

print(classification_report(truth, predictions, zero_division=0))
print(sum(accs) / len(accs))