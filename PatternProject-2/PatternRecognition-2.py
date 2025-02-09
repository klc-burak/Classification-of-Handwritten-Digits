import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score, f1_score

# "data" dosyasının okunması
data = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_data.csv")

# "data_test" dosyasının okunması
data_test = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_data_test.csv")

# "data_train" dosyasının okunması
data_train = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_data_train.csv")

# "target" dosyasının okunması
target = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_target.csv")
series_target = pd.Series(target['column_0'])

# "target_test" dosyasının okunması
target_test = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_target_test.csv")
series_target_test = pd.Series(target_test['column_0'])

# "target_train" dosyasının okunması
target_train = pd.read_csv("C:/Users/Burak/Desktop/nrippner-mnist-handwritten-digits/nrippner-mnist-handwritten-digits/MNIST_target_train.csv")
series_target_train = pd.Series(target_train['column_0'])

# Decision Tree algoritmasının tanımlanması
model_DT = DecisionTreeClassifier()

# En iyi 100 özniteliğin seçimi için Decision Tree'nin kullanılması
selector = SelectFromModel(estimator=model_DT, max_features=100)
selected_features = selector.fit_transform(data, target)

# Cross-Validation Accuracy değerinin hesaplanması ve yazdırılması
cv_results = cross_val_score(model_DT, selected_features, target, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_results.mean())

# Modelin eğitim verileriyle eğitilmesi
model_DT.fit(data_train, series_target_train)

# Test seti üzerinden tahmin üretilmesi
predictions = model_DT.predict(data_test)

# Classification Accuracy değerinin hesaplanması ve yazdırılması
accuracy = accuracy_score(series_target_test, predictions)
print("Classification Accuracy:", accuracy)

# Confusion Matrix'in oluşturulması
conf_matrix = confusion_matrix(series_target_test, predictions)
TN = conf_matrix[0][0]  # True Negative
FP = conf_matrix[0][1]  # False Positive
FN = conf_matrix[1][0]  # False Negative
TP = conf_matrix[1][1]  # True Positive

# Sensitivity ve Specificity değerlerinin hesaplanması
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Değerlerin yazdırılması
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# AUC değerinin hesaplanması ve yazdırılması
probabilities = model_DT.predict_proba(data_test)
auc = roc_auc_score(series_target_test, probabilities, multi_class='ovr')
print("AUC:", auc)

# Precision değerinin hesaplanması ve yazdırılması
precision = precision_score(series_target_test, predictions, average='weighted')
print("Precision:", precision)

# F1 değerinin hesaplanması ve yazdırılması
f1 = f1_score(series_target_test, predictions, average='weighted')
print("F1 Score:", f1)