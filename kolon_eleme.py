from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def find_best_feature_subset(X_train, X_test, y_train, y_test):
    original_columns = X_train.columns.tolist()
    
    # Tüm özelliklerle başlangıçta modeli eğit
    ada_model_sonuc = AdaBoostClassifier(random_state=42, learning_rate=0.1, n_estimators=200)
    ada_model_sonuc.fit(X_train, y_train)
    y_pred_sonuc = ada_model_sonuc.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_sonuc)
    best_column_to_remove = None

    for column_to_remove in original_columns:
        X_train_modified = X_train.drop(columns=[column_to_remove])
        X_test_modified = X_test.drop(columns=[column_to_remove])

        ada_model_sonuc = AdaBoostClassifier(random_state=42, learning_rate=0.1, n_estimators=200)
        ada_model_sonuc.fit(X_train_modified, y_train)

        y_pred_sonuc = ada_model_sonuc.predict(X_test_modified)

        accuracy = accuracy_score(y_test, y_pred_sonuc)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_column_to_remove = column_to_remove
            X_train = X_train_modified
            X_test = X_test_modified

    return best_column_to_remove, best_accuracy

# Veri setini girdi olarak al
best_column, best_accuracy = find_best_feature_subset(X_train, X_test, y_train, y_test)
print(f"En iyi sonucu veren kolon: {best_column}, En İyi Accuracy Skoru: {best_accuracy}")
