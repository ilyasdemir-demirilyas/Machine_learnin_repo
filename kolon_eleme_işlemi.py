from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def find_best_columns(X_train, X_test, y_train, y_test):
    # Başlangıç modelini oluştur ve eğit
    ada_model_sonuc = AdaBoostClassifier(random_state=42, learning_rate=0.1, n_estimators=200)
    ada_model_sonuc.fit(X_train, y_train)
    
    # Başlangıç modelinin performansını değerlendir
    y_pred_sonuc = ada_model_sonuc.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_sonuc)
    
    # Başlangıçta en iyi sonuç ve çıkarılacak kolonun saklanacağı değişkenleri tanımla
    best_column_to_remove_list = []
    
    # Tüm özelliklerin isimlerini al
    original_columns = X_train.columns.tolist()
    
    # Her bir özelliği tek tek çıkararak en iyi sonucu bulmaya çalış
    for _ in range(len(original_columns)):
        best_accuracy_for_iteration = 0  # Her adımda en iyi sonuç sıfırla
        best_column_for_iteration = None   # Her adımda en iyi sonuc veren kolonu sıfırla
        
        for column_to_remove in original_columns:
            # Belirli kolonu çıkartarak yeni veri seti oluştur
            X_train_modified = X_train.drop(columns=[column_to_remove])
            X_test_modified = X_test.drop(columns=[column_to_remove])
            
            # AdaBoostClassifier modelini oluştur ve eğit
            ada_model_sonuc = AdaBoostClassifier(random_state=42, learning_rate=0.1, n_estimators=200)
            ada_model_sonuc.fit(X_train_modified, y_train)
            
            # Test verisi üzerinde tahmin yap
            y_pred_sonuc = ada_model_sonuc.predict(X_test_modified)
            
            # Model performansını değerlendir
            accuracy = accuracy_score(y_test, y_pred_sonuc)
            
            # Eğer bulunan sonuç daha iyi ise güncelle
            if accuracy > best_accuracy_for_iteration:
                best_accuracy_for_iteration = accuracy
                best_column_for_iteration = column_to_remove
        
        # En iyi sonuç ve çıkarılacak kolonu sakla
        if best_accuracy_for_iteration > best_accuracy:
            best_accuracy = best_accuracy_for_iteration
            best_column_to_remove_list.append(best_column_for_iteration)
            
            # Modeldeki en iyi performansı elde etmek için veri setini güncelle
            X_train = X_train.drop(columns=[best_column_for_iteration])
            X_test = X_test.drop(columns=[best_column_for_iteration])
            original_columns.remove(best_column_for_iteration)
            
        else:
            break  # Sonuçlar artık daha iyi değilse döngüyü sonlandır
    
    return best_column_to_remove_list, best_accuracy

# Fonksiyonu kullanarak en iyi sonuç veren kolonları bul
best_columns, best_accuracy = find_best_columns(X_train, X_test, y_train, y_test)
print("En iyi sonuç veren kolonlar: ", best_columns)
print("En iyi Accuracy Skoru: ", best_accuracy)
