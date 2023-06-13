from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

def evaluate_models_classification(X_train, y_train, X_test, y_test, metric='accuracy', random_state=42):
    models = [
        LGBMClassifier(random_state=random_state),
        XGBClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        SVC(random_state=random_state),
        KNeighborsClassifier(),
        MLPClassifier(random_state=random_state),
        RidgeClassifier(random_state=random_state),
        Lasso(),
        ElasticNet(random_state=random_state)
    ]
    print("Modeller yüklendi.")

    model_names = [
        'LGBMClassifier',
        'XGBClassifier',
        'GradientBoostingClassifier',
        'Random Forest Classifier',
        'AdaBoost Classifier',
        'Decision Tree Classifier',
        'SVC',
        'K-Nearest Neighbors Classifier',
        'Neural Network Classifier',
        'Ridge Classifier',
        'Lasso Classifier',
        'ElasticNet Classifier'
    ]
    print("Tahminler yapılıyor.")

    # Değerleri saklamak için bir veri çerçevesi oluşturma
    results = pd.DataFrame(columns=['Model', metric.upper()])

    predictions = []

    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred)
        elif metric == 'f1':
            score = f1_score(y_test, y_pred)
        elif metric == 'roc':
            score = roc_auc_score(y_test, y_pred)
        
        results = results.append({'Model': name, metric.upper(): score}, ignore_index=True)
        predictions.append(y_pred)

    # Sonuçları metriğe göre sıralama
    results = results.sort_values(by=metric.upper(), ascending=True)

    # Tabloyu görüntüleme
    print(results)

    print("Şimdi görselleştirme:")

    # Tahmin ve gerçek değerleri çizdirme
    for i in range(len(predictions)):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test))[:100], y_test[:100], label='Gerçek Değerler')
        plt.plot(np.arange(len(y_test))[:100], predictions[i][:100], label=model_names[i] + ' Tahminleri')

        plt.xlabel('Örnekler')
        plt.ylabel('Sınıf')
        plt.title('Gerçek ve Tahmin Edilen Değerler')
        plt.legend()
        plt.show()

# Fonksiyonu kullanarak istenen metrik değerlerini elde etme ve tahminleri çizdirme
evaluate_models_classification(X_train, y_train, X_test, y_test, metric='roc')
