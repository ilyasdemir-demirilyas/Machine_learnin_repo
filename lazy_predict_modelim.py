import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

def evaluate_models(X_train, y_train, X_test, y_test, metric='mae',random_state=42):
    models = [
        LGBMRegressor(random_state=random_state),
        XGBRegressor(random_state=random_state),
        GradientBoostingRegressor(random_state=random_state),
        RandomForestRegressor(random_state=random_state),
        AdaBoostRegressor(random_state=random_state),
        DecisionTreeRegressor(random_state=random_state),
        SVR(),
        KNeighborsRegressor(),
        MLPRegressor(random_state=random_state),
        Ridge(random_state=random_state),
        Lasso(random_state=random_state),
        ElasticNet(random_state=random_state)
    ]
    print("modeller yüklendi .")

    model_names = [
        'LGBMRegressor',
        'XGBRegressor',
        'GradientBoostingRegressor',
        'Random Forest Regressor',
        'AdaBoost Regressor',
        'Decision Tree Regressor',
        'SVR',
        'K-Nearest Neighbors Regressor',
        'Neural Network Regressor',
        'Ridge Regression',
        'Lasso Regression',
        'ElasticNet Regression'
    ]
    print("tahminler yapılıyor .")
    # Değerleri saklamak için bir veri çerçevesi oluşturma
    results = pd.DataFrame(columns=['Model', metric.upper()])

    predictions = []

    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        results = results.append({'Model': name, metric.upper(): score}, ignore_index=True)
        predictions.append(y_pred)

    # Sonuçları metriğe göre sıralama
    results = results.sort_values(by=metric.upper(), ascending=True)

    # Tabloyu görüntüleme
    print(results)
    
    print("şimdide görselleştirme :")

    # Tahmin ve gerçek değerleri çizdirme
    for i in range(len(predictions)):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test))[:100], y_test[:100], label='Gerçek Değerler')
        plt.plot(np.arange(len(y_test))[:100], predictions[i][:100], label=model_names[i] + ' Tahminleri')

        plt.xlabel('Örnekler')
        plt.ylabel('Age')
        plt.title('Gerçek ve Tahmin Edilen Değerler')
        plt.legend()
        plt.show()

# Fonksiyonu kullanarak MAE değerlerini elde etme ve tahminleri çizdirme
evaluate_models(X_train, y_train, X_test, y_test, metric='mae')
