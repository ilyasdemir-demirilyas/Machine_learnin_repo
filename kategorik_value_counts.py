import pandas as pd

def kategorik_value_counts(data):
    """
    Veri setindeki kategorik değerlerin her biri için value_counts() fonksiyonunu kullanarak sayıları döndüren fonksiyon.

    Parametreler:
        - data: pd.DataFrame - Kategorik değerlerin bulunduğu veri seti

    Çıktı:
        - dict - Her bir kategorik değer için sayıları içeren sözlük
    """
    kategorik_degiskenler = data.select_dtypes(include=['object', 'category']) # Kategorik değerleri seç
    value_counts_dict = {} # Sonuçları saklayacağımız sözlük

    # Her bir kategorik değer için value_counts() fonksiyonunu kullanarak sayıları hesapla
    for col in kategorik_degiskenler.columns:
        value_counts_dict[col] = kategorik_degiskenler[col].value_counts().to_dict()

    # Çıktıyı daha güzel görüntüleme
    for col, value_counts in value_counts_dict.items():
        print(f"Kategori: {col}")
        print(f"Sayılar: {value_counts}")
        print("-----------")

kategorik_value_counts(ilk_9_ay)