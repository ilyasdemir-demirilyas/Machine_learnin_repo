import pandas as pd
from sklearn.preprocessing import LabelEncoder

def perform_encoding(data, encoding_dict):
    """
    Veri setindeki belirtilen kategorik değişkenleri belirtilen encoding yöntemleri ile dönüştüren fonksiyon.

    Parametreler:
        - data: pd.DataFrame - Kategorik değerlerin bulunduğu veri seti
        - encoding_dict: dict - Encoding yöntemlerini ve ilgili değişkenleri içeren sözlük. 
          Anahtarlar encoding türlerini ("label encode", "One-Hot Encoding") temsil eder, 
          değerler ise ilgili değişkenleri liste olarak içerir.

    Çıktı:
        - pd.DataFrame - Encoding işlemi tamamlanmış veri seti
    """
    encoded_data = data.copy() # Veri setini kopyala

    # Label Encoding
    if "label encode" in encoding_dict:
        label_encoded_vars = encoding_dict["label encode"]
        for var in label_encoded_vars:
            label_encoder = LabelEncoder()
            encoded_data[var] = label_encoder.fit_transform(encoded_data[var])

    # One-Hot Encoding
    if "One-Hot Encoding" in encoding_dict:
        one_hot_encoded_vars = encoding_dict["One-Hot Encoding"]
        for var in one_hot_encoded_vars:
            dummy_vars = pd.get_dummies(encoded_data[var], prefix=var)
            encoded_data = pd.concat([encoded_data, dummy_vars], axis=1)
            encoded_data.drop(var, axis=1, inplace=True)

    return encoded_data

# example 
encoding_dict = { "label encode" : ["Gender", "Channel" ,"Discount"], "One-Hot Encoding" : ["PaymentType", "Category" ,"Location"] }
encoded_data = perform_encoding(data, encoding_dict) # Encoding işlemini uygula
 # Encoding işlemi sonrası ilk 5 satırı görüntüle