import pandas as pd
from tabulate import tabulate

def eksik_deger_tablosu(df):
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isnull().sum() / len(df)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
        columns={0: 'Eksik Değerler', 1: '% Değeri'})
    eksik_deger_tablo_son["Veri Tipi"] = df.dtypes
    
    # Benzersiz değer sayılarını ekleyin
    benzersiz_degerler = df.nunique()
    eksik_deger_tablo_son["Benzersiz Değerler"] = benzersiz_degerler
    
    return eksik_deger_tablo_son

eksik_deger_tablo = eksik_deger_tablosu(df_train)

# Tabloyu düzenli bir şekilde görüntüleyin
print(tabulate(eksik_deger_tablo, headers='keys', tablefmt='pretty'))
