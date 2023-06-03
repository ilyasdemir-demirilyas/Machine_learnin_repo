import pandas as pd
def eksik_deger_tablosu(df):
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isnull().sum() / len(df)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
        columns={0: 'Eksik Değerler', 1: '% Değeri'})
    eksik_deger_tablo_son["Veri Tipi"] = df.dtypes
    return eksik_deger_tablo_son