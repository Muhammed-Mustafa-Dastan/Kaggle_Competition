import pandas as pd

# Verileri oku
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# İlk 5 satırı göster
print('Train ilk 5 satır:')
print(train.head())
print('\nTest ilk 5 satır:')
print(test.head())

# Sütun isimleri
print('\nTrain sütunları:', train.columns.tolist())
print('Test sütunları:', test.columns.tolist())

# Eksik değer kontrolü
print('\nEksik değerler (train):')
print(train.isnull().sum())
print('\nEksik değerler (test):')
print(test.isnull().sum())

# Temel istatistikler
print('\nTrain istatistikleri:')
print(train.describe())
