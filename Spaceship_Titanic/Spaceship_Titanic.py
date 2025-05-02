# spaceship_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf

print("TensorFlow:", tf.__version__)
print("TFDF:", tfdf.__version__)

# Veri Yükleme
train_df = pd.read_csv("data/train.csv")
print("Eğitim verisi boyutu:", train_df.shape)
print(train_df.head())

# Hedef değişkenin dağılımı
train_df["Transported"].value_counts().plot(kind="bar")
plt.title("Transported Dağılımı")
plt.show()
