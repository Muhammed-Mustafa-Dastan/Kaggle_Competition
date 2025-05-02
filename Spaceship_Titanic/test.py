# Spaceship Titanic – EDA + Feature-Engineering Rationale
# --------------------------------------------------------
# A runnable notebook skeleton that *both* shows the raw EDA
# visuals and explicitly demonstrates **why** every engineered
# feature in `engineer()` was created.
# Copy this into a Kaggle notebook and run top-to-bottom.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

DATA_PATH = '/data'
train = pd.read_csv(f'{DATA_PATH}/train.csv')
test  = pd.read_csv(f'{DATA_PATH}/test.csv')

# ---------------- Feature-engineering helper ----------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Booking group -----------------------------------------------------
    grp = df['PassengerId'].str.split('_', expand=True).astype(int)
    df['GroupID'], df['PaxNum'] = grp[0], grp[1]
    df['GroupSize'] = df.groupby('GroupID')['PassengerId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # 2. Cabin split -------------------------------------------------------
    cab = df['Cabin'].str.split('/', expand=True)
    df['Deck'], df['CabinNum'], df['Side'] = cab[0], cab[1], cab[2]

    # 3. Spending & Cryo consistency --------------------------------------
    global SPEND_COLS
    SPEND_COLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0)
    df['TotalSpend'] = df[SPEND_COLS].sum(axis=1)

    mask = df['CryoSleep'].isna()
    df.loc[mask & (df['TotalSpend'] == 0), 'CryoSleep'] = True
    df.loc[mask & (df['TotalSpend'] > 0),  'CryoSleep'] = False
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['CryoMismatch'] = ((df['CryoSleep'] == 1) & (df['TotalSpend'] > 0)).astype(int)

    # 4. Age buckets -------------------------------------------------------
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeGroup'] = pd.cut(df['Age'], [-1,12,18,40,65,200],
                            labels=['Child','Teen','Adult','Middle','Senior'])

    # 5. Categorical cleanup ----------------------------------------------
    for col in ['HomePlanet','Destination','Deck','Side','AgeGroup']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    return df

# Combine train+test for EDA context
train['set'] = 'train'; test['set'] = 'test'
full_raw = pd.concat([train, test], ignore_index=True)
full_fe  = engineer(full_raw)

# ------------------------------------------------------------
# SECTION 1: Missing-Value Heatmap (raw vs engineered)
# ------------------------------------------------------------
plt.figure(figsize=(10,4))
sns.heatmap(full_raw.isna(), cbar=False)
plt.title('Raw Dataset – Missing Value Matrix'); plt.show()

plt.figure(figsize=(10,4))
sns.heatmap(full_fe.isna(), cbar=False)
plt.title('After Feature Engineering – Missing Value Matrix'); plt.show()

# ------------------------------------------------------------
# SECTION 2: Booking Group Insights
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=full_fe, x='GroupSize')
plt.title('Distribution of GroupSize'); plt.show()

print('\nTransported ratio by IsAlone (train only):')
print(train.assign(IsAlone=full_fe.loc[train.index,'IsAlone'])
          .groupby('IsAlone')['Transported'].mean())

# Plot IsAlone vs Transported
plt.figure(figsize=(6,4))
sns.countplot(data=train.assign(IsAlone=full_fe.loc[train.index,'IsAlone']),
              x='IsAlone', hue='Transported')
plt.title('IsAlone vs Transported'); plt.show()

# ------------------------------------------------------------
# SECTION 3: Cabin Decomposition
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=train.assign(Deck=full_fe.loc[train.index,'Deck']),
              x='Deck', hue='Transported')
plt.title('Deck vs Transported'); plt.show()

# ------------------------------------------------------------
# SECTION 4: Spending vs CryoSleep
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(data=train.assign(TotalSpend=full_fe.loc[train.index,'TotalSpend'],
                              CryoSleep=full_fe.loc[train.index,'CryoSleep']),
            x='CryoSleep', y='TotalSpend')
plt.title('TotalSpend by CryoSleep'); plt.show()

print('\nCryoMismatch count (anomaly flag):', full_fe['CryoMismatch'].sum())

# ------------------------------------------------------------
# SECTION 4A: TotalSpend vs Transported
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(data=train.assign(TotalSpend=full_fe.loc[train.index,'TotalSpend']),
            x='Transported', y='TotalSpend')
plt.title('TotalSpend vs Transported'); plt.show()

# ------------------------------------------------------------
# SECTION 4B: **Individual Spending Distributions**
# ------------------------------------------------------------
for col in SPEND_COLS:
    plt.figure(figsize=(6,3))
    sns.histplot(train.assign(**{col: full_fe.loc[train.index,col]}), x=col, bins=40, kde=True)
    plt.xlim(0, 800)
    plt.title(f'{col} Distribution (train)'); plt.show()

    plt.figure(figsize=(4,3))
    sns.barplot(x='Transported', y=col, data=train.assign(**{col: full_fe.loc[train.index,col]}), ci=None)
    plt.title(f'{col} mean by Transported'); plt.show()

# ------------------------------------------------------------
# SECTION 5: Age Buckets
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=train.assign(AgeGroup=full_fe.loc[train.index,'AgeGroup']),
              x='AgeGroup', hue='Transported')
plt.title('AgeGroup vs Transported'); plt.show()

# ------------------------------------------------------------
# SECTION 6: Unique Values for Categorical Features
# ------------------------------------------------------------
cat_features = ['HomePlanet','Destination','Deck','Side','AgeGroup','VIP','CryoSleep','IsAlone']
print('\nUnique value counts for core categorical columns:')
for col in cat_features:
    uniqs = full_fe[col].unique()
    print(f"{col:<12} → {len(uniqs)} unique | sample: {uniqs[:10]}")

# ------------------------------------------------------------
# SECTION 7: Correlation Heatmap for Numeric Features
# ------------------------------------------------------------
num_cols = ['Age','GroupSize','PaxNum','TotalSpend','CryoMismatch']
plt.figure(figsize=(5,4))
sns.heatmap(full_fe[num_cols].corr(), annot=True)
plt.title('Numeric Correlations'); plt.show()

# ------------------------------------------------------------
# SECTION 8: Rationale Summary
# ------------------------------------------------------------
print("\n***Feature-Engineering Rationale Summary***")
print("1. GroupID/GroupSize/IsAlone capture party dynamics which strongly affect transport odds.")
print("2. Deck & Side add spatial information once Cabin is split.")
print("3. TotalSpend (sum of five service columns) plus CryoMismatch encode spending behaviour & data errors.")
print("4. AgeGroup provides non-linear age effects; raw Age is retained for numeric models.")
print("5. Categorical mode fills + VIP binary ensure no NA leakage.")