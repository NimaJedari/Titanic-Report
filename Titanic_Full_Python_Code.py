# ============================================================
# Titanic Survival Analysis
# End-to-End Machine Learning Pipeline
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ─────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== Raw Dataset ===")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n=== Missing Values ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── 2. Data cleaning & imputation ────────────────────────────
# Age — mean imputation
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Cabin — binary flag
df['Has_Cabin'] = df['Cabin'].notna().astype(int)

# Embarked — mode imputation
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fare — log transformation to handle right skew and outliers
df['Fare_log'] = np.log1p(df['Fare'])

# Encode categorical variables
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])          # female=0, male=1
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])  # C=0, Q=1, S=2

# Drop columns not useful for modeling
df.drop(columns=['Cabin', 'Fare', 'Name', 'Ticket', 'PassengerId'], inplace=True)

print("\n=== Clean Dataset ===")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n=== Missing Values After Cleaning (should be 0) ===")
print(df.isnull().sum())

# ── 3. Exploratory data analysis ─────────────────────────────
save_path = os.path.expanduser('~/Desktop')

# Feature importance via Random Forest
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare_log', 'Has_Cabin', 'Embarked']
X = df[features]
y = df['Survived']

rf_eda = RandomForestClassifier(n_estimators=100, random_state=42)
rf_eda.fit(X, y)
importances = pd.Series(rf_eda.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#185FA5' if v >= 0.10 else '#85B7EB' for v in importances.values]
bars = ax.barh(importances.index, importances.values, color=colors)
ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Survival rate by categorical predictors
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Survival Rate by Categorical Predictors', fontsize=13, fontweight='bold')
cat_vars = [
    ('Pclass', {1: '1st', 2: '2nd', 3: '3rd'}),
    ('Sex', {0: 'Female', 1: 'Male'}),
    ('Has_Cabin', {0: 'No Cabin', 1: 'Has Cabin'}),
    ('Embarked', {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}),
]
for ax, (col, labels) in zip(axes, cat_vars):
    surv = df.groupby(col)['Survived'].mean()
    surv.index = [labels[i] for i in surv.index]
    bars = ax.bar(surv.index, surv.values, color='#185FA5', alpha=0.8)
    ax.set_title(col)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Survival Rate')
    for bar, val in zip(bars, surv.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'survival_categorical.png'), dpi=150, bbox_inches='tight')
plt.close()

# Age group survival
df['Age_group'] = pd.cut(df['Age'], bins=[0, 18, 50, 100], labels=['Young (0-18)', 'Mid age (19-50)', 'Old (51+)'])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Age Group vs Survival', fontsize=13, fontweight='bold')
surv = df.groupby('Age_group', observed=True)['Survived'].mean()
axes[0].bar(surv.index, surv.values, color='#185FA5', alpha=0.8)
axes[0].set_title('Survival Rate by Age Group')
axes[0].set_ylabel('Survival Rate')
axes[0].set_ylim(0, 1)
for i, (bar, val) in enumerate(zip(axes[0].patches, surv.values)):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
counts = df.groupby(['Age_group', 'Survived'], observed=True).size().unstack()
counts.plot(kind='bar', ax=axes[1], color=['#D3D1C7', '#185FA5'], alpha=0.85, width=0.6)
axes[1].set_title('Passenger Count by Age Group & Survival')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('')
axes[1].legend(['Did not survive', 'Survived'], loc='upper right')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'survival_age_groups.png'), dpi=150, bbox_inches='tight')
plt.close()

# Log Fare box plot
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle('Log Fare Distribution by Survival', fontsize=13, fontweight='bold')
data = [df[df['Survived'] == 0]['Fare_log'], df[df['Survived'] == 1]['Fare_log']]
bp = ax.boxplot(data, patch_artist=True, labels=['Did not survive', 'Survived'])
bp['boxes'][0].set_facecolor('#D3D1C7')
bp['boxes'][1].set_facecolor('#185FA5')
for patch in bp['boxes']:
    patch.set_alpha(0.7)
ax.set_ylabel('Log Fare')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'fare_log_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n=== EDA plots saved to Desktop ===")

# ── 4. Modeling ───────────────────────────────────────────────
df.drop(columns=['Age_group'], inplace=True)
X = df[features]
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_rf, X_test_rf, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression — all features
lr_all = LogisticRegression(random_state=42)
lr_all.fit(X_train, y_train)
acc_lr_all = accuracy_score(y_test, lr_all.predict(X_test))

# Model 2: Logistic Regression — Pclass, Sex, Cabin, Age
X2 = scaler.fit_transform(df[['Pclass', 'Sex', 'Has_Cabin', 'Age']])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
lr_4 = LogisticRegression(random_state=42)
lr_4.fit(X2_train, y2_train)
acc_lr_4 = accuracy_score(y2_test, lr_4.predict(X2_test))

# Model 3: Logistic Regression — Pclass, Sex, Cabin
X3 = scaler.fit_transform(df[['Pclass', 'Sex', 'Has_Cabin']])
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.2, random_state=42)
lr_3 = LogisticRegression(random_state=42)
lr_3.fit(X3_train, y3_train)
acc_lr_3 = accuracy_score(y3_test, lr_3.predict(X3_test))

# Model 4: Random Forest — all features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_rf))

# ── 5. Results ────────────────────────────────────────────────
print("\n=== Model Comparison ===")
print(f"  Logistic Regression (all features):         {acc_lr_all:.1%}  ← best")
print(f"  Logistic Regression (Pclass+Sex+Cabin+Age): {acc_lr_4:.1%}")
print(f"  Random Forest (all features):               {acc_rf:.1%}")
print(f"  Logistic Regression (Pclass+Sex+Cabin):     {acc_lr_3:.1%}")

print("\n=== Best Model Coefficients ===")
for feat, coef in sorted(zip(features, lr_all.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat}: {coef:.4f}")

print("\n=== Classification Report — Best Model ===")
print(classification_report(y_test, lr_all.predict(X_test)))