"""
Model Building Challenge
================================================
This script builds a model that predicts the binary response variable "target" influenced by:
1. Account quality 
2. A seasonal effect
3. The combined macroeconomic effects over time

Author: Angel Thu Do
Date: November 2025
"""

"""
model.py
Usage:
  python model.py --data-dir ./Test2 --out-dir ./results
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Set style and random seed
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 100)
print("MODEL BUILDING CHALLENGE")
print("=" * 100)

# ============================================================================
# PARAMETERS
# ============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Multi-Model Binary Classification')
parser.add_argument('--data-dir', type=str, default='./Test2', 
                    help='Directory containing CSV files (default: ./Test2)')
parser.add_argument('--out-dir', type=str, default='./results',
                    help='Output directory for results (default: ./results)')
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# 1. DATA LOADING AND MERGING
# ============================================================================

def load_and_merge_data(file_paths):
    """Load multiple CSV files and merge them into a single dataset."""
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"\nMerged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df

print("\n[1] Loading and merging data...")
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if len(all_files) == 0:
    print(f"ERROR: No CSV files found in {DATA_DIR}")
    print(f"Please ensure all 6 CSV files are in the directory.")
    exit(1)

print(f"Found {len(all_files)} CSV files")
data = load_and_merge_data(all_files)
data['Date'] = pd.to_datetime(data['Date'])
data['Date_ym'] = pd.to_datetime(data['Date_ym'])

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create temporal and seasonal features."""
    print("\n[2] Features engineering...")
    
    df = df.copy()
    
    # Temporal features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Seasonal features (Jan/Feb effect)
    df['is_jan_feb'] = df['month'].isin([1, 2]).astype(int)
    df['is_jan'] = (df['month'] == 1).astype(int)
    df['is_feb'] = (df['month'] == 2).astype(int)
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Time trend
    min_date = df['Date_ym'].min()
    df['months_since_start'] = ((df['Date_ym'].dt.year - min_date.year) * 12 + 
                                 (df['Date_ym'].dt.month - min_date.month))
    
    print(f"  ✓ Total features: {len(df.columns)}")
    return df

data = engineer_features(data)

# ============================================================================
# 3. TRAIN/VALIDATION/TEST SPLIT (BUILD/HOLD-OUT SET)
# ============================================================================

print("\n[3] Splitting data...")

# Build set (Date_ym < "2022-05")
build_df = data[data['Date_ym'] < '2022-05'].copy().sort_values('Date_ym').reset_index(drop=True)

# Holdout set (Date_ym >= "2022-05")
holdout_df = data[data['Date_ym'] >= '2022-05'].copy().reset_index(drop=True)

print(f"  Build set: {len(build_df):,} rows ({build_df['Date_ym'].min()} to {build_df['Date_ym'].max()})")
print(f"  Holdout set: {len(holdout_df):,} rows ({holdout_df['Date_ym'].min()} to {holdout_df['Date_ym'].max()})")

# Split build into train (80%) and validation (20%)
split_idx = int(len(build_df) * 0.8)
train_df = build_df.iloc[:split_idx].copy()
val_df = build_df.iloc[split_idx:].copy()

print(f"  Train set: {len(train_df):,} rows")
print(f"  Validation set: {len(val_df):,} rows")

# Feature columns
predictor_cols = [f'P{i}' for i in range(1, 20)]
temporal_features = ['year', 'month', 'quarter', 'day_of_year',
                     'is_jan_feb', 'is_jan', 'is_feb',
                     'month_sin', 'month_cos', 'months_since_start']
feature_cols = predictor_cols + temporal_features

print(f"  Features used: {len(feature_cols)}")

# Prepare arrays
X_train = train_df[feature_cols].values
y_train = train_df['target'].values
X_val = val_df[feature_cols].values
y_val = val_df['target'].values
X_holdout = holdout_df[feature_cols].values
y_holdout = holdout_df['target'].values

# ============================================================================
# 4. MODEL DEFINITIONS (5 MODELS: LG, RF, HGB, LGBM, XGB)
# ============================================================================

def get_models():
    """Define all 5 models."""
    print("\n[4] Defining models...")
    
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'name': 'Logistic Regression',
            'scale': True,
            'color': '#E63946'
        },
        'Random_Forest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'name': 'Random Forest',
            'scale': False,
            'color': '#457B9D'
        },
        'HistGradientBoosting': {
            'model': HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42
            ),
            'name': 'HistGradientBoosting',
            'scale': False,
            'color': '#F4A261'
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'name': 'LightGBM',
            'scale': False,
            'color': '#2A9D8F'
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'name': 'XGBoost',
            'scale': False,
            'color': '#6A4C93'
        }
    }
    
    print(f"  ✓ Total models: {len(models)}")
    return models

models_config = get_models()

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\n[5] Training all models...")

results = {}
trained_models = {}
scalers = {}

for model_key, config in models_config.items():
    print(f"\n  Training {config['name']}...")
    
    model = config['model']
    
    # Scale if needed
    if config['scale']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        scalers[model_key] = scaler
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        scalers[model_key] = None
    
    # Train model
    if model_key == 'LightGBM':
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
    elif model_key == 'XGBoost':
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_val = model.predict_proba(X_val_scaled)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_pred_train)
    val_auc = roc_auc_score(y_val, y_pred_val)
    train_acc = accuracy_score(y_train, (y_pred_train >= 0.5).astype(int))
    val_acc = accuracy_score(y_val, (y_pred_val >= 0.5).astype(int))
    
    results[model_key] = {
        'name': config['name'],
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'color': config['color']
    }
    
    trained_models[model_key] = model
    
    print(f"    Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    print(f"    Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save model
    model_path = os.path.join(OUT_DIR, f"{model_key}_model.pkl")
    joblib.dump({'model': model, 'scaler': scalers[model_key]}, model_path)
    print(f"    ✓ Saved: {model_path}")

# ============================================================================
# 6. BEST MODEL SELECTION
# ============================================================================

print("\n[6] Selecting best model based on validation AUC...")

results_df = pd.DataFrame(results).T
best_model_key = results_df['val_auc'].idxmax()
best_model_name = results_df.loc[best_model_key, 'name']
best_val_auc = results_df.loc[best_model_key, 'val_auc']

print(f"  ✓ Best model: {best_model_name}")
print(f"  ✓ Validation AUC: {best_val_auc:.4f}")

# Save best model separately
best_model_path = os.path.join(OUT_DIR, "best_model.pkl")
joblib.dump({
    'model': trained_models[best_model_key],
    'scaler': scalers[best_model_key],
    'model_name': best_model_name,
    'model_key': best_model_key
}, best_model_path)
print(f"  ✓ Saved best model: {best_model_path}")

# ============================================================================
# 7. HOLDOUT SET EVALUATION
# ============================================================================

print("\n[7] Evaluating all models on holdout set...")

holdout_results = {}

for model_key, model in trained_models.items():
    scaler = scalers[model_key]
    
    # Scale if needed
    X_holdout_scaled = scaler.transform(X_holdout) if scaler else X_holdout
    
    # Predictions
    y_pred_holdout = model.predict_proba(X_holdout_scaled)[:, 1]
    y_pred_class = (y_pred_holdout >= 0.5).astype(int)
    
    # Metrics
    holdout_auc = roc_auc_score(y_holdout, y_pred_holdout)
    holdout_acc = accuracy_score(y_holdout, y_pred_class)
    
    holdout_results[model_key] = {
        'holdout_auc': holdout_auc,
        'holdout_accuracy': holdout_acc,
        'predictions': y_pred_holdout,
        'predictions_class': y_pred_class
    }
    
    print(f"  {results[model_key]['name']}: AUC={holdout_auc:.4f}, Acc={holdout_acc:.4f}")

# ============================================================================
# 8. METRICS TABLE
# ============================================================================

print("\n[8] Creating metrics table...")

metrics_data = []
for model_key in results.keys():
    metrics_data.append({
        'Model': results[model_key]['name'],
        'Train_AUC': results[model_key]['train_auc'],
        'Val_AUC': results[model_key]['val_auc'],
        'Holdout_AUC': holdout_results[model_key]['holdout_auc'],
        'Train_Accuracy': results[model_key]['train_accuracy'],
        'Val_Accuracy': results[model_key]['val_accuracy'],
        'Holdout_Accuracy': holdout_results[model_key]['holdout_accuracy']
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values('Val_AUC', ascending=False)

# Save metrics
metrics_path = os.path.join(OUT_DIR, "model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"  ✓ Saved: {metrics_path}")

print("\n" + "=" * 100)
print("MODEL PERFORMANCE METRICS")
print("=" * 100)
print(metrics_df.to_string(index=False))
print("=" * 100)

# ============================================================================
# 9. TEMPORAL EVALUATION (BY DATE_YM)
# ============================================================================

print("\n[9] Evaluating performance over time (Date_ym)...")

# Use best model for temporal evaluation
best_model = trained_models[best_model_key]
best_scaler = scalers[best_model_key]

holdout_df['pred_proba'] = holdout_results[best_model_key]['predictions']
holdout_df['pred_class'] = holdout_results[best_model_key]['predictions_class']

# By Date_ym
time_metrics = []
for date_ym in sorted(holdout_df['Date_ym'].unique()):
    subset = holdout_df[holdout_df['Date_ym'] == date_ym]
    if len(subset) > 0 and subset['target'].nunique() > 1:
        auc = roc_auc_score(subset['target'], subset['pred_proba'])
        acc = accuracy_score(subset['target'], subset['pred_class'])
        time_metrics.append({
            'Date_ym': date_ym,
            'AUC': auc,
            'Accuracy': acc,
            'N': len(subset)
        })

time_df = pd.DataFrame(time_metrics)
time_path = os.path.join(OUT_DIR, "performance_by_time.csv")
time_df.to_csv(time_path, index=False)
print(f"  ✓ Saved: {time_path}")
print(f"  Mean AUC over time: {time_df['AUC'].mean():.4f}")
print(f"  Std AUC over time: {time_df['AUC'].std():.4f}")

# ============================================================================
# 10. SEGMENT × DATE_YM EVALUATION
# ============================================================================

print("\n[10] Evaluating performance by Segment × Date_ym...")

segment_time_metrics = []
for segment in sorted(holdout_df['S'].unique()):
    for date_ym in sorted(holdout_df['Date_ym'].unique()):
        subset = holdout_df[(holdout_df['S'] == segment) & (holdout_df['Date_ym'] == date_ym)]
        if len(subset) > 0 and subset['target'].nunique() > 1:
            auc = roc_auc_score(subset['target'], subset['pred_proba'])
            acc = accuracy_score(subset['target'], subset['pred_class'])
            segment_time_metrics.append({
                'Segment': segment,
                'Date_ym': date_ym,
                'AUC': auc,
                'Accuracy': acc,
                'N': len(subset)
            })

segment_time_df = pd.DataFrame(segment_time_metrics)
segment_time_path = os.path.join(OUT_DIR, "performance_by_segment_time.csv")
segment_time_df.to_csv(segment_time_path, index=False)
print(f"  ✓ Saved: {segment_time_path}")

# Summary by segment
segment_summary = segment_time_df.groupby('Segment')['AUC'].agg(['mean', 'std', 'min', 'max'])
print("\n  Performance by Segment:")
print(segment_summary)

# ============================================================================
# 11. INDIVIDUAL PLOTS
# ============================================================================

print("\n[11] Creating individual plots...")

# Plot 1: Model Comparison - AUC
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(metrics_df))
width = 0.25
ax.bar(x_pos - width, metrics_df['Train_AUC'], width, label='Train AUC', alpha=0.8)
ax.bar(x_pos, metrics_df['Val_AUC'], width, label='Val AUC', alpha=0.8)
ax.bar(x_pos + width, metrics_df['Holdout_AUC'], width, label='Holdout AUC', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('AUC')
ax.set_title('Model Comparison - AUC Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_01_model_comparison_auc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_01_model_comparison_auc.png")

# Plot 2: Model Comparison - Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_pos - width, metrics_df['Train_Accuracy'], width, label='Train Accuracy', alpha=0.8)
ax.bar(x_pos, metrics_df['Val_Accuracy'], width, label='Val Accuracy', alpha=0.8)
ax.bar(x_pos + width, metrics_df['Holdout_Accuracy'], width, label='Holdout Accuracy', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison - Accuracy Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_02_model_comparison_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_02_model_comparison_accuracy.png")

# Plot 3: AUC Over Time (Best Model)
from matplotlib.dates import MonthLocator, DateFormatter

fig, ax = plt.subplots(figsize=(12, 6))
time_df['Date_ym_dt'] = pd.to_datetime(time_df['Date_ym'])
ax.plot(time_df['Date_ym_dt'], time_df['AUC'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.axhline(y=time_df['AUC'].mean(), color='r', linestyle='--', linewidth=2, 
           label=f"Mean AUC: {time_df['AUC'].mean():.3f}")
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.set_xlabel('Date')
ax.set_ylabel('AUC')
ax.set_title(f'AUC Over Time - {best_model_name} (Holdout Set)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_03_auc_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_03_auc_over_time.png")

# Plot 4: AUC by Segment × Time (Heatmap)
segment_time_df['Date_ym'] = pd.to_datetime(segment_time_df['Date_ym']).dt.strftime('%Y-%m')
fig, ax = plt.subplots(figsize=(14, 6)) 
pivot_auc = segment_time_df.pivot(index='Segment', columns='Date_ym', values='AUC') 
sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', center=0.7,  
            ax=ax, cbar_kws={'label': 'AUC'}) 
ax.set_title(f'AUC by Segment × Date - {best_model_name} (Holdout Set)', fontsize=14, fontweight='bold') 
ax.set_xlabel('Date') 
ax.set_ylabel('Segment')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_04_auc_segment_time_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_04_auc_segment_time_heatmap.png")

# Plot 5: ROC Curves - All Models
fig, ax = plt.subplots(figsize=(10, 8))
for model_key in trained_models.keys():
    y_pred = holdout_results[model_key]['predictions']
    fpr, tpr, _ = roc_curve(y_holdout, y_pred)
    auc_score = holdout_results[model_key]['holdout_auc']
    ax.plot(fpr, tpr, linewidth=2, 
            label=f"{results[model_key]['name']} (AUC={auc_score:.3f})",
            color=results[model_key]['color'])
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models (Holdout Set)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_05_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_05_roc_curves.png")

# Plot 6: Precision-Recall Curves - All Models
fig, ax = plt.subplots(figsize=(10, 8))
for model_key in trained_models.keys():
    y_pred = holdout_results[model_key]['predictions']
    precision, recall, _ = precision_recall_curve(y_holdout, y_pred)
    ax.plot(recall, precision, linewidth=2,
            label=results[model_key]['name'],
            color=results[model_key]['color'])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves - All Models (Holdout Set)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_06_precision_recall_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_06_precision_recall_curves.png")

# Plot 7: Accuracy Over Time
from matplotlib.dates import MonthLocator, DateFormatter

fig, ax = plt.subplots(figsize=(12, 6))
time_df['Date_ym_dt'] = pd.to_datetime(time_df['Date_ym'])
ax.plot(time_df['Date_ym_dt'], time_df['Accuracy'], marker='s', linewidth=2, 
        markersize=8, color='#E76F51')
ax.margins(x=0.02)
ax.axhline(y=time_df['Accuracy'].mean(), color='r', linestyle='--', linewidth=2,
           label=f"Mean Accuracy: {time_df['Accuracy'].mean():.3f}")
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.set_xlabel('Date')
ax.set_ylabel('Accuracy')
ax.set_title(f'Accuracy Over Time - {best_model_name} (Holdout Set)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_07_accuracy_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_07_accuracy_over_time.png")

# Plot 8: Accuracy by Segment × Time (Heatmap)
segment_time_df['Date_ym'] = pd.to_datetime(segment_time_df['Date_ym']).dt.strftime('%Y-%m')
fig, ax = plt.subplots(figsize=(14, 6))
pivot_acc = segment_time_df.pivot(index='Segment', columns='Date_ym', values='Accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', center=0.7,
            ax=ax, cbar_kws={'label': 'Accuracy'})
ax.set_title(f'Accuracy by Segment × Date - {best_model_name} (Holdout Set)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Segment')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_08_accuracy_segment_time_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plot_08_accuracy_segment_time_heatmap.png")

# ============================================================================
# 12. DASHBOARD
# ============================================================================

print("\n[12] Creating dashboard...")

fig = plt.figure(figsize=(20, 12))

# Subplot 1: Model Comparison AUC
ax1 = plt.subplot(3, 3, 1)
x_pos = np.arange(len(metrics_df))
width = 0.25
ax1.bar(x_pos - width, metrics_df['Train_AUC'], width, label='Train', alpha=0.8, color='skyblue')
ax1.bar(x_pos, metrics_df['Val_AUC'], width, label='Val', alpha=0.8, color='orange')
ax1.bar(x_pos + width, metrics_df['Holdout_AUC'], width, label='Holdout', alpha=0.8, color='green')
ax1.set_ylabel('AUC')
ax1.set_title('Model AUC Comparison', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics_df['Model'], rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Model Comparison Accuracy
ax2 = plt.subplot(3, 3, 2)
ax2.bar(x_pos - width, metrics_df['Train_Accuracy'], width, label='Train', alpha=0.8, color='skyblue')
ax2.bar(x_pos, metrics_df['Val_Accuracy'], width, label='Val', alpha=0.8, color='orange')
ax2.bar(x_pos + width, metrics_df['Holdout_Accuracy'], width, label='Holdout', alpha=0.8, color='green')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy Comparison', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_df['Model'], rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: AUC Over Time
ax3 = plt.subplot(3, 3, 3)
time_df['Date_ym_dt'] = pd.to_datetime(time_df['Date_ym'])
ax3.plot(time_df['Date_ym_dt'], time_df['AUC'], marker='o', linewidth=2, color='#2E86AB')
ax3.axhline(y=time_df['AUC'].mean(), color='r', linestyle='--', linewidth=1)
ax3.xaxis.set_major_locator(MonthLocator())
ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax3.set_ylabel('AUC')
ax3.set_title(f'AUC Over Time ({best_model_name})', fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

# Subplot 4: Accuracy Over Time
ax4 = plt.subplot(3, 3, 4)
time_df['Date_ym_dt'] = pd.to_datetime(time_df['Date_ym'])
ax4.plot(time_df['Date_ym_dt'], time_df['Accuracy'], marker='s', linewidth=2, color='#E76F51')
ax4.axhline(y=time_df['Accuracy'].mean(), color='r', linestyle='--', linewidth=1)
ax4.xaxis.set_major_locator(MonthLocator())
ax4.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax4.set_ylabel('Accuracy')
ax4.set_title(f'Accuracy Over Time ({best_model_name})', fontweight='bold', fontsize=10)
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

# Subplot 5: ROC Curves
ax5 = plt.subplot(3, 3, 5)
for model_key in trained_models.keys():
    y_pred = holdout_results[model_key]['predictions']
    fpr, tpr, _ = roc_curve(y_holdout, y_pred)
    ax5.plot(fpr, tpr, linewidth=2, label=results[model_key]['name'],
            color=results[model_key]['color'])
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax5.set_xlabel('FPR')
ax5.set_ylabel('TPR')
ax5.set_title('ROC Curves', fontweight='bold')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# Subplot 6: Precision-Recall Curves
ax6 = plt.subplot(3, 3, 6)
for model_key in trained_models.keys():
    y_pred = holdout_results[model_key]['predictions']
    precision, recall, _ = precision_recall_curve(y_holdout, y_pred)
    ax6.plot(recall, precision, linewidth=2, label=results[model_key]['name'],
            color=results[model_key]['color'])
ax6.set_xlabel('Recall')
ax6.set_ylabel('Precision')
ax6.set_title('Precision-Recall Curves', fontweight='bold')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# Subplot 7: AUC by Segment × Time Heatmap
ax7 = plt.subplot(3, 3, 7)
segment_time_df['Date_ym'] = pd.to_datetime(segment_time_df['Date_ym']).dt.strftime('%Y-%m')
pivot_auc = segment_time_df.pivot(index='Segment', columns='Date_ym', values='AUC')
sns.heatmap(pivot_auc, annot=True, fmt='.2f', cmap='RdYlGn', center=0.7,
            ax=ax7, cbar_kws={'label': 'AUC'}, annot_kws={'fontsize': 7})
ax7.set_title('AUC by Segment × Date', fontweight='bold', fontsize=10)
ax7.set_xlabel('Date', fontsize=8)
ax7.set_ylabel('Segment', fontsize=8)
plt.setp(ax7.xaxis.get_majorticklabels(), fontsize=7)
plt.setp(ax7.yaxis.get_majorticklabels(), fontsize=7)

# Subplot 8: Accuracy by Segment × Time Heatmap
ax8 = plt.subplot(3, 3, 8)
segment_time_df['Date_ym'] = pd.to_datetime(segment_time_df['Date_ym']).dt.strftime('%Y-%m')
pivot_acc = segment_time_df.pivot(index='Segment', columns='Date_ym', values='Accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.2f', cmap='RdYlGn', center=0.7,
            ax=ax8, cbar_kws={'label': 'Accuracy'}, annot_kws={'fontsize': 7})
ax8.set_title('Accuracy by Segment × Date', fontweight='bold', fontsize=10)
ax8.set_xlabel('Date', fontsize=8)
ax8.set_ylabel('Segment', fontsize=8)
plt.setp(ax8.xaxis.get_majorticklabels(), fontsize=7)
plt.setp(ax8.yaxis.get_majorticklabels(), fontsize=7)

# Subplot 9: Best Model Metrics Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
best_metrics = metrics_df[metrics_df['Model'] == best_model_name].iloc[0]
summary_text = f"""
BEST MODEL: {best_model_name}

Train Metrics:
  AUC: {best_metrics['Train_AUC']:.4f}
  Accuracy: {best_metrics['Train_Accuracy']:.4f}

Validation Metrics:
  AUC: {best_metrics['Val_AUC']:.4f}
  Accuracy: {best_metrics['Val_Accuracy']:.4f}

Holdout Metrics:
  AUC: {best_metrics['Holdout_AUC']:.4f}
  Accuracy: {best_metrics['Holdout_Accuracy']:.4f}

Temporal Stability:
  Mean AUC: {time_df['AUC'].mean():.4f}
  Std AUC: {time_df['AUC'].std():.4f}
"""
ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')

plt.suptitle('MODEL EVALUATION DASHBOARD', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(os.path.join(OUT_DIR, 'model_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: model_dashboard.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("MODEL BUILDING COMPLETE!")
print("=" * 100)

print(f"\nBest Model: {best_model_name}")
print(f"  Validation AUC: {best_metrics['Val_AUC']:.4f}")
print(f"  Holdout AUC: {best_metrics['Holdout_AUC']:.4f}")
print(f"  Temporal Stability (std): {time_df['AUC'].std():.4f}")

print("\nAll Model Performance (Holdout AUC):")
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']}: {row['Holdout_AUC']:.4f}")

print(f"\nOutputs saved to: {OUT_DIR}/")
print("\nGenerated files:")
print("  Models:")
for model_key in trained_models.keys():
    print(f"    - {model_key}_model.pkl")
print(f"    - best_model.pkl ({best_model_name})")
print("\n  Metrics:")
print("    - model_metrics.csv")
print("    - performance_by_time.csv")
print("    - performance_by_segment_time.csv")
print("\n  Plots:")
for i in range(1, 9):
    print(f"    - plot_{i:02d}_*.png")
print("    - model_dashboard.png")

print("\n" + "=" * 100)
print("SUCCESS! All models trained and evaluated.")
print("=" * 100)