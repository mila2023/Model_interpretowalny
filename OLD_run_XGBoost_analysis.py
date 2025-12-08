import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import random

from sklearn.calibration import calibration_curve
from sklearn import metrics
from sklearn.metrics import brier_score_loss, accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer 
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

shap.initjs()

class PKDKodWoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10, smoothing=0.5):
        self.top_n = top_n
        self.smoothing = smoothing
    
    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        
        self.top_values_ = X['pkdKod'].value_counts().nlargest(self.top_n).index
        
        grouped = X['pkdKod'].where(X['pkdKod'].isin(self.top_values_), other='0')
        
        df = pd.DataFrame({'group': grouped, 'target': y})
        agg = df.groupby('group')['target'].agg(['sum', 'count'])
        agg = agg.rename(columns={'sum':'bad', 'count':'total'})
        agg['good'] = agg['total'] - agg['bad']

        agg['bad_s'] = agg['bad'] + self.smoothing
        agg['good_s'] = agg['good'] + self.smoothing

        total_bad = agg['bad_s'].sum()
        total_good = agg['good_s'].sum()

        agg['woe'] = np.log((agg['good_s'] / total_good) / (agg['bad_s'] / total_bad))

        self.woe_map_ = agg['woe'].to_dict()
        self.fallback_ = agg.loc['0', 'woe'] if '0' in agg.index else np.mean(list(self.woe_map_.values()))
        
        return self
    
    def transform(self, X):
        X = X.copy()
        grouped = X['pkdKod'].where(X['pkdKod'].isin(self.top_values_), other='0')
        X['WoE_pkdKod_grouped'] = grouped.map(self.woe_map_).fillna(self.fallback_)
        return X.drop(columns=['pkdKod'])


class DropConstantColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cols_to_drop_ = [col for col in X.columns if X[col].nunique() <= 1]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
             raise ValueError("input_features nie może być None dla DropConstantColumns")
        return [col for col in input_features if col not in self.cols_to_drop_]

class MissingValueIndicatorAndImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan).copy()
        self.base_cols_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X[self.base_cols_])

        self.indicator_cols_ = [f"{c}_mial_braki_danych" for c in self.base_cols_]

        return self

    def transform(self, X):
        X = X.replace([np.inf, -np.inf], np.nan).copy()

        X_imputed = pd.DataFrame(
            self.imputer_.transform(X[self.base_cols_]),
            columns=self.base_cols_,
            index=X.index
        )

        indicator_df = X[self.base_cols_].isna().astype(int)
        indicator_df.columns = self.indicator_cols_
        indicator_df.index = X.index

        X_out = pd.concat([X_imputed, indicator_df], axis=1)

        return X_out
    
    def get_feature_names_out(self, input_features=None):
            base_cols = self.base_cols_
            indicator_cols = self.indicator_cols_
            return base_cols + indicator_cols

data = pd.read_csv("zbiór_10.csv")

# Definicja cech X i y
X = data.drop(columns=["default"])
y = data["default"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

print("Fitting WoE Encoder...")
woe_encoder = PKDKodWoEEncoder(top_n=10, smoothing=0.5)

woe_encoder.fit(X_train, y_train)

X_train_woe = woe_encoder.transform(X_train)
X_val_woe = woe_encoder.transform(X_val)
X_test_woe = woe_encoder.transform(X_test)

print("WoE encoding complete.")

categorical_features = [
    'formaWlasnosci_Symbol', 
    'schemat_wsk_bilans', 
    'schemat_wsk_rzis'
]

numerical_features = [
    col for col in X_train_woe.columns if col not in categorical_features
]


numeric_transformer = Pipeline(steps=[
    ('missing', MissingValueIndicatorAndImputer(strategy="median")),
    ('drop_constant', DropConstantColumns()),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.05,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() 
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

X_val_processed = pipeline['preprocessor'].fit(X_train_woe).transform(X_val_woe)

fit_params = {
    'model__eval_set': [(X_val_processed, y_val)],
    'model__verbose' : 100
}

pipeline.fit(X_train_woe, y_train, **fit_params)

joblib.dump(pipeline, 'XGBoost_model_pipeline.pkl')

print("Pipeline training complete.")
print("Model was successfully exported to file 'XGBoost_model_pipeline.pkl'.")

preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

try:
    final_feature_names = preprocessor.get_feature_names_out()
    print(f"Pobrano {len(final_feature_names)} finalnych nazw cech.")
except Exception as e:
    print(f"Błąd przy get_feature_names_out: {e}")

X_test_transformed_np = preprocessor.transform(X_test_woe)

X_test_transformed_df = pd.DataFrame(
    X_test_transformed_np, 
    columns=final_feature_names, 
    index=X_test_woe.index
)

print("Dane dla SHAP przygotowane.")

explainer = shap.TreeExplainer(model)

print("Obliczanie wartości SHAP...")
shap_values_all_classes = explainer(X_test_transformed_df)
print("Wartości SHAP obliczone.")

shap_values = shap_values_all_classes
baseline = float(shap_values_all_classes.base_values[0])

print(f"Baseline (wartość bazowa log-odds): {baseline:.4f}")

idx = 0
obserwacja_idx = X_test_transformed_df.index[idx]
przetworzone_dane_obs = X_test_transformed_df.iloc[[idx]]

shap_sum = float(shap_values.values[idx].sum())

raw_prediction = model.predict(przetworzone_dane_obs, output_margin=True)[0]

print("\nWeryfikacja dla pierwszej obserwacji testowej:")
print("=" * 60)
print(f"Baseline (wartość bazowa):         {baseline:.4f}")
print(f"Suma wartości SHAP:                {shap_sum:+.4f}")
print(f"Baseline + Suma SHAP:              {baseline + shap_sum:.4f}")
print(f"Surowa predykcja modelu (log-odds): {raw_prediction:.4f}")

if abs((baseline + shap_sum) - raw_prediction) < 1e-5:
    print("\nWeryfikacja: Wyjaśnienia są spójne z modelem.")
else:
    print("\nWeryfikacja: Wyjaśnienia NIE są spójne z modelem.")

print("Globalna ważność cech (Beeswarm):")

shap.summary_plot(shap_values, X_test_transformed_df)

CECHA = 'num__wsk_poziom_kapitalu_obrotowego_netto'

print(f"Wykres zależności dla: {CECHA}")

shap.dependence_plot(
    CECHA,
    shap_values.values,
    X_test_transformed_df,
    interaction_index="auto"
)

CECHA = 'num__Kapital_wlasny'

print(f"Wykres zależności dla: {CECHA}")

shap.dependence_plot(
    CECHA,
    shap_values.values,
    X_test_transformed_df,
    interaction_index="auto"
)

CECHA = 'num__zysk_netto'

print(f"Wykres zależności dla: {CECHA}")

shap.dependence_plot(
    CECHA,
    shap_values.values,
    X_test_transformed_df,
    interaction_index="auto"
)

CECHA = 'num__wsk_cykl_konwersji_gotowki'

print(f"Wykres zależności dla: {CECHA}")

shap.dependence_plot(
    CECHA,
    shap_values.values,
    X_test_transformed_df,
    interaction_index="auto"
)

CECHA = 'num__wsk_sprzedaz_kap_obrotowy'

print(f"Wykres zależności dla: {CECHA}")

shap.dependence_plot(
    CECHA,
    shap_values.values,
    X_test_transformed_df,
    interaction_index="auto"
)

y_proba = pipeline.predict_proba(X_test_woe)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba, pos_label=1)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall vs. Threshold')
plt.legend()
plt.grid()
plt.show()

f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

best_f1_index = np.argmax(f1_scores)

best_threshold = thresholds[best_f1_index]
best_f1_score = f1_scores[best_f1_index]

print(f"\n--- Optimal Threshold Finder ---")
print(f"Best F1-Score: {best_f1_score:.4f}")
print(f"Found at Threshold: {best_threshold:.4f}")

y_pred_best = (y_proba >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_best)
print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Best Iteration: {pipeline.named_steps['model'].best_iteration}")
print(f"Best Score (Validation AUC): {pipeline.named_steps['model'].best_score:.4f}")

print(f"\n--- Classification Report (Threshold = {best_threshold:.4f}) ---")
print(classification_report(y_test, y_pred_best, target_names=['No Default (0)', 'Default (1)']))

print("--- Confusion Matrix (Threshold = {best_threshold:.4f}) ---")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: No Default', 'Predicted: Default'], 
            yticklabels=['Actual: No Default', 'Actual: Default'])
plt.title(f'Confusion Matrix (Threshold = {best_threshold:.4f})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
plt.show()

fn_indices = X_test_woe[(y_pred_best == 0) & (y_test == 1)].index

if not fn_indices.empty:

    num_to_check = min(3, len(fn_indices))

    positions_to_check = random.sample(range(len(fn_indices)), num_to_check)

    for i in positions_to_check:
    
        idx_to_explain = fn_indices[i] 
        
        idx_position = X_test_transformed_df.index.get_loc(idx_to_explain)

        print(f"\n--- Wyjaśnienie dla obserwacji FALSE NEGATIVE (Indeks: {idx_to_explain}) ---")
        
        shap.plots.waterfall(shap_values[idx_position])
    
else:
    print("Gratulacje, brak pomyłek False Negative do analizy.")

fn_indices = X_test_woe[(y_pred_best == 0) & (y_test == 1)].index

if not fn_indices.empty:
    print(f"Analizuję {len(fn_indices)} przypadków False Negative...")
    
    fn_positions = X_test_transformed_df.index.get_indexer(fn_indices)
    
    fn_positions = fn_positions[fn_positions != -1]

    print("Generuję wykres 'summary_plot' tylko dla przypadków FN:")
    
    shap.summary_plot(
        shap_values[fn_positions], 
        X_test_transformed_df.iloc[fn_positions]
    )
    
else:
    print("Brak pomyłek False Negative do analizy.")

def expected_calibration_error(y_true, p, n_bins=10):

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.any():
            conf = p[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.sum()/len(p)) * abs(acc - conf)
    return ece

def reliability_plot(y_true, p, title):

    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(7, 6))
    plt.plot([0,1],[0,1], '--', label='Perfekcyjna kalibracja')
    plt.plot(mean_pred, frac_pos, marker='o', label='Model XGBoost')
    plt.title(title)
    plt.xlabel("Średnia prognozowana PD (Pewność)")
    plt.ylabel("Rzeczywista częstość zdarzeń (Celność)")
    plt.legend()
    plt.grid(True)
    plt.show()

def hist_predictions(p, title):

    plt.figure(figsize=(7, 5))
    plt.hist(p, bins=30, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel("Prognozowane Prawdopodobieństwo (PD)")
    plt.ylabel("Liczność obserwacji")
    plt.grid(axis='y')
    plt.show()


print("---  DIAGNOSTYKA MODELU (PRE-CALIBRATION) ---")

reliability_plot(y_test, y_proba, "Krzywa Reliability — Model XGBoost (Test, pre-cal)")

hist_predictions(y_proba, "Histogram Predykcji — Model XGBoost (Test, pre-cal)")

ece_pre = expected_calibration_error(y_test, y_proba, n_bins=10)
brier_pre = brier_score_loss(y_test, y_proba)

print(f"\nWyniki (pre-cal):")
print(f"  Oczekiwany Błąd Kalibracji (ECE): {ece_pre:.4f}")
print(f"  Wynik Briera (Brier Score):       {brier_pre:.4f}")

def fit_platt(y, p):
    eps = 1e-12
    logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    lr = LogisticRegression(max_iter=500, C=1e6) 
    lr.fit(logit.reshape(-1,1), y)
    return lr

def apply_platt(lr, p):
    eps = 1e-12
    logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    return lr.predict_proba(logit.reshape(-1,1))[:,1]

def fit_isotonic(y, p):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(p, y)
    return ir

def apply_isotonic(ir, p):
    return ir.transform(p)


print("Obliczanie prognoz dla zbioru walidacyjnego...")
y_proba_val = pipeline.predict_proba(X_val_woe)[:, 1]

y_proba_test_pre_cal = y_proba 

print("Dopasowywanie kalibratorów na zbiorze walidacyjnym...")
platt_model = fit_platt(y_val, y_proba_val)
iso_model = fit_isotonic(y_val, y_proba_val)
print("Kalibratory wytrenowane.")

print("Stosowanie kalibratorów na zbiorze testowym...")
y_proba_test_platt = apply_platt(platt_model, y_proba_test_pre_cal)
y_proba_test_iso   = apply_isotonic(iso_model, y_proba_test_pre_cal)

print("\n--- OCENA (POST-CALIBRATION) ---")

calibrated_probas = {
    'Platt': y_proba_test_platt,
    'Isotonic': y_proba_test_iso
}

for name, p in calibrated_probas.items():
    print(f"\n--- Model: {name} (post-cal) ---")
            
    reliability_plot(y_test, p, f"Reliability — {name} (test, post-cal)")
    hist_predictions(p, f"Histogram predykcji — {name} (test, post-cal)")
    
    ece_post = expected_calibration_error(y_test, p, n_bins=10)
    brier_post = brier_score_loss(y_test, p)
    
    print(f"Wyniki dla {name} (post-cal):")
    print(f"  Oczekiwany Błąd Kalibracji (ECE): {ece_post:.4f}")
    print(f"  Wynik Briera (Brier Score):       {brier_post:.4f}")

def logit_fn(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def inv_logit(z):
    return 1/(1+np.exp(-z))

def shift_to_target_mean(p, target_mean=0.04, tol=1e-6, max_iter=100):
    
    z = logit_fn(p)
    
    lo, hi = -10.0, 10.0
    
    for _ in range(max_iter):
        mid = (lo+hi)/2
        m = inv_logit(z + mid).mean()
        
        if abs(m - target_mean) < tol:
            return inv_logit(z + mid)
        
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    
    print("Ostrzeżenie: Osiągnięto limit iteracji w wyszukiwaniu.")
    return inv_logit(z + (lo+hi)/2)


proba_iso_pre_shift = y_proba_test_iso

print(f"Średnia prognoza (Isotonic, przed shiftem): {proba_iso_pre_shift.mean():.4f}")

target_pd_mean = 0.04
print(f"Rozpoczynam dostrajanie do średniej = {target_pd_mean}...")
proba_iso_4pct = shift_to_target_mean(proba_iso_pre_shift, target_mean=target_pd_mean)

print("\n--- OCENA (POST-SHIFT 4%) ---")
print(f"Średnia po dostrojeniu: {proba_iso_4pct.mean():.4f} (Cel: {target_pd_mean})")

brier_4pct = brier_score_loss(y_test, proba_iso_4pct)
ece_4pct = expected_calibration_error(y_test, proba_iso_4pct)

print(f"Brier (post-shift): {brier_4pct:.4f}")
print(f"ECE (post-shift):   {ece_4pct:.4f}")

reliability_plot(y_test, proba_iso_4pct, f"Reliability — Isotonic + Shift {target_pd_mean*100}% (test)")
hist_predictions(proba_iso_4pct, f"Histogram — Isotonic + Shift {target_pd_mean*100}% (test)")



print("\n--- Obliczanie Statystyki KS ---")

p_data = proba_iso_4pct 
y_true_data = y_test

fpr, tpr, thresholds = metrics.roc_curve(y_true_data, p_data)

ks_statistic = np.max(tpr - fpr)

ks_threshold = thresholds[np.argmax(tpr - fpr)]

print(f"Statystyka KS (Kolmogorov-Smirnov): {ks_statistic:.4f}")
print(f"Próg, przy którym osiągnięto KS: {ks_threshold:.4f}")

COST_TP = 0.0
COST_FP = 1.0
COST_FN = 18.0
COST_TN = -1.0

def cost_for_threshold(y_true, p, thr):
    yhat = (p >= thr).astype(int)
    
    tp = np.sum((yhat==1) & (y_true==1))
    fp = np.sum((yhat==1) & (y_true==0))
    fn = np.sum((yhat==0) & (y_true==1))
    tn = np.sum((yhat==0) & (y_true==0))
    
    total_cost = tp*COST_TP + fp*COST_FP + fn*COST_FN + tn*COST_TN
    return total_cost, tp, fp, fn, tn

def sweep_costs(y_true, p, n=201):
    thrs = np.linspace(0,1,n)
    costs, details = [], []
    for t in thrs:
        c, tp, fp, fn, tn = cost_for_threshold(y_true, p, t)
        costs.append(c); details.append((tp,fp,fn,tn))
    return thrs, np.array(costs), details

y_true_data = y_test
p_data = proba_iso_4pct 

print(f"Rozpoczynam analizę kosztów dla {len(p_data)} obserwacji...")

thrs, costs, details = sweep_costs(y_true_data, p_data, n=201)

best_idx = int(np.argmin(costs))
best_thr_cost = float(thrs[best_idx])
best_cost = costs[best_idx]
best_tp, best_fp, best_fn, best_tn = details[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(thrs, costs, label='Całkowity koszt biznesowy')
plt.axvline(x=best_thr_cost, color='red', linestyle='--', 
            label=f'Optymalny próg: {best_thr_cost:.4f}\n(Min. koszt: {best_cost:.2f})')
plt.title("Krzywa kosztu vs próg decyzyjny")
plt.xlabel("Próg decyzyjny (Odmów, jeśli PD >= Próg)")
plt.ylabel("Całkowity koszt (Im niżej, tym lepiej)")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Wyniki Optymalnego Progu Biznesowego ---")
print(f"Optymalny próg (minimalizujący koszt): {best_thr_cost:.4f}")
print(f"Minimalny osiągnięty koszt: {best_cost:.2f}")
print("\nMacierz pomyłek dla tego progu:")
print(f"  Prawdziwie Pozytywni (TP - Odmówiono złym): {best_tp}")
print(f"  Fałszywie Pozytywni (FP - Odmówiono dobrym): {best_fp}  (Koszt: {best_fp * COST_FP})")
print(f"  Fałszywie Negatywni (FN - Udzielono złym):  {best_fn}  (Koszt: {best_fn * COST_FN})")
print(f"  Prawdziwie Negatywni (TN - Udzielono dobrym): {best_tn}  (Zysk: {best_tn * COST_TN})")

accept_rate = (best_fn + best_tn) / len(y_true_data)
print(f"\nStopa akceptacji (udzielono kredytu): {accept_rate:.2%}")

rating_bins = [0.00, 0.045, 0.15, 1.01]

rating_labels = [
    "A (Akceptacja)", 
    "B (Akceptacja lub analiza)", 
    "C (Odrzucenie)"
]

def pd_to_rating(p, bins, labels):
    return pd.cut(p, bins=bins, labels=labels, right=False, include_lowest=True)


final_pd = proba_iso_4pct

ratings = pd_to_rating(final_pd, rating_bins, rating_labels)

print("--- Liczność klientów w każdej klasie ratingowej ---")
tab_licznosci = pd.crosstab(ratings, columns="Liczność klientów")
print(tab_licznosci)


validation_df = pd.DataFrame({
    'Rating': ratings,
    'Predicted_PD': final_pd,
    'Actual_Default': y_test
})

rating_summary = validation_df.groupby('Rating').agg(
    Liczność=('Rating', 'count'),
    Średnie_Prognozowane_PD=('Predicted_PD', 'mean'),
    Rzeczywisty_Odsetek_Default=('Actual_Default', 'mean')
)

print("\n--- Walidacja Monotoniczności Ratingów ---")
print(rating_summary)

plt.figure(figsize=(10, 6))
rating_summary['Rzeczywisty_Odsetek_Default'].plot(kind='bar', color='salmon')
plt.title("Walidacja: Rzeczywisty % Defaultu vs Klasa Ratingowa")
plt.xlabel("Klasa Ratingowa")
plt.ylabel("Rzeczywisty Odsetek Defaultu (im wyżej, tym gorzej)")
plt.grid(axis='y')
plt.show()

decision_table = rating_summary[['Średnie_Prognozowane_PD', 'Rzeczywisty_Odsetek_Default']].copy()

decision_table['Sugerowana Decyzja Biznesowa'] = [
    "Akceptacja Automatyczna", 
    "Odrzucenie (lub Analiza Manualna)",
    "Odrzucenie Automatyczne"
]

print("\n--- Finalna Tabela Decyzyjna / Mapa Ratingowa ---")

decision_table['Średnie_Prognozowane_PD'] = decision_table['Średnie_Prognozowane_PD'].map('{:.2%}'.format)
decision_table['Rzeczywisty_Odsetek_Default'] = decision_table['Rzeczywisty_Odsetek_Default'].map('{:.2%}'.format)

print(decision_table.to_markdown(numalign="left", stralign="left"))


