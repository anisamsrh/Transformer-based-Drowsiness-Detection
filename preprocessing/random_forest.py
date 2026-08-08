import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold # <-- Solusi untuk masalahmu
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score, 
    cohen_kappa_score, 
    roc_auc_score
)

RANDOM_SEED = 42

def main():
    print("Memuat data tabular...")
    # Sesuaikan jika kamu menyimpannya sebagai .pkl di langkah sebelumnya
    data = np.load("data_ready/train_filter.npz")
    X_tab, y, groups = data["X_tab"], data["y"], data["id"] 

    bad_subjects = ['22020', '22056', '24020', '24038', '24051', '25116']
    # bad_subjects = ['22020', '24088', '24070', '24020', '23056', '23051', '22047', '22064', '23051'] 
    # bad_subjects = ['22064', '23051', '24088'] 

    # Buat mask untuk membuang subjek-subjek tersebut
    mask = ~np.isin(groups, bad_subjects)
    X_tab = X_tab[mask]
    y = y[mask]
    groups = groups[mask]
    
    print(f"Total Sampel setelah filtering subjek buruk: {len(X_tab)}")

    # Gunakan StratifiedGroupKFold (Bisa dicoba 5 fold)
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    all_y_true = []
    all_y_pred = []
    all_y_prob = [] # Untuk binary, kita hanya butuh probabilitas kelas positif (1)

    print(f"Total Sampel: {len(X_tab)} | Total Fitur Tabular: {X_tab.shape[1]}")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups)):
        # Menampilkan subjek siapa saja yang masuk ke dalam fold validasi ini
        val_subjects = np.unique(groups[val_idx])
        print(f"Fold {fold + 1} | Validation on Subjects: {val_subjects}")

        X_train, y_train = X_tab[train_idx], y[train_idx]
        X_val, y_val = X_tab[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Menggunakan Random Forest dengan class_weight='balanced'
        model = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        pred_class = model.predict(X_val_scaled)
        
        # Ambil probabilitas kelas 1 (Drowsy) untuk menghitung ROC-AUC binary
        pred_prob = model.predict_proba(X_val_scaled)[:, 1] 

        all_y_true.extend(y_val)
        all_y_pred.extend(pred_class)
        all_y_prob.extend(pred_prob)

        fold_acc = accuracy_score(y_val, pred_class)
        fold_bal_acc = balanced_accuracy_score(y_val, pred_class)
        print(f"  -> Akurasi: {fold_acc:.4f} | Balanced Acc: {fold_bal_acc:.4f}\n")

    # ==========================================================
    # EVALUASI GLOBAL
    # ==========================================================
    print("=" * 50)
    print(f"EVALUASI GLOBAL (STRATIFIED GROUP {n_splits}-FOLD)")
    print("=" * 50)

    # Disesuaikan menjadi 2 kelas
    target_names = ["Awake", "Drowsy"] 

    cm = confusion_matrix(all_y_true, all_y_pred)
    print("Confusion Matrix:\n", cm, "\n")

    cr = classification_report(all_y_true, all_y_pred, target_names=target_names, zero_division=0)
    print("Classification Report:\n", cr)

    global_bal_acc = balanced_accuracy_score(all_y_true, all_y_pred)
    global_macro_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    
    # Perhitungan ROC-AUC disesuaikan untuk Binary Classification
    roc_auc = roc_auc_score(all_y_true, all_y_prob)

    print("-" * 50)
    print(f"BALANCED ACCURACY : {global_bal_acc:.4f}")
    print(f"MACRO F1          : {global_macro_f1:.4f}")
    print(f"COHEN'S KAPPA     : {kappa:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print("\nTop 5 Fitur Paling Berpengaruh (Dari fold terakhir):")
        top_indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Fitur Indeks ke-{idx} (Score: {importances[idx]:.4f})")

if __name__ == "__main__":
    main()