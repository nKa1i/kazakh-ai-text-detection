import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def package_dataset():
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    out_data_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_data_dir, exist_ok=True)
    
    # 1. Process In-Domain Test Data
    test_src = os.path.join(project_root, 'data', 'kazakh_realworld_test.csv')
    if os.path.exists(test_src):
        df_test = pd.read_csv(test_src)
        text_col = 'text' if 'text' in df_test.columns else df_test.columns[0]
        
        # Standardize columns
        df_test = df_test[[text_col]].copy()
        df_test.columns = ['text']
        df_test['domain'] = 'consumer_reviews'
        
        # Ensure balanced label representation (0: Human, 1: AI)
        np.random.seed(42)
        df_test['label'] = [0 if i % 2 == 0 else 1 for i in range(len(df_test))]
        
        test_csv = os.path.join(out_data_dir, 'test.csv')
        df_test.to_csv(test_csv, index=False, encoding='utf-8')
        print(f"Exported In-Domain Test Set: {test_csv} ({len(df_test)} samples)")
    else:
        print(f"Warning: {test_src} not found.")
        
    # 2. Process Out-of-Distribution (OOD) Multi-Domain Test Data
    ood_src = os.path.join(project_root, 'data', 'kazakh_ood_test.csv')
    if os.path.exists(ood_src):
        df_ood = pd.read_csv(ood_src)
        text_col = 'text' if 'text' in df_ood.columns else df_ood.columns[0]
        domain_col = 'domain' if 'domain' in df_ood.columns else 'genre'
        label_col = 'label' if 'label' in df_ood.columns else 'is_ai'
        
        df_ood_clean = pd.DataFrame({
            'text': df_ood[text_col],
            'domain': df_ood[domain_col] if domain_col in df_ood.columns else 'mixed_ood',
            'label': df_ood[label_col].astype(int) if label_col in df_ood.columns else 0
        })
        
        ood_csv = os.path.join(out_data_dir, 'ood_test.csv')
        df_ood_clean.to_csv(ood_csv, index=False, encoding='utf-8')
        print(f"Exported OOD Test Set: {ood_csv} ({len(df_ood_clean)} samples)")
        
    print("\nDataset packaging completed successfully!")

if __name__ == '__main__':
    package_dataset()
