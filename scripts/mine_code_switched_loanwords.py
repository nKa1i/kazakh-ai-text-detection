import os
import re
import json
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Comprehensive list of common Russian/international loanword roots used in Kazakh reviews & social media
COMMON_LOANWORD_ROOTS = [
    'доставка', 'оплата', 'возврат', 'заказ', 'бонус', 'акция', 'клиент', 'карта',
    'банк', 'номер', 'приложение', 'аккаунт', 'профиль', 'страница', 'пароль', 'код',
    'каспи', 'инстаграм', 'ватсап', 'телеграм', 'сет', 'сеть', 'чек', 'счет',
    'отзыв', 'товар', 'заявка', 'гранд', 'грант', 'кабинет', 'сумма', 'меню', 'плюс',
    'видео', 'фото', 'файл', 'арна', 'оператор', 'комиссия', 'процент', 'качество',
    'сервис', 'курьер', 'бронь', 'скидка', 'аккаунт'
]

# Standard Kazakh agglutinative suffixes attached to nouns
KAZAKH_SUFFIXES = [
    'ыңыз', 'іңіз', 'ңыздар', 'ңіздер', 'сыңдар', 'сіңдер',
    'лар', 'лер', 'тар', 'тер', 'дар', 'дер',
    'дың', 'дің', 'тың', 'тің', 'ның', 'нің',
    'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
    'да', 'де', 'та', 'те', 'нда', 'нде',
    'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
    'ға', 'ге', 'қа', 'ке', 'на', 'не',
    'мен', 'бен', 'пен',
    'ым', 'ім', 'мыз', 'міз', 'ың', 'ің', 'сы', 'сі'
]

def mine_loanwords(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    
    extracted = {}
    for text in df[text_col].dropna():
        words = re.findall(r'\b[а-яА-ЯёЁәӘғҒқҚңҢөӨұҰүҮһҺіІ]+\b', str(text).lower())
        for w in words:
            for root in COMMON_LOANWORD_ROOTS:
                if w.startswith(root) and len(w) > len(root):
                    sfx = w[len(root):]
                    if sfx in KAZAKH_SUFFIXES:
                        if w not in extracted:
                            extracted[w] = {"root": root, "suffix": f"-{sfx}", "count": 0}
                        extracted[w]["count"] += 1

    return extracted

def run_mining():
    data_dir = "data"
    test_csv = os.path.join(data_dir, "kazakh_realworld_test.csv")
    
    results = mine_loanwords(test_csv)
    
    # Sort by frequency descending
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["count"], reverse=True))
    
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "code_switched_loanword_lexicon.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        
    print("=" * 60)
    print(f"CODE-SWITCHED LOANWORD MINING COMPLETED")
    print(f"Unique hybrid loanwords mined: {len(sorted_results)}")
    print(f"Saved lexicon to: {out_path}")
    print("Sample Mined Loanwords:")
    for k, v in list(sorted_results.items())[:10]:
        print(f"  {k} -> root: '{v['root']}', suffix: '{v['suffix']}' (count: {v['count']})")
    print("=" * 60)
    
    return sorted_results

if __name__ == "__main__":
    run_mining()
