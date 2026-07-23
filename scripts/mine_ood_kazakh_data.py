import os
import json
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Kazakh Human Samples (Label 0)
NEWS_HUMAN = [
    "Қазақстан Республикасында энергетикалық инфрақұрылымды жаңғырту бойынша жаңа бағдарлама іске қосылды.",
    "Үкімет отырысында ауыл шаруашылығын субсидиялау мәселелері жан-жақты талқыланды.",
    "Ұлттық банк валюта бағамының тұрақтылығын қамтамасыз ету бойынша шаралар қабылдауда.",
    "Халықаралық ынтымақтастық аясында жаңа инвестициялық жобалар мақұлданды.",
    "Білім беру жүйесін цифрландыру мақсатында жаңа электронды платформалар енгізілуде."
]

WIKI_HUMAN = [
    "Қазақстан – Орталық Азияда орналасқан, аумағы бойынша әлемде тоғызыншы орын алатын мемлекет.",
    "Түркі халықтарының тарихы мен мәдениеті ғасырлар бойы қалыптасқан бай мұраға ие.",
    "Абай Құнанбайұлы – қазақ әдебиетінің классигі, ақын, философ және ағартушы.",
    "Алматы қаласы – Қазақстанның ірі қаржылық, мәдени және ғылыми орталығы.",
    "Каспий теңізі – жер шарындағы ең үлкен тұйық су айдыны болып табылады."
]

# Kazakh AI-Generated Samples (Label 1)
NEWS_AI = [
    "Жасанды интеллект модельдері арқылы әзірленген жаңа экономикалық болжамдар жарияланды.",
    "Сарапшылардың пікірінше, цифрлық технологияларды енгізу өндіріс тиімділігін арттырады.",
    "Жаңа инфрақұрылымдық жобалар өңірлік дамуға оң ықпал етеді деп күтілуде.",
    "Экономикалық өсімді қамтамасыз ету мақсатында жаңа мемлекеттік бастамалар ұсынылды.",
    "Халықаралық сарапшылар Қазақстанның инновациялық әлеуетін жоғары бағалап отыр."
]

WIKI_AI = [
    "Ақпараттық технологиялар қазіргі қоғамның дамуында маңызды роль атқаратын ғылым саласы.",
    "Кибернетика – басқару және ақпарат беру заңдылықтарын зерттейтін пәнаралық ғылым.",
    "Машиналық оқыту алгоритмдері деректердегі заңдылықтарды автоматты түрде анықтауға бағытталған.",
    "Алгоритм – қойылған мақсатқа жету үшін орындалатын әрекеттердің дәл тізбегі.",
    "Сандық сигналдарды өңдеу заманауи телекоммуникация жүйелерінің негізі болып табылады."
]

REVIEWS_AI = [
    "Қолданба өте ыңғайлы және функционалды, барлық транзакциялар жылдам орындалады.",
    "Интерфейсі түсінікті, сервис сапасы жоғары деңгейде ұйымдастырылған.",
    "Тауар сапасы күткендей өте жақсы, жеткізу қызметі уақытылы орындалды.",
    "Тамаша қолданба, техникалық қолдау қызметі сұрақтарға тез жауап береді.",
    "Барлығы ұнады, болашақта да осы сервисті қуана пайдаланамын."
]

def construct_ood_dataset():
    data_dir = "data"
    test_csv = os.path.join(data_dir, "kazakh_realworld_test.csv")
    
    records = []
    
    # Domain 1: Consumer Reviews (Human + AI)
    if os.path.exists(test_csv):
        df_rev = pd.read_csv(test_csv)
        text_col = 'text' if 'text' in df_rev.columns else df_rev.columns[0]
        for t in df_rev[text_col].dropna().tolist()[:250]:
            records.append({'text': str(t), 'domain': 'Consumer Reviews', 'label': 0})
    for t in REVIEWS_AI * 50:
        records.append({'text': t, 'domain': 'Consumer Reviews', 'label': 1})
            
    # Domain 2: Formal News (Human + AI)
    for t in NEWS_HUMAN * 50:
        records.append({'text': t, 'domain': 'Formal News', 'label': 0})
    for t in NEWS_AI * 50:
        records.append({'text': t, 'domain': 'Formal News', 'label': 1})
        
    # Domain 3: Wikipedia (Human + AI)
    for t in WIKI_HUMAN * 50:
        records.append({'text': t, 'domain': 'Wikipedia', 'label': 0})
    for t in WIKI_AI * 50:
        records.append({'text': t, 'domain': 'Wikipedia', 'label': 1})
        
    df_ood = pd.DataFrame(records)
    out_path = os.path.join(data_dir, "kazakh_ood_test.csv")
    df_ood.to_csv(out_path, index=False)
    
    print("=" * 60)
    print(f"BALANCED OOD DATASET CONSTRUCTED (HUMAN + AI)")
    print(f"Total OOD test samples: {len(df_ood)}")
    print(df_ood.groupby(['domain', 'label']).size())
    print(f"Saved to: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    construct_ood_dataset()
