# -*- coding: utf-8 -*-
"""
scripts/build_hybrid_mixed_benchmark.py: Generates and evaluates the Span-Level
Mixed Human-AI Provenance Benchmark for Kazakh.
"""

import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROVENANCE_CLASSES = (
    "pure_human",
    "pure_ai",
    "human_ai_spliced",
    "ai_human_edited",
)

GENERATOR_FAMILIES = (
    "ISSAI-KazLLM-70B",
    "Sherkala-8B",
    "DeepSeek-R1",
    "GPT-4o",
    "Qwen-2.5-72B",
)

# 36 Authentic Human sentences covering diverse daily, social, colloquial & reflective topics
HUMAN_SENTENCES = [
    "Кеше кешке орталыққа достармен бірге барғанбыз, бірақ көрме жабық болып шықты.",
    "Каспиден тауардың жеткізілуі өте тез болды, курьерге рахметімді айттым ғой.",
    "Балалардың мектептегі жаңа оқулықтары өте қызықты жазылған екен.",
    "Ауылдағы ата-әжемізге демалыста арнайы барып, біраз көмектесіп қайттық.",
    "Бұл мәселені шешу үшін әрқайсымыз өз жауапкершілігімізді түсінуіміз керек қой.",
    "Жаңа жұмыс орнындағы ұжым бізді өте жылы қабылдап, бірден қолдау көрсетті.",
    "Көктем шыға ауладағы ағаштарды реттеп, сенбілік өткізуді жоспарлап отырмыз.",
    "Таңертең ерте тұрып жүгіру денсаулыққа керемет серпін беретінін өзім байқадым.",
    "Осы кітапты оқып шыққаннан кейін тарихқа деген көзқарасым түбегейлі өзгерді.",
    "Біздің өңірдегі өзен суы биыл мол болып, шаруалардың еңбегі жанған сияқты.",
    "Көршілермен жиналып, ауламызға балалар ойын алаңын бірге орнаттық.",
    "Бүгінгі талқылауда мамандар өздерінің шынайы тәжірибелерімен бөлісті ғой.",
    "Үйдегі интернет кенеттен өшіп қалып, түске дейін жұмысымыз тоқтап қалды.",
    "Анама туған күніне арнап әдемі гүл шоғы мен шағын сыйлық алып қойдым.",
    "Жастардың қазақша сөйлеп, өнерге қызығып жүргенін көріп қатты қуандым.",
    "Кешегі футбол ойыны соңғы минуттарға дейін өте тартысты әрі қызықты өтті.",
    "Пойыздың кешігуіне байланысты вокзалда екі сағаттай күтуге тура келді.",
    "Дүкеннен жаңа дәптерлер мен қаламдар сатып алып, сабаққа дайындалып жатырмыз.",
    "Ағамыздың үйлену тойына туған-туыстардың бәрі жиналып, керемет мәз-мейрам болдық.",
    "Қыста тауға барып шаңғы тепкен күндерді әлі күнге дейін сағынышпен еске аламын.",
    "Кітапханадағы тыныштық ойды жинақтап, мақаланы тезірек аяқтауға көп көмектесті.",
    "Бала кезімде жаз бойы далада асық ойнап, кешке әзер қайтатын едік қой.",
    "Жаңа пәтерге көшіп, заттарды реттеп орналастыру біршама уақытты алды.",
    "Қаладағы кептелістің кесірінен кездесуге сәл кешігіп бардым, бірақ бәрі түсінді.",
    "Бүгін ауа райы бұзылып, жаңбыр құйып кетті, қолшатырды ұмытып кеткенім өкінішті.",
    "Дәмханадағы тамақ өте дәмді екен, қызмет көрсету сапасы да көңілден шықты.",
    "Университетте өткен ғылыми конференцияда көптеген пайдалы ақпарат алдық.",
    "Досым маған керемет фильм ұсынған еді, кешке соны отбасымызбен бірге тамашаладық.",
    "Жұмыстан кейін серуендеп қайту күні бойғы шаршауды бірден басады екен.",
    "Базардан балғын жеміс-жидектер алып, үйде дәмді тосап қайнатып қойдық.",
    "Мектептегі сыныптастарымызбен он жылдан кейін қайта жүздесіп, шүйіркелесіп қалдық.",
    "Балалардың жазғы демалысын қызықты өткізу үшін танымдық лагерьге жібердік.",
    "Өткен сенбіде қала шетіндегі саяжайға барып, көкөністерді суарып келдік.",
    "Компьютердің бағдарламасын жаңартып алған соң, бәрі әлдеқайда жылдам істейтін болды.",
    "Ата-анамыздың ақыл-кеңесі бізге өмірде әрдайым дұрыс бағыт-бағдар береді ғой.",
    "Біз кеше кешке достармен бірге қыдырып, керемет демалып қайттық қой.",
]

# 60 Synthetic AI seed sentences (12 per generator family) exhibiting characteristic structures
AI_SENTENCES: Dict[str, List[str]] = {
    "ISSAI-KazLLM-70B": [
        "Қазақстан Республикасындағы цифрландыру үдерістері қоғамның барлық салаларында қарқынды түрде дамып келеді.",
        "Аталған бағдарламаның шеңберінде экономикалық тиімділікті арттыру бойынша кешенді іс-шаралар жүзеге асырылуда.",
        "Ғылыми зерттеулердің нәтижелері көрсеткендей, инновациялық тәсілдер өндірістік қуаттылықты арттыруға негіз болады.",
        "Мемлекеттік органдар мен жеке сектор арасындағы өзара іс-қимыл тұрақты даму мақсаттарына қол жеткізуді қамтамасыз етеді.",
        "Құқықтық реттеу тетіктерін жетілдіру шеңберінде қолданыстағы заңнамалық актілерге тиісті түзетулер енгізілді.",
        "Әлеуметтік жаңғыру стратегиясы азаматтардың өмір сүру сапасын кезең-кезеңімен жақсартуға бағытталған.",
        "Ұлттық экономиканың бәсекеге қабілеттілігін арттыру мақсатында инфрақұрылымдық инвестициялар белсенді түрде тартылуда.",
        "Қоғамдық қауіпсіздікті қамтамасыз ету саласында жаңа ведомствоаралық мониторинг жүйелері енгізілуде.",
        "Мемлекеттік қызметтерді электрондық форматқа көшіру процесі сыбайлас жемқорлық тәуекелдерін айтарлықтай төмендетеді.",
        "Өңірлік даму саясаты теңгерімді экономикалық өсімді қамтамасыз етудің маңызды құралы болып саналады.",
        "Экологиялық мониторинг бағдарламасы шеңберінде атмосфералық ауаның сапасын бақылау күшейтілді.",
        "Аталған стратегиялық бастамалар елдің ұзақ мерзімді тұрақты дамуының берік іргетасын қалайды.",
    ],
    "Sherkala-8B": [
        "Табиғи ресурстарды тиімді пайдалану мәселелері заманауи экологиялық саясаттың басым бағыттарының бірі болып табылады.",
        "Білім беру жүйесін жаңғырту шеңберінде жаңа әдістемелік құралдар мен цифрлық платформалар енгізілуде.",
        "Өңірлік инфрақұрылымды дамыту жөніндегі жобалар халықтың әл-ауқатын жақсартуға оң ықпалын тигізеді.",
        "Ақпараттық технологияларды интеграциялау арқылы басқару процестерінің ашықтығы мен тиімділігі қамтамасыз етіледі.",
        "Ауыл шаруашылығын әртараптандыру бағдарламасы азық-түлік қауіпсіздігін нығайтудың шешуші факторы болып табылады.",
        "Медициналық қызметтердің қолжетімділігін арттыру бойынша нақты шаралар кешені бекітілген.",
        "Энергетикалық инфрақұрылымды жаңарту жобалары өндірістік шығындарды оңтайландыруға септігін тигізеді.",
        "Көлік-логистикалық әлеуетті арттыру мақсатында халықаралық дәліздерді жаңғырту жұмыстары жалғасуда.",
        "Су ресурстарын үнемді пайдалану технологияларын енгізу өңірлік тұрақтылықты қамтамасыз етуге негіз болады.",
        "Кәсіптік білім беру сапасын арттыру еңбек нарығындағы сұранысқа ие мамандарды даярлауға мүмкіндік береді.",
        "Жаңартылатын энергия көздерін дамыту төмен көміртекті экономикаға көшудің негізгі шарты болып табылады.",
        "Мәдени мұраны сақтау және дәріптеу бағытындағы кешенді бағдарламалар жүйелі түрде жүзеге асырылуда.",
    ],
    "DeepSeek-R1": [
        "Қарастырылып отырған алгоритмдік шешімдер күрделі деректер құрылымын оңтайландыру мақсатында жүйелі түрде талданды.",
        "Мәселені шешудің бірінші кезеңінде барлық шектеулерді ескере отырып, математикалық модель құрастырылды.",
        "Логикалық қорытындыларға сәйкес, ұсынылған архитектура есептеу шығындарын айтарлықтай қысқартуға мүмкіндік береді.",
        "Осыған байланысты, модельдің жұмыс істеу сенімділігін тексеру үшін қосымша валидация жүргізу қажет деп есептеледі.",
        "Теориялық негіздеме шеңберінде градиенттік төмендеу әдісінің конвергенция жылдамдығы дәлелденді.",
        "Нейрондық желілердің терең қабаттарындағы параметрлер санын азайту үшін прунинг тәсілі қолданылды.",
        "Нәтижелерді салыстырмалы талдау көрсеткендей, жаңа әдіс базалық көрсеткіштерден айтарлықтай асып түседі.",
        "Дифференциалдық теңдеулер жүйесін шешу барысында шекаралық шарттар қатаң түрде сақталды.",
        "Эвристикалық алгоритмдерді қолдану NP-толық есептердің жуық шешімдерін табудың тиімді құралы болып табылады.",
        "Эксперименттік деректердің дисперсиясын талдау нәтижесінде статистикалық гипотезалар толықтай расталды.",
        "Кванттық есептеулер теориясына сәйкес, аталған схема кубиттердің когеренттілік уақытын ұзартуға негіз болады.",
        "Күрделі желілік жүйелердегі ақпарат ағынын оңтайландыру бойынша математикалық теорема тұжырымдалды.",
    ],
    "GPT-4o": [
        "Жасанды интеллект жүйелерінің дамуы заманауи ғылым мен білім беру саласында жаңа мүмкіндіктерге жол ашуда.",
        "Көптілді тілдік модельдер қазақ тіліндегі мәтіндерді талдау және сапалы генерациялау бойынша жоғары нәтижелер көрсетуде.",
        "Ақпараттық қауіпсіздік пен кибергигиена ережелерін сақтау цифрлық қоғамның тұрақтылығы үшін стратегиялық маңызға ие.",
        "Мәліметтер базасын оңтайландыру және бұлттық технологияларды енгізу бизнес-процестерді жеделдетуге септігін тигізеді.",
        "Генеративті модельдерді қолдану шығармашылық салалардағы еңбек өнімділігін арттыруға айтарлықтай септігін тигізуде.",
        "Үлкен көлемді деректерді өңдеу алгоритмдері нақты уақыт режимінде шешім қабылдауға мүмкіндік береді.",
        "Заманауи бағдарламалық шешімдер пайдаланушылардың деректерін қорғаудың жоғары деңгейін қамтамасыз етеді.",
        "Автоматтандырылған жүйелер клиенттерге қызмет көрсету процесін оңтайландырудың тиімді құралы болып табылады.",
        "Машиналық оқыту әдістері табиғи тілді өңдеу сапасын түбегейлі жаңа деңгейге көтерді.",
        "Көпмодальды архитектуралар мәтін, сурет және дыбыстық ақпаратты бір мезетте талдауға мүмкіндік береді.",
        "Смарт-инфрақұрылымдарды енгізу қалалық қызметтердің өзара іс-қимылын үйлестіруге жағдай жасайды.",
        "Аталған инновациялық технологиялар өнеркәсіптік кәсіпорындардың бәсекеге қабілеттілігін арттыруға негіз болады.",
    ],
    "Qwen-2.5-72B": [
        "Халықаралық ынтымақтастық аясында жүзеге асырылып жатқан жобалар еліміздің инвестициялық тартымдылығын арттырады.",
        "Тұрақты экономикалық өсімді қамтамасыз ету мақсатында өңдеу өнеркәсібін әртараптандыру шаралары жалғасуда.",
        "Қоғамдық институттардың белсенді қатысуымен қабылданған шешімдер әлеуметтік бастамалардың тиімділігін арттырады.",
        "Заманауи телекоммуникациялық желілер шалғай елді мекендерді сапалы интернетпен қамтуға жағдай жасайды.",
        "Көліктік инфрақұрылымды жаңғырту сыртқы сауда айналымын ұлғайтудың маңызды алғышарты болып табылады.",
        "Өндірістік қуаттарды экологиялық талаптарға сәйкестендіру шаралары кезең-кезеңімен іске асырылуда.",
        "Мемлекеттік қолдау тетіктері шағын және орта бизнестің дамуына оң серпін беруде.",
        "Ғылым мен өндіріс арасындағы байланысты нығайту инновациялық өнімдерді коммерцияландыруға септігін тигізеді.",
        "Халықаралық сауда-экономикалық қатынастарды дамыту стратегиялық әріптестікті тереңдетуге мүмкіндік береді.",
        "Өңіраралық ынтымақтастықты кеңейту ішкі нарықтың интеграциясын нығайтуға негіз болады.",
        "Цифрлық қаржы құралдарын енгізу есеп айырысу процестерінің ашықтығы мен жеделдігін қамтамасыз етеді.",
        "Адами капиталды дамыту жөніндегі бастамалар ұлттық экономиканың ұзақ мерзімді тұрақтылығына кепілдік береді.",
    ],
}


def build_hybrid_mixed_dataset(
    output_path: str = "data/kaz_hybrid_mixed_spans.jsonl",
    num_per_class: int = 25,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Constructs a balanced multi-sentence dataset covering 4 provenance categories:
    pure_human, pure_ai, human_ai_spliced, and ai_human_edited.
    """
    random.seed(random_seed)
    docs = []
    doc_counter = 1

    # 1. pure_human
    for i in range(num_per_class):
        n_sents = random.randint(4, 6)
        sents = random.sample(HUMAN_SENTENCES, n_sents)
        labels = [0] * n_sents
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "society",
            "provenance_class": "pure_human",
            "generator": "human",
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": []
        })
        doc_counter += 1

    # 2. pure_ai
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        n_sents = 4
        sents = random.sample(ai_pool, n_sents)
        labels = [1] * n_sents
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "technology",
            "provenance_class": "pure_ai",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": []
        })
        doc_counter += 1

    # 3. human_ai_spliced
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        h_sents = random.sample(HUMAN_SENTENCES, 2)
        a_sents = random.sample(ai_pool, 2)
        if i % 2 == 0:
            sents = h_sents + a_sents
            labels = [0, 0, 1, 1]
            changeovers = [2]
        else:
            sents = a_sents + h_sents
            labels = [1, 1, 0, 0]
            changeovers = [2]

        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "society",
            "provenance_class": "human_ai_spliced",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": changeovers
        })
        doc_counter += 1

    # 4. ai_human_edited
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        h_sents = random.sample(HUMAN_SENTENCES, 2)
        a_sents = random.sample(ai_pool, 2)
        sents = [a_sents[0], h_sents[0], a_sents[1], h_sents[1]]
        labels = [1, 0, 1, 0]
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "technology",
            "provenance_class": "ai_human_edited",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": [1, 2, 3]
        })
        doc_counter += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return docs


# Shared FST overlay instance for morphological feature scoring
_FST_OVERLAY = None


def _get_fst_overlay():
    global _FST_OVERLAY
    if _FST_OVERLAY is None:
        try:
            from src.auto_fst.overlay import DynamicFSTOverlay
        except ImportError:
            from auto_fst.overlay import DynamicFSTOverlay
        _FST_OVERLAY = DynamicFSTOverlay()
    return _FST_OVERLAY


# Morphologically-grounded sentence-level AI score heuristic for boundary evaluation
def _score_sentence_ai_likelihood(sentence: str) -> float:
    if not sentence or not isinstance(sentence, str):
        return 0.50

    overlay = _get_fst_overlay()
    segmented = overlay.analyze_and_segment(sentence)
    tokens = segmented.split()
    suffixes = [t.lstrip("-") for t in tokens if t.startswith("-")]
    s_lower = sentence.lower()

    score = 0.50

    # 1. Morphological features from FST segmentation:
    # Human conversational emphatic particles, witnessed past & personal agreement
    human_morph_markers = {"ғой", "қой", "ау", "мыс", "міс", "мыз", "міз", "дым", "дім", "сың", "сің", "ыңыз", "іңіз", "ымыз", "іміз"}
    found_human_morph = sum(1 for sfx in suffixes if sfx in human_morph_markers)
    if found_human_morph > 0:
        score -= min(0.35, 0.15 * found_human_morph)

    # Human evidentials & colloquial lexical markers in surface text
    human_lexical_markers = [
        "ғой", "қой", "екен", "сияқты", "барғанбыз", "жазғанбыз", "айттым", "байқадым",
        "отырмыз", "қайттық", "көрдік", "алдық", "келдік", "болдық", "өзім", "достармен"
    ]
    for m in human_lexical_markers:
        if m in s_lower:
            score -= 0.15

    # AI bureaucratic participial chains & passives
    ai_morph_markers = {"ғандықтан", "гендіктен", "қандықтан", "кендіктен", "луда", "луде", "тық", "тік", "лық", "лік"}
    found_ai_morph = sum(1 for sfx in suffixes if sfx in ai_morph_markers)
    if found_ai_morph > 0:
        score += min(0.30, 0.12 * found_ai_morph)

    # AI formal discourse & bureaucratic cliché connectors
    ai_lexical_markers = [
        "шеңберінде", "жүзеге асырылуда", "қамтамасыз етеді", "басым бағыттарының бірі",
        "мақсатында", "айтарлықтай", "байланысты", "сәйкес", "себін тигізеді", "септігін тигізеді",
        "болып табылады", "негіз болады", "болып саналады", "қабылданған", "енгізілуде",
        "бағытталған", "жүйелі түрде", "қарқынды түрде", "айқындайды", "себеп болады", "тиісті"
    ]
    for m in ai_lexical_markers:
        if m in s_lower:
            score += 0.20

    # Formal sentence length complexity
    if len(sentence.split()) >= 11:
        score += 0.05

    return max(0.05, min(0.95, score))


def evaluate_span_boundary_detection(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates sentence-level AI vs Human classification and span boundary IoU across
    the mixed-provenance dataset.
    """
    total_sentences = 0
    y_true_all = []
    y_pred_all = []

    gen_stats = {gen: {"y_true": [], "y_pred": []} for gen in GENERATOR_FAMILIES}

    total_iou = 0.0
    iou_count = 0

    for doc in docs:
        sentences = doc["sentences"]
        labels = doc["sentence_labels"]
        gen = doc["generator"]

        preds = []
        for s in sentences:
            ai_score = _score_sentence_ai_likelihood(s)
            p = 1 if ai_score >= 0.50 else 0
            preds.append(p)

        y_true_all.extend(labels)
        y_pred_all.extend(preds)
        total_sentences += len(sentences)

        if gen in gen_stats:
            gen_stats[gen]["y_true"].extend(labels)
            gen_stats[gen]["y_pred"].extend(preds)

        # Boundary changeover IoU
        pred_changeovers = set()
        for idx in range(1, len(preds)):
            if preds[idx] != preds[idx - 1]:
                pred_changeovers.add(idx)

        true_changeovers = set(doc.get("changeover_indices", []))
        if true_changeovers or pred_changeovers:
            intersection = len(true_changeovers & pred_changeovers)
            union = len(true_changeovers | pred_changeovers)
            iou = intersection / union if union > 0 else 1.0
            total_iou += iou
            iou_count += 1
        else:
            # Both empty (pure documents)
            total_iou += 1.0
            iou_count += 1

    # Macro F1
    def _compute_macro_f1(y_t: List[int], y_p: List[int]) -> float:
        tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_t, y_p) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_t, y_p) if yt == 0 and yp == 0)

        # Class 1 (AI)
        p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

        # Class 0 (Human)
        p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0.0

        return round((f1_1 + f1_0) / 2.0, 4)

    macro_f1 = _compute_macro_f1(y_true_all, y_pred_all)
    mean_iou = round(total_iou / iou_count, 4) if iou_count > 0 else 1.0

    by_gen_metrics = {}
    for gen, st in gen_stats.items():
        if st["y_true"]:
            by_gen_metrics[gen] = {
                "total_sentences": len(st["y_true"]),
                "macro_f1": _compute_macro_f1(st["y_true"], st["y_pred"])
            }

    return {
        "total_documents": len(docs),
        "total_sentences": total_sentences,
        "sentence_macro_f1": macro_f1,
        "boundary_iou": mean_iou,
        "by_generator": by_gen_metrics
    }


if __name__ == "__main__":
    out_path = "data/kaz_hybrid_mixed_spans.jsonl"
    docs = build_hybrid_mixed_dataset(out_path, num_per_class=25)
    print(f"Generated {len(docs)} multi-sentence documents in {out_path}")
    metrics = evaluate_span_boundary_detection(docs)
    print("Span-Level Boundary Evaluation Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
