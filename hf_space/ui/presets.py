"""Curated benchmark presets representing the 6 cells of our Kazakh AI detection benchmark.

Includes authentic samples from:
1. Authentic Kaspi Consumer Review (Human)
2. Authentic Formal News Article (Human)
3. Authentic Academic Wikipedia Article (Human)
4. Sherkala-7B Generated Article (AI)
5. Qwen-2.5-7B-Instruct Wild Sample (AI)
6. Injected Hybrid Essay (Partially AI)
"""

from typing import Dict, List, Optional

PRESET_SAMPLES: Dict[str, Dict[str, str]] = {
    "1. Authentic Kaspi Consumer Review (Human)": {
        "title": "1. Authentic Kaspi Consumer Review (Human)",
        "domain": "Consumer Reviews",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Authentic colloquial Kazakh review with consumer terminology, positive sentiment, and informal conversational style.",
        "text": (
            "Kaspi-ден тапсырыс берген шаңсорғыш өте керемет екен! Сапасы мықты, бағасына сай. "
            "Жеткізуші жігіт тез әкеліп берді, қорабы бүлінбеген, таза. Үйде мысық бар еді, жүнін кілемнен "
            "өте жақсы тазалайды, шуы да қатты емес. Өзіме қатты ұнады, рахмет сатушыға және Kaspi дүкеніне! "
            "Барлығына осы тауарды сатып алуға кеңес беремін."
        ),
    },
    "2. Authentic Formal News Article (Human)": {
        "title": "2. Authentic Formal News Article (Human)",
        "domain": "Formal News",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Formal journalistic news release with official discourse markers and Tengrinews-style journalistic structure.",
        "text": (
            "Астана қаласында цифрландыру және жасанды интеллект саласындағы ынтымақтастыққа арналған "
            "халықаралық форум өз жұмысын бастады. Іс-шараға отандық және шетелдік жетекші сарапшылар, "
            "IT-компаниялардың басшылары мен мемлекеттік органдардың өкілдері қатысуда. "
            "Жиын барысында Қазақстанда мемлекеттік қызметтерді автоматтандыру, деректерді қорғау және "
            "ұлттық тілдік модельдерді дамыту мәселелері талқыланды. Сарапшылардың айтуынша, жаңа цифрлық "
            "шешімдерді өндіріске енгізу ел экономикасының бәсекеге қабілеттілігін арттыруға зор мүмкіндік береді."
        ),
    },
    "3. Authentic Academic Wikipedia Article (Human)": {
        "title": "3. Authentic Academic Wikipedia Article (Human)",
        "domain": "Academic / Wikipedia",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Encyclopedic academic prose with formal definitions, historical context, and rich agglutinative morphology.",
        "text": (
            "Қазақ хандығы — XV ғасырдың ортасында Жетісу және Дешті Қыпшақ аумағында құрылған ортағасырлық "
            "дербес мемлекет. Оның негізін қалаушылар Керей мен Жәнібек хандар болып табылады. Хандық қазақ "
            "ру-тайпаларының басын біріктіріп, көшпелі және отырықшы өркениеттер арасындағы саяси және "
            "экономикалық тепе-теңдікті қамтамасыз етті. Мемлекеттің құқықтық негіздері дәстүрлі дала заңдарымен "
            "және кейіннен Қасым ханның қасқа жолы, Есім ханның ескі жолы мен Тәуке ханның Жеті жарғысы сияқты "
            "заң жинақтарымен реттелді."
        ),
    },
    "4. Sherkala-7B Generated Article (AI)": {
        "title": "4. Sherkala-7B Generated Article (AI)",
        "domain": "Sherkala-7B LLM",
        "expected_verdict": "Machine-Generated",
        "generator": "Sherkala-7B",
        "description": "Sherkala-7B Kazakh LLM generated text showing characteristic synthetic patterns, structural uniformity, and repetition.",
        "text": (
            "Жасанды интеллект — қазіргі таңда қарқынды дамып келе жатқан маңызды технологиялық бағыт болып табылады. "
            "Жасанды интеллект адам өмірін жеңілдетуге және өндірістік процестерді оңтайландыруға көмектеседі. "
            "Бұл жүйелер үлкен көлемдегі деректерді өңдеуге және шешім қабылдауды жеделдетуге арналған. "
            "Сондықтан жасанды интеллект технологияларын дамыту бүгінгі қоғам үшін өте қажет. "
            "Осылайша, бұл бағыт білім беру, денсаулық сақтау және өнеркәсіп салаларында жоғары нәтижелер көрсетіп келеді."
        ),
    },
    "5. Qwen-2.5-7B-Instruct Wild Sample (AI)": {
        "title": "5. Qwen-2.5-7B-Instruct Wild Sample (AI)",
        "domain": "Wild LLM (Qwen-2.5)",
        "expected_verdict": "Machine-Generated",
        "generator": "Qwen-2.5-7B-Instruct",
        "description": "Unseen wild generator sample exhibiting high syntactic fluency but detectable subtle synthetic markers.",
        "text": (
            "Заманауи цифрлық дәуірде бағдарламалау тілдерін меңгеру кез келген маман үшін маңызды стратегиялық "
            "артықшылыққа айналды. Атап айтқанда, Python тілі өзінің түсінікті синтаксисі мен бай кітапханалық "
            "инфрақұрылымының арқасында машиналық оқыту мен деректер талдауында жетекші орын алады. "
            "Сонымен қатар, ақпараттық технологиялар нарығы үнемі жаңа инновациялық құралдарды талап етуде. "
            "Осы себепті терең алгоритмдік білім мен үздіксіз тәжірибе мамандардың кәсіби тұрғыда сұранысқа "
            "ие болуын қамтамасыз етеді."
        ),
    },
    "6. Injected Hybrid Essay (Partially AI)": {
        "title": "6. Injected Hybrid Essay (Partially AI)",
        "domain": "Student Essay / Hybrid",
        "expected_verdict": "Partially AI / Hybrid",
        "generator": "Human + Qwen-2.5-7B-Instruct",
        "description": "Authentic student essay with an injected AI paragraph demonstrating multi-paragraph hybrid document detection.",
        "text": (
            "Менің ойымша, әрбір жас буын өз елінің тарихы мен ана тілін терең білуі керек. Тіл — халықтың ғасырлар "
            "бойы қалыптасқан рухани қазынасы әрі ұлттық болмысының айнасы. Өз тамырын құрметтемеген адамның "
            "келешегі бұлыңғыр болары сөзсіз.\n\n"
            "Жасанды интеллект технологиялары білім беру жүйесін жаңғыртуда маңызды рөл атқарады. "
            "Интеллектуалды алгоритмдер әрбір студенттің білім деңгейіне қарай бейімделген оқу бағдарламаларын "
            "автоматты түрде құрастыруға мүмкіндік береді. Бұл тәсіл оқу үлгерімін жақсартады және педагогикалық "
            "процестің тиімділігін айтарлықтай арттырады.\n\n"
            "Алайда, кез келген заманауи технологияны игеру ұлттық құндылықтарымызды ұмытуға себеп болмауы тиіс. "
            "Біз озық білімді меңгере отырып, рухани мұрамызды сақтап, еліміздің гүлденуіне өз үлесімізді қосуымыз қажет."
        ),
    },
}


def get_preset_choices() -> List[str]:
    """Returns the ordered list of preset demonstration titles."""
    return list(PRESET_SAMPLES.keys())


def get_preset_text(preset_name: Optional[str]) -> str:
    """Returns the text for the given preset name, or empty string if not found."""
    if not preset_name or preset_name not in PRESET_SAMPLES:
        return ""
    return PRESET_SAMPLES[preset_name].get("text", "")


def get_preset_metadata(preset_name: Optional[str]) -> Dict[str, str]:
    """Returns the metadata dictionary for the given preset name, or empty dict if not found."""
    if not preset_name or preset_name not in PRESET_SAMPLES:
        return {}
    sample = PRESET_SAMPLES[preset_name]
    return {
        "title": sample.get("title", preset_name),
        "domain": sample.get("domain", ""),
        "expected_verdict": sample.get("expected_verdict", ""),
        "generator": sample.get("generator", ""),
        "description": sample.get("description", ""),
    }


VERIFICATION_PRESET_SAMPLES: Dict[str, Dict[str, str]] = {
    "Quadrant 1: Verified Human Fact": {
        "title": "Quadrant 1: Verified Human Fact",
        "quadrant": "Verified Human Fact",
        "description": "Authentic human prose containing verified historical facts about Kazakhstan's independence and capital.",
        "text": (
            "Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады. "
            "Астана қаласы — Қазақстанның елордасы болып табылады. "
            "1997 жылы елорда Алматы қаласынан Ақмолаға көшірілді."
        )
    },
    "Quadrant 2: Human Misinformation": {
        "title": "Quadrant 2: Human Misinformation",
        "quadrant": "Human Misinformation",
        "description": "Human-written informal text containing widespread factual errors and false entity claims.",
        "text": (
            "Меніңше, Шымкент қаласы — Қазақстанның ресми елордасы болып табылады. "
            "Қазақстан өз тәуелсіздігін 1998 жылы ресми түрде жариялаған болатын."
        )
    },
    "Quadrant 3: Hallucinatory AI Disinformation": {
        "title": "Quadrant 3: Hallucinatory AI Disinformation",
        "quadrant": "Hallucinatory AI Disinformation",
        "description": "LLM-generated text exhibiting typical hallucinated historical disinformation and false dates.",
        "text": (
            "Абай Құнанбайұлы француз тілінде бес роман жазған және айға ұшып барған. "
            "Қазақстанның ұлттық валютасы теңге 1917 жылы айналымға енгізілген болатын."
        )
    },
    "Quadrant 4: Accurate AI Synthesis": {
        "title": "Quadrant 4: Accurate AI Synthesis",
        "quadrant": "Accurate AI Synthesis",
        "description": "LLM-generated encyclopedic synthesis that is stylistically artificial but factually accurate.",
        "text": (
            "Тоқтар Оңғарбайұлы Әубәкіров — қазақтан шыққан тұңғыш ғарышкер болып табылады. "
            "Ол 1991 жылы «Союз ТМ-13» ғарыш кемесімен ғарышқа сапар шекті. "
            "Байқоңыр — әлемдегі тұңғыш әрі ең ірі ғарыш айлағы."
        )
    }
}


def get_verification_preset_choices() -> List[str]:
    """Returns the ordered list of 4-quadrant verification demonstration titles."""
    return list(VERIFICATION_PRESET_SAMPLES.keys())


def get_verification_preset_text(preset_name: Optional[str]) -> str:
    """Returns the text for the given verification preset name, or empty string if not found."""
    if not preset_name or preset_name not in VERIFICATION_PRESET_SAMPLES:
        return ""
    return VERIFICATION_PRESET_SAMPLES[preset_name].get("text", "")


def get_verification_preset_metadata(preset_name: Optional[str]) -> Dict[str, str]:
    """Returns the metadata dictionary for the given verification preset name."""
    if not preset_name or preset_name not in VERIFICATION_PRESET_SAMPLES:
        return {}
    sample = VERIFICATION_PRESET_SAMPLES[preset_name]
    return {
        "title": sample.get("title", preset_name),
        "quadrant": sample.get("quadrant", ""),
        "description": sample.get("description", "")
    }

