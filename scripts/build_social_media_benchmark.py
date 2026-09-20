# -*- coding: utf-8 -*-
"""
scripts/build_social_media_benchmark.py: Kazakh Social Media Cross-Domain Benchmark Tooling.

Provides tools for:
- Automated PII anonymization (user handles, URLs, Kazakh/Russian phone formats, excessive whitespace).
- Colloquial Kazakh prompt generation with slang, code-switching, informal orthography, and personas.
- Strict schema and linguistic validation for social media benchmark records.
- CLI runner for benchmark generation, dry-run testing, and dataset validation.
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Regex patterns for PII detection and anonymization
URL_REGEX = re.compile(r'(?:https?://|ftp://|www\.|t\.me/)\S+', re.IGNORECASE)
HANDLE_REGEX = re.compile(r'@[A-Za-z0-9_]+')
PHONE_REGEX = re.compile(
    r'(?<!\w)(?:\+7|8)[\s\-]*(?:\(\s*\d{3}\s*\)|\d{3})[\s\-]*(?:\d{3}[\s\-]?\d{2}[\s\-]?\d{2}|\d{3}[\s\-]?\d{4}|\d{7})(?!\d)'
)


def _replace_url(match: re.Match) -> str:
    """Replaces URL preserving trailing sentence punctuation."""
    full = match.group(0)
    trailing = ""
    while full and full[-1] in ".,!?:;)»\"'":
        trailing = full[-1] + trailing
        full = full[:-1]
    return f"[URL]{trailing}"


def anonymize_social_text(text: str) -> str:
    """
    Anonymizes social media text by masking personal information:
    - User handles (e.g. '@username', '@daulet_kz') are replaced with '@user_anon'.
    - URLs (e.g. 'http://...', 'https://...', 't.me/...') are replaced with '[URL]'.
    - Phone numbers (Kazakh/Russian formats e.g. '+7701...', '8707...', '+7(701)...') are replaced with '[PHONE]'.
    - Cleans up excessive whitespace.

    Args:
        text: Raw social media post or comment.

    Returns:
        Anonymized clean text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Mask URLs first to prevent handles/numbers inside URLs from being misidentified
    masked = URL_REGEX.sub(_replace_url, text)

    # 2. Mask user handles
    masked = HANDLE_REGEX.sub('@user_anon', masked)

    # 3. Mask phone numbers
    masked = PHONE_REGEX.sub('[PHONE]', masked)

    # 4. Clean up excessive whitespace
    masked = re.sub(r'\s+', ' ', masked).strip()

    return masked


def generate_social_media_prompt(genre: str, persona: str) -> str:
    """
    Generates structured prompts for LLMs (Qwen-2.5, Llama-3.1, GPT-4o)
    to produce authentic colloquial Kazakh social media posts and comments.

    Args:
        genre: Target platform or genre ('telegram', 'twitter', 'forum', 'post', 'comment').
        persona: User persona ('student', 'consumer', 'tech_enthusiast', 'casual_chat').

    Returns:
        Structured prompt string.
    """
    persona_descriptions = {
        "student": "Студент (университетте оқитын жас, сессия, емтихан, стипендия, жатақхана, студенттік өмір жайлы сөйлейді)",
        "consumer": "Тұтынушы (қызмет көрсету сапасы, баға, дүкендер, жеткізу, тауарлар мен каспи қосымшасы жайлы пікір білдіреді)",
        "tech_enthusiast": "IT/Техно-әуесқой (смартфондар, гаджеттер, жасанды интеллект, бағдарламалау, стартаптар жайлы ой бөліседі)",
        "casual_chat": "Күнделікті еркін әңгімелесуші (өмірлік жағдайлар, достармен әңгіме, көңіл-күй, әзіл-қалжың, күнделікті оқиғалар)"
    }
    persona_desc = persona_descriptions.get(persona.lower(), f"Тұлға бейнесі: {persona}")

    platform_descriptions = {
        "telegram": "Telegram (чат немесе арнадағы еркін жазба/пікір, реакциялар, сұрақтар мен талқылаулар)",
        "twitter": "Twitter / X (қысқа, ұтымды, өткір ойлар, пікір білдіру немесе жаңалықты талқылау)",
        "forum": "Форум (кеңес сұрау, пікірталас, тәжірибе бөлісу, сұрақ-жауап стилі)"
    }
    platform_desc = platform_descriptions.get(genre.lower(), f"Платформа/Жанр: {genre}")

    prompt = f"""Сіз — қазақтілді әлеуметтік желілердің белсенді қолданушысысыз.
Берілген тұлға (persona) мен платформа (genre/platform) негізінде шынайы, ауызекі қазақ тіліндегі жазба немесе пікір жазыңыз.

ПАРАМЕТРЛЕР:
- Платформа/Жанр: {genre} ({platform_desc})
- Тұлға бейнесі (Persona): {persona} ({persona_desc})

МІНДЕТТІ ЛИНГВИСТИКАЛЫҚ ТАЛАПТАР (АУЫЗЕКІ СТИЛЬ):
1. Ауызекі жалғаулар мен сөйлеу қосымшалары (Slang & Conversational suffixes):
   Жазбаңызда табиғи ауызекі жұрнақтар мен демеуліктерді қолданыңыз:
   - `-сың ғой` / `-сің ғой` (мысалы: "білесің ғой", "көрдің ғой")
   - `-ма екен` / `-ме екен` (мысалы: "барсам ба екен", "алсам ба екен")
   - `-шы` / `-ші` (мысалы: "айтшы", "көмектесіңізші")
   - `-ау` / `-ей` (мысалы: "қиын болды-ау", "керемет-ай")

2. Код ауыстыру және неологизмдер (Code-switching & Loanwords):
   Қазақ тілінің грамматикасы орыс және ағылшын тілінен енген сөздерге жалғанатын заманауи сөйлеу үлгілерін қолданыңыз:
   - Мысалдар: `донаттау`, `хайптану`, `краш`, `рилс түсіру`, `баг табу`, `дедлайн жақындау`.

3. Бейресми орфография (Informal orthography):
   Әлеуметтік желілерде жиі кездесетін еркін жазу үлгісін пайдаланыңыз:
   - Пернетақтадағы ыңғайлылық үшін `қ` орнына `к` (мысалы: "калайсындар" / "керек"), `ғ` орнына `г` (мысалы: "гой" / "келдик") әріптерін қолдануға болады.

4. Ұзындық шектеуі (Length constraints):
   - Жазба ұзындығы: 10-нан 120 сөзге дейін болуы шарт.

5. Құпиялылық және PII ережесі:
   - Ешқандай нақты телефон нөмірлерін, жеке мәліметтерді немесе нақты аккаунттарды жазбаңыз. Барлық сілтемелер [URL], телефондар [PHONE], ал пайдаланушы атаулары @user_anon түрінде көрсетілуі тиіс.

Жауапты төмендегідей JSON форматында қайтарыңыз:
{{
  "text": "Ауызекі қазақ тіліндегі жазба (10-120 сөз аралығында)",
  "platform": "{genre}",
  "persona": "{persona}",
  "label": "ai"
}}
"""
    return prompt.strip()


def validate_social_record(record: dict) -> bool:
    """
    Validates schema, token count, platform, label, and PII anonymization of a social media benchmark record.

    Requirements:
        - Required keys: 'id', 'text', 'label', 'platform'.
        - 'label' must be 'human' or 'ai'.
        - 'platform' must be one of: 'telegram', 'twitter', 'forum'.
        - Length in tokens: 10 <= len(text.strip().split()) <= 120.
        - No leaked PII:
          * No raw unmasked user handles (only '@user_anon' allowed).
          * No raw URLs (e.g. 'http://', 'https://', 't.me/').
          * No raw phone numbers (Kazakh/Russian phone formats).

    Args:
        record: Candidate social media record dictionary.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(record, dict):
        return False

    required_keys = {"id", "text", "label", "platform"}
    if not required_keys.issubset(record.keys()):
        return False

    rec_id = record.get("id")
    if not isinstance(rec_id, str) or not rec_id.strip():
        return False

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return False

    label = record.get("label")
    if label not in {"human", "ai"}:
        return False

    platform = record.get("platform")
    if platform not in {"telegram", "twitter", "forum"}:
        return False

    tokens = text.strip().split()
    if not (10 <= len(tokens) <= 120):
        return False

    # Check for raw unmasked URLs
    if re.search(r'(?:https?://|ftp://|www\.|t\.me/)\S+', text, re.IGNORECASE):
        return False

    # Check for raw phone numbers
    if PHONE_REGEX.search(text):
        return False

    # Check for unmasked handles (only '@user_anon' is allowed)
    handles = re.findall(r'@[A-Za-z0-9_]+', text)
    if any(h != "@user_anon" for h in handles):
        return False

    return True


def _get_mock_social_records() -> List[Dict[str, Any]]:
    """Generates a balanced mock dataset for dry-run verification."""
    mock_records = [
        # Human - Telegram
        {
            "id": "soc_mock_0001",
            "text": "Бүгін университетте сессия басталды достар, дайындық қалай болып жатыр? @user_anon айтқан конспектілер шынымен көмектесті ғой, рақмет!",
            "label": "human",
            "platform": "telegram",
            "persona": "student"
        },
        {
            "id": "soc_mock_0002",
            "text": "Мына дүкеннің сервисі нашар екен, менеджерлер дұрыс жауап бермейді. Хабарласу үшін [PHONE] нөмірін берген, бірақ ешкім көтермей қойды ғой.",
            "label": "human",
            "platform": "telegram",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0003",
            "text": "Жаңа ноутбук алдым, программалау үшін өте ыңғайлы екен. Бағасы да тиімді, толық сипаттамасын мына [URL] сілтемеден қарап шығыңыздар достар.",
            "label": "human",
            "platform": "telegram",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0004",
            "text": "Кеш жарық баршаңызға! Бүгін ауа райы керемет болып тұр-ау, кешкі серуенге шығуды ұмытпаңыздар достар, демалыс күндеріңіз сәтті өтсін!",
            "label": "human",
            "platform": "telegram",
            "persona": "casual_chat"
        },
        # Human - Twitter
        {
            "id": "soc_mock_0005",
            "text": "Емтихан сұрақтары қиын болды-ау, бірақ бәріміз жақсы тапсырдық деп ойлаймын. @user_anon досымның көмегі зор болды, бәріне сәттілік тілеймін!",
            "label": "human",
            "platform": "twitter",
            "persona": "student"
        },
        {
            "id": "soc_mock_0006",
            "text": "Каспи арқылы тауарға тапсырыс берген ем, жеткізу уақыты өте жылдам болды. Барлық мәліметтерді [URL] арқылы тексеруге болады екен.",
            "label": "human",
            "platform": "twitter",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0007",
            "text": "Python мен PyTorch кітапханаларын жаңартып жатырмын, жаңа модельді үйрету жылдамдығы екі есе өсті. IT саласындағыларға осы нұсқаны ұсынамын.",
            "label": "human",
            "platform": "twitter",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0008",
            "text": "Бүгінгі күн өте қызықты оқиғаларға толы болды, жаңа достармен танысып, керемет әңгіме құрдық. Бәріңізге тек жақсы көңіл-күй тілеймін!",
            "label": "human",
            "platform": "twitter",
            "persona": "casual_chat"
        },
        # Human - Forum
        {
            "id": "soc_mock_0009",
            "text": "Стипендия түсті ме екен, кім біледі? Каспи қосымшасын қайта-қайта тексеріп отырмыз ғой, жауап жазып жіберіңіздерші достар.",
            "label": "human",
            "platform": "forum",
            "persona": "student"
        },
        {
            "id": "soc_mock_0010",
            "text": "Интернет-дүкеннен тапсырыс берген затым келді, сапасы жақсы екен. Бағасы да қалтаға қонымды, [URL] сілтемесінен өзіңіз де көре аласыз.",
            "label": "human",
            "platform": "forum",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0011",
            "text": "Жасанды интеллект бағытындағы жаңа мақаланы оқып шықтым, өте пайдалы ойлар бар екен. Толық шолуын форумда бөлісемін достар.",
            "label": "human",
            "platform": "forum",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0012",
            "text": "Кешегі футбол матчын көрдіңіздер ме, ойын соңына дейін тартысты өтті ғой! Біздің команда жеңіске жеткеніне қатты қуанып отырмын.",
            "label": "human",
            "platform": "forum",
            "persona": "casual_chat"
        },
        # AI - Telegram
        {
            "id": "soc_mock_0013",
            "text": "Студенттерге арналған оқу бағдарламасы жаңартылды, барлық тапсырмаларды уақытылы орындау қажет. Толық ақпаратты [URL] арқылы біле аласыз.",
            "label": "ai",
            "platform": "telegram",
            "persona": "student"
        },
        {
            "id": "soc_mock_0014",
            "text": "Тұтынушылардың құқығын қорғау бойынша жаңа ережелер бекітілді, кез келген сапасыз тауарды қайтару мүмкіндігі заң жүзінде қарастырылған.",
            "label": "ai",
            "platform": "telegram",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0015",
            "text": "Жасанды интеллект алгоритмдері қазақ тіліндегі мәтіндерді талдау сапасын арттыруда. Бұл бағыттағы зерттеу нәтижелерімен [URL] сайтында танысуға болады.",
            "label": "ai",
            "platform": "telegram",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0016",
            "text": "Күнделікті жоспарды дұрыс құру уақытты тиімді пайдалануға көмектеседі. Әрбір істі жүйелі түрде орындау табысқа жетелейді ғой.",
            "label": "ai",
            "platform": "telegram",
            "persona": "casual_chat"
        },
        # AI - Twitter
        {
            "id": "soc_mock_0017",
            "text": "Сессия кезінде дұрыс тамақтану мен ұйқы режимін сақтау денсаулыққа өте маңызды. Осы қарапайым ережелерді орындасаңыз, шаршамайсыз.",
            "label": "ai",
            "platform": "twitter",
            "persona": "student"
        },
        {
            "id": "soc_mock_0018",
            "text": "Онлайн сауда жасағанда қауіпсіздік ережелерін сақтау қажет, күмәнді сайттарға жеке мәліметтерді енгізбеңіз. Толығырақ [URL] парақшасында жазылған.",
            "label": "ai",
            "platform": "twitter",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0019",
            "text": "Машиналық оқыту модельдерін оңтайландыру үшін жаңа әдістер ұсынылды. Бағдарламалық жасақтаманы жаңартып, код тиімділігін арттыруға болады.",
            "label": "ai",
            "platform": "twitter",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0020",
            "text": "Бос уақытта кітап оқу адамның ой-өрісін кеңейтіп, сөздік қорын байытады. Жақсы шығармалар адамға жаңа күш-қуат сыйлайды.",
            "label": "ai",
            "platform": "twitter",
            "persona": "casual_chat"
        },
        # AI - Forum
        {
            "id": "soc_mock_0021",
            "text": "Университет қабырғасында өтетін ғылыми конференцияға қатысу үшін тіркелу басталды. Барлық сұрақтар бойынша [PHONE] нөміріне хабарласуға болады.",
            "label": "ai",
            "platform": "forum",
            "persona": "student"
        },
        {
            "id": "soc_mock_0022",
            "text": "Қызмет көрсету сапасын жақсарту мақсатында жаңа кері байланыс жүйесі іске қосылды. Пікірлеріңізді [URL] арқылы қалдыра аласыз.",
            "label": "ai",
            "platform": "forum",
            "persona": "consumer"
        },
        {
            "id": "soc_mock_0023",
            "text": "Киберқауіпсіздік шараларын күшейту кез келген ұйым үшін бірінші кезектегі міндет болып табылады. Жүйе қауіпсіздігін үнемі тексеріп отырыңыз.",
            "label": "ai",
            "platform": "forum",
            "persona": "tech_enthusiast"
        },
        {
            "id": "soc_mock_0024",
            "text": "Жағымды жаңалықтармен бөлісу адамдар арасындағы қарым-қатынасты нығайтады, бір-бірімізге әрдайым қолдау көрсетіп жүрейік достар.",
            "label": "ai",
            "platform": "forum",
            "persona": "casual_chat"
        }
    ]
    return mock_records


def validate_file(file_path: str) -> int:
    """
    Validates an existing JSONL social media benchmark file and prints statistics.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        0 if all records are valid, 1 if any errors or invalid records found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    total = 0
    valid_count = 0
    invalid_count = 0
    label_counts: Dict[str, int] = defaultdict(int)
    platform_counts: Dict[str, int] = defaultdict(int)
    token_lengths: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            total += 1
            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSONDecodeError: {e}", file=sys.stderr)
                invalid_count += 1
                continue

            if validate_social_record(rec):
                valid_count += 1
                label_counts[rec["label"]] += 1
                platform_counts[rec["platform"]] += 1
                token_lengths.append(len(rec["text"].split()))
            else:
                invalid_count += 1
                print(f"Line {line_num}: Validation failed for record id='{rec.get('id')}': {rec}", file=sys.stderr)

    print("=" * 60)
    print(f"Social Media Benchmark Validation Report: {file_path}")
    print(f"Total Records: {total}")
    print(f"Valid Records: {valid_count} ({valid_count / max(1, total) * 100:.1f}%)")
    print(f"Invalid Records: {invalid_count}")
    print("-" * 60)
    print("Label Distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt} ({cnt / max(1, valid_count) * 100:.1f}%)")
    print("-" * 60)
    print("Platform Distribution:")
    for plt, cnt in sorted(platform_counts.items()):
        print(f"  {plt}: {cnt} ({cnt / max(1, valid_count) * 100:.1f}%)")
    if token_lengths:
        print("-" * 60)
        print(f"Token Lengths: min={min(token_lengths)}, max={max(token_lengths)}, avg={sum(token_lengths) / len(token_lengths):.1f}")
    print("=" * 60)

    return 0 if (valid_count > 0 and invalid_count == 0) else 1


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for social media cross-domain benchmark tooling."""
    parser = argparse.ArgumentParser(
        description="Kazakh Social Media Cross-Domain Benchmark Tooling"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw social text JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kazakh_social_media_benchmark.jsonl",
        help="Path to output benchmark JSONL (default: data/kazakh_social_media_benchmark.jsonl)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate mock candidate dataset for verification without external APIs"
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate an existing social media benchmark file and print statistics"
    )
    parser.add_argument(
        "--generate-prompts",
        action="store_true",
        help="Generate sample colloquial prompts for different personas and platforms"
    )

    parsed_args = parser.parse_args(args if args is not None else sys.argv[1:])

    if parsed_args.validate:
        return validate_file(parsed_args.validate)

    if parsed_args.dry_run:
        records = _get_mock_social_records()
        out_dir = os.path.dirname(parsed_args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(parsed_args.output, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Dry run complete: Wrote {len(records)} mock social records to {parsed_args.output}")
        return 0

    if parsed_args.generate_prompts:
        personas = ["student", "consumer", "tech_enthusiast", "casual_chat"]
        platforms = ["telegram", "twitter", "forum"]
        for persona in personas:
            for platform in platforms:
                prompt = generate_social_media_prompt(genre=platform, persona=persona)
                print(f"--- Prompt: Platform={platform}, Persona={persona} ---")
                print(prompt)
                print()
        return 0

    if parsed_args.input:
        if not os.path.exists(parsed_args.input):
            print(f"Error: Input file not found: {parsed_args.input}", file=sys.stderr)
            return 1

        total = 0
        written = 0
        out_dir = os.path.dirname(parsed_args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(parsed_args.input, "r", encoding="utf-8") as in_f, \
             open(parsed_args.output, "w", encoding="utf-8") as out_f:
            for line_idx, line in enumerate(in_f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                total += 1
                try:
                    rec = json.loads(line_str)
                except json.JSONDecodeError as e:
                    print(f"Line {line_idx}: JSONDecodeError: {e}", file=sys.stderr)
                    continue

                raw_text = rec.get("text", "")
                anonymized_text = anonymize_social_text(raw_text)

                processed_rec = {
                    "id": rec.get("id", f"soc_{line_idx:05d}"),
                    "text": anonymized_text,
                    "label": rec.get("label", "human"),
                    "platform": rec.get("platform", "telegram"),
                }
                if "persona" in rec:
                    processed_rec["persona"] = rec["persona"]

                if validate_social_record(processed_rec):
                    out_f.write(json.dumps(processed_rec, ensure_ascii=False) + "\n")
                    written += 1

        print(f"Processed {total} input records. Wrote {written} valid benchmark records to {parsed_args.output}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
