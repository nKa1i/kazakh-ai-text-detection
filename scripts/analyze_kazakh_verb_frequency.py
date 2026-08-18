import os
import re
import json
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Verbal Suffix Patterns
VERBAL_SUFFIX_PATTERNS = [
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(ғандықтан|гендіктен|қандықтан|кендіктен)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(ған|ген|қан|кен)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(ады|еді|йды|йді)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(мақ|мек|бақ|бек|пақ|пек)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(атын|етін|йтын|йтін)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(дым|дім|тым|тім|дық|дік|тық|тік)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(амын|емін|ймын|ймін)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(аламын|елемін|й аламын)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(ылды|ілді|лды|лді)\b',
    r'[а-яА-ЯөӨүҮұҰіІғҒқҚңҢһҺ]+(арды|ерді|рді|рды)\b'
]

COMBINED_VERB_RE = re.compile('|'.join(VERBAL_SUFFIX_PATTERNS), re.IGNORECASE)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fst_analyzer import analyze_and_segment

def analyze_verb_frequency():
    data_path = os.path.join("data", "kazakh_realworld_test.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    sample_texts = df[text_col].dropna().tolist()
    
    total_words = 0
    total_sentences = len(sample_texts)
    detected_verbs = 0
    verb_sentence_counts = []
    verb_examples = []
    
    for text in sample_texts:
        words = str(text).split()
        if not words:
            continue
        total_words += len(words)
        
        verbs_in_sent = 0
        found_verbs = []
        
        for w in words:
            if COMBINED_VERB_RE.search(w) and len(w) >= 4:
                verbs_in_sent += 1
                found_verbs.append(w)
                
        detected_verbs += verbs_in_sent
        verb_sentence_counts.append(verbs_in_sent)
        
        if verbs_in_sent >= 1 and len(verb_examples) < 5:
            fst_text = analyze_and_segment(text)
            verb_examples.append({
                'original_sentence': text,
                'fst_segmented_sentence': fst_text,
                'detected_verbs': found_verbs
            })
            
    verb_percentage = (detected_verbs / total_words) * 100 if total_words > 0 else 0.0
    avg_verbs_per_sentence = detected_verbs / total_sentences if total_sentences > 0 else 0.0
    sentences_with_verbs = sum(1 for c in verb_sentence_counts if c > 0)
    pct_sentences_with_verbs = (sentences_with_verbs / total_sentences) * 100 if total_sentences > 0 else 0.0
    
    results = {
        'total_sentences_analyzed': total_sentences,
        'total_words_analyzed': total_words,
        'total_verbs_detected': detected_verbs,
        'verb_word_percentage': round(verb_percentage, 2),
        'avg_verbs_per_sentence': round(avg_verbs_per_sentence, 2),
        'sentences_containing_verbs_count': sentences_with_verbs,
        'sentences_containing_verbs_percentage': round(pct_sentences_with_verbs, 2),
        'real_sentence_examples': verb_examples
    }
    
    print("=" * 65)
    print("KAZAKH VERB FREQUENCY & MORPHOLOGICAL DENSITY ANALYSIS")
    print("=" * 65)
    print(f"Total Sentences Analyzed      : {total_sentences:,}")
    print(f"Total Words Scanned          : {total_words:,}")
    print(f"Total Verbs Detected         : {detected_verbs:,}")
    print(f"Verb Word Frequency          : {verb_percentage:.2f}% of all words")
    print(f"Sentences Containing Verbs   : {sentences_with_verbs:,} ({pct_sentences_with_verbs:.2f}%)")
    print(f"Average Verbs per Sentence   : {avg_verbs_per_sentence:.2f} verbs/sentence")
    print("\nReal Dataset Examples:")
    for i, ex in enumerate(verb_examples, 1):
        print(f"\nExample {i}:")
        print(f"  Original : {ex['original_sentence']}")
        print(f"  Verbs    : {ex['detected_verbs']}")
        print(f"  FST Parsed: {ex['fst_segmented_sentence']}")
    print("=" * 65)
    
    out_json = os.path.join("data", "verb_frequency_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved verb frequency summary to: {out_json}")

if __name__ == "__main__":
    analyze_verb_frequency()
