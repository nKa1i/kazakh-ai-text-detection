def build_generation_prompt(seed_entry: dict) -> str:
    """
    Builds an authentic Kazakh review generation prompt matching the seed review's
    length bracket and domain. Designed to elicit natural consumer feedback without
    formal AI disclaimers or repetitive introductory tokens.
    """
    bracket = seed_entry.get("length_bracket", "medium")
    domain = seed_entry.get("domain", "consumer_reviews")
    
    if bracket == "short":
        len_instruction = "Өте қысқа пікір жазыңыз (1-2 сөйлем, 10-15 сөзден аспасын)."
    elif bracket == "medium":
        len_instruction = "Орташа ұзындықтағы шынайы пікір жазыңыз (2-3 сөйлем, 20-35 сөз)."
    else:
        len_instruction = "Толыққанды, егжей-тегжейлі пікір жазыңыз (3-5 сөйлем, 40+ сөз)."

    system_instruction = (
        "Сіз интернет-дүкендегі (Kaspi.kz) нақты сатып алушысыз. "
        "Қазақ тілінде шынайы пікір (review) жазыңыз. "
        "Ешқандай жасанды интеллект кіріспе сөздерінсіз, қарапайым халықтық ауызекі тілде жазыңыз.\n"
        f"Тақырыбы: {domain}. {len_instruction}\n"
        "Пікір:"
    )
    return system_instruction
