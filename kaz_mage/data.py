import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class KazMageSample:
    id: str
    domain: str
    generator: str
    is_unseen_domain: bool
    is_unseen_generator: bool
    quadrant: str
    prefix: str
    text: str
    label: int
    char_length: int
    word_count: int

def load_mage_dataset(path: str) -> List[KazMageSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAGE dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        samples.append(KazMageSample(
            id=str(item.get("id", "")),
            domain=str(item.get("domain", "")),
            generator=str(item.get("generator", "")),
            is_unseen_domain=bool(item.get("is_unseen_domain", False)),
            is_unseen_generator=bool(item.get("is_unseen_generator", False)),
            quadrant=str(item.get("quadrant", "")),
            prefix=str(item.get("prefix", "")),
            text=str(item.get("text", "")),
            label=int(item.get("label", 0)),
            char_length=int(item.get("char_length", len(str(item.get("text", ""))))),
            word_count=int(item.get("word_count", len(str(item.get("text", "")).split())))
        ))
    return samples

def filter_quadrant(dataset: List[KazMageSample], quadrant: str) -> List[KazMageSample]:
    return [s for s in dataset if s.quadrant == quadrant]

def get_quadrant_slices(dataset: List[KazMageSample]) -> Dict[str, List[KazMageSample]]:
    slices = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for s in dataset:
        if s.quadrant in slices:
            slices[s.quadrant].append(s)
    return slices
