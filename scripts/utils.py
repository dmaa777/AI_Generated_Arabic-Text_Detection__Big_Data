"""
utils.py - Helper functions for Arabic AI-text detection project
Contains common utilities, configuration, and shared functions
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Set, Tuple
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Project Configuration
# ============================================================================

@dataclass
class ProjectPaths:
    """Manages project directory structure."""
    root: str = os.path.abspath("./arabic_ai_detection_f023_f046")

    @property
    def raw(self) -> str:
        return os.path.join(self.root, "data", "raw")
    
    @property
    def processed(self) -> str:
        return os.path.join(self.root, "data", "processed")
    
    @property
    def models(self) -> str:
        return os.path.join(self.root, "models")
    
    @property
    def figures(self) -> str:
        return os.path.join(self.root, "reports", "figures")
    
    @property
    def stream_in(self) -> str:
        return os.path.join(self.root, "stream", "incoming")
    
    @property
    def stream_out(self) -> str:
        return os.path.join(self.root, "stream", "predictions")
    
    @property
    def checkpoint(self) -> str:
        return os.path.join(self.root, "stream", "checkpoint")

    def ensure(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [self.raw, self.processed, self.models, self.figures,
                     self.stream_in, self.stream_out, self.checkpoint]:
            os.makedirs(path, exist_ok=True)


def setup_environment(base_dir: str = "./arabic_ai_detection_f023_f046") -> ProjectPaths:
    """
    Setup project environment and return paths object.
    
    Args:
        base_dir: Root directory for the project
        
    Returns:
        ProjectPaths object with all directory paths
    """
    paths = ProjectPaths(root=os.path.abspath(base_dir))
    paths.ensure()
    return paths


def install_packages(packages: List[str] = None):
    """
    Install required Python libraries if not already present.
    
    Args:
        packages: List of package names to install (uses default if None)
    """
    if packages is None:
        packages = [
            "datasets", "huggingface_hub", "pyarabic", "regex",
            "nltk", "wordcloud", "seaborn", "pyarrow"
        ]
    
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    # Download NLTK resources
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords"
                          else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def configure_matplotlib():
    """Configure matplotlib settings for consistent plots."""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False
    })


# ============================================================================
# Arabic Text Processing Functions
# ============================================================================

# Arabic vowel sets for syllable counting
_LONG_VOWELS = set("اوي")
_SHORT_VOWELS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0652")


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text using pyarabic.araby.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Normalized text
    """
    if text is None:
        return None
    
    import pyarabic.araby as araby
    
    t = araby.strip_tashkeel(text)   # remove diacritics
    t = araby.strip_tatweel(t)       # remove kashida
    t = araby.normalize_hamza(t)     # ء/أ/إ/آ → ا
    t = araby.normalize_alef(t)      # alef variants
    t = araby.normalize_ligature(t)  # لا ligatures
    
    # Keep Arabic letters, ASCII letters/digits, basic punctuation
    t = re.sub(r"[^\u0600-\u06FF\sA-Za-z0-9\.\,\!\?\:\;\n]", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    
    return t


def count_syllables_in_word(word: str) -> int:
    """
    Count syllables in a single Arabic word.
    One syllable = one vowel cluster.
    
    Args:
        word: Arabic word
        
    Returns:
        Number of syllables (minimum 1)
    """
    if not word:
        return 0
    
    n = 0
    in_vowel = False
    
    for ch in word:
        is_vowel = (ch in _LONG_VOWELS) or (ch in _SHORT_VOWELS)
        if is_vowel and not in_vowel:
            n += 1
            in_vowel = True
        elif not is_vowel:
            in_vowel = False
    
    return max(n, 1)


def avg_syllables_per_word(words: List[str]) -> float:
    """
    Calculate average syllables per word (Feature f023).
    
    Args:
        words: List of tokens
        
    Returns:
        Average syllable count per word
    """
    if not words:
        return 0.0
    
    total_syllables = sum(count_syllables_in_word(w) for w in words)
    return float(total_syllables) / float(len(words))


def looks_like_noun_heuristic(word: str) -> bool:
    """
    Heuristic to identify nouns without POS tagger.
    
    Args:
        word: Arabic word
        
    Returns:
        True if word appears to be a noun
    """
    if not word or len(word) < 3:
        return False
    
    _COMMON_VERBS = frozenset("""
        قال يقول قام يقوم ذهب يذهب جاء يأتي كتب يكتب رأى يرى وجد يجد
        أصبح يصبح صار يصير ظل يظل أعطى يعطي شاهد يشاهد سمع يسمع أكل
        يأكل شرب يشرب نام ينام ركض يركض أحب يحب كره يكره عمل يعمل
        فعل يفعل ترك يترك بدأ يبدأ انتهى ينتهي أنشأ ينشئ بنى يبني
    """.split())
    
    _AL_PREFIX = "ال"
    _NOUN_SUFFIXES = ("ة", "ات", "ون", "ين", "ية", "ياً", "يا", "ها", "هم", "هن")
    
    if word in _COMMON_VERBS:
        return False
    if word.startswith(_AL_PREFIX) and len(word) > 3:
        return True
    if word.endswith(_NOUN_SUFFIXES):
        return True
    if len(word) == 3:
        return True
    
    return False


# ============================================================================
# Dataset Utilities
# ============================================================================

class HFLoader:
    """
    Hugging Face dataset loader with CSV caching.
    """
    
    AI_COLS = ("allam_generated_abstract", "jais_generated_abstract",
               "llama_generated_abstract", "openai_generated_abstract")

    def __init__(self, target_dir: str, dataset):
        """
        Initialize loader.
        
        Args:
            target_dir: Directory to save CSV files
            dataset: Loaded Hugging Face dataset
        """
        self.target_dir = target_dir
        self.ds = dataset

    def fetch_all(self) -> List[Tuple[str, str]]:
        """
        Save each split to CSV.
        
        Returns:
            List of (split_name, csv_path) tuples
        """
        out = []
        for split_name, split_data in self.ds.items():
            csv_path = os.path.join(self.target_dir,
                                   f"arabic_abstracts_{split_name}.csv")
            if os.path.exists(csv_path):
                print(f"  [cache] {split_name}: {csv_path}")
            else:
                print(f"  [save]  {split_name}: {csv_path}")
                split_data.to_pandas().to_csv(csv_path, index=False,
                                              encoding="utf-8-sig")
            out.append((split_name, csv_path))
        return out

    def melt(self, csv_path: str, source_name: str) -> pd.DataFrame:
        """
        Flatten a split CSV into long format.
        
        Args:
            csv_path: Path to CSV file
            source_name: Source split name
            
        Returns:
            DataFrame with columns: text, label, model, source
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        rows = []
        
        for _, row in df.iterrows():
            # Human-written (label 0)
            orig = row.get("original_abstract")
            if isinstance(orig, str) and orig.strip():
                rows.append((orig, 0, "human", source_name))
            
            # AI-generated (label 1)
            for col in self.AI_COLS:
                v = row.get(col)
                if isinstance(v, str) and v.strip():
                    rows.append((v, 1, col.replace("_generated_abstract", ""),
                                 source_name))
        
        return pd.DataFrame(rows, columns=["text", "label", "model", "source"])