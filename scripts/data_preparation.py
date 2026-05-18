"""
data_preparation.py - Distributed functions for data cleaning and feature engineering
Handles Spark DataFrame transformations, Arabic text preprocessing, and feature extraction
"""

import os
from typing import Tuple, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, VectorAssembler, StandardScaler

from utils import (
    normalize_arabic, avg_syllables_per_word, looks_like_noun_heuristic,
    HFLoader
)


# ============================================================================
# Arabic Preprocessor (Spark UDFs)
# ============================================================================

class ArabicPreprocessor:
    """
    Wraps pyarabic + nltk and exposes Spark UDFs for distributed processing.
    """
    
    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize Arabic text."""
        return normalize_arabic(text)

    @staticmethod
    def _tokenize(text: str, stopwords_set: set) -> list:
        """Tokenize and filter stopwords."""
        if not text:
            return []
        
        import pyarabic.araby as araby
        toks = araby.tokenize(text)
        return [w for w in toks if len(w) > 1 and w not in stopwords_set and not w.isspace()]

    @classmethod
    def create_udfs(cls, spark_context, stopwords_set: set) -> Tuple:
        """
        Create Spark UDFs for Arabic text processing.
        
        Args:
            spark_context: SparkContext for broadcasting
            stopwords_set: Set of Arabic stopwords
            
        Returns:
            Tuple of (normalize_udf, tokenize_udf)
        """
        sw_bc = spark_context.broadcast(stopwords_set)
        
        normalize_udf = F.udf(cls._normalize, T.StringType())
        
        def _tokenize_with_stopwords(text):
            return cls._tokenize(text, sw_bc.value)
        
        tokenize_udf = F.udf(_tokenize_with_stopwords, T.ArrayType(T.StringType()))
        
        return normalize_udf, tokenize_udf


def create_stylometric_udfs(use_camel: bool = False):
    """
    Create Spark UDFs for stylometric features f023 and f046.
    
    Args:
        use_camel: Whether to use camel-tools for POS tagging
        
    Returns:
        Tuple of (f023_udf, f046_udf, camel_available_flag)
    """
    # f023 - Average syllables per word
    f023_udf = F.udf(lambda ws: avg_syllables_per_word(ws), T.DoubleType())
    
    # Setup camel-tools if requested
    camel_tagger = None
    camel_available = False
    
    if use_camel:
        try:
            from camel_tools.tagger.default import DefaultTagger
            from camel_tools.disambig.mle import MLEDisambiguator
            
            disamb = MLEDisambiguator.pretrained()
            camel_tagger = DefaultTagger(disamb, "pos")
            camel_available = True
            print("camel-tools POS tagger ready.")
        except Exception as e:
            print(f"camel-tools setup failed ({e}) — using pyarabic fallback.")
    
    def count_nouns(words):
        """Count nouns using camel-tools or heuristic."""
        if not words:
            return 0
        
        if camel_available and camel_tagger is not None:
            try:
                tags = camel_tagger.tag(list(words))
                return sum(1 for t in tags if t and t.lower().startswith("noun"))
            except Exception:
                pass
        
        # Fallback heuristic
        return sum(1 for w in words if looks_like_noun_heuristic(w))
    
    f046_udf = F.udf(count_nouns, T.IntegerType())
    
    return f023_udf, f046_udf, camel_available


def clean_dataframe(df: DataFrame, normalize_udf, tokenize_udf,
                    min_tokens: int = 5) -> DataFrame:
    """
    Apply cleaning and preprocessing to the input DataFrame.
    
    Args:
        df: Input Spark DataFrame with 'text' column
        normalize_udf: UDF for Arabic normalization
        tokenize_udf: UDF for tokenization
        min_tokens: Minimum tokens required to keep a row
        
    Returns:
        Cleaned DataFrame with cleaned text and tokens
    """
    df_clean = (
        df
        .withColumn("clean_text_columns", normalize_udf(F.col("text")))
        .withColumn("tokens", tokenize_udf(F.col("clean_text_columns")))
        .withColumn("n_tokens", F.size("tokens"))
        .filter(F.col("n_tokens") >= min_tokens)
    )
    
    return df_clean.cache()


def add_stylometric_features(df: DataFrame, f023_udf, f046_udf,
                             feature_name_f023: str, feature_name_f046: str) -> DataFrame:
    """
    Add stylometric features f023 and f046 to the DataFrame.
    
    Args:
        df: DataFrame with 'tokens' column
        f023_udf: UDF for syllable counting
        f046_udf: UDF for noun counting
        feature_name_f023: Name for f023 feature column
        feature_name_f046: Name for f046 feature column
        
    Returns:
        DataFrame with added feature columns
    """
    df_feat = (
        df
        .withColumn(feature_name_f023, f023_udf(F.col("tokens")))
        .withColumn(feature_name_f046, f046_udf(F.col("tokens")))
    ).cache()
    
    return df_feat


def save_parquet_partitioned(df: DataFrame, output_path: str, partition_col: str = "label"):
    """
    Save DataFrame to Parquet with partitioning.
    
    Args:
        df: Spark DataFrame to save
        output_path: Destination path
        partition_col: Column to partition by
    """
    (df
     .repartition(8, partition_col)
     .write
     .mode("overwrite")
     .partitionBy(partition_col)
     .parquet(output_path))


def load_parquet(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load DataFrame from Parquet format.
    
    Args:
        spark: SparkSession
        input_path: Source path
        
    Returns:
        Loaded Spark DataFrame
    """
    return spark.read.parquet(input_path).cache()


def load_and_prepare_data(spark: SparkSession, paths: 'ProjectPaths') -> DataFrame:
    """
    Complete data loading and preprocessing pipeline.
    
    Args:
        spark: SparkSession
        paths: ProjectPaths object with directory configuration
        
    Returns:
        Prepared DataFrame with 'features' column
    """
    from datasets import load_dataset
    
    # Load dataset from Hugging Face
    dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
    
    # Save and flatten
    loader = HFLoader(paths.raw, dataset)
    csvs = loader.fetch_all()
    long_pd = pd.concat([loader.melt(p, n) for n, p in csvs], ignore_index=True)
    
    # Create Spark DataFrame
    schema = T.StructType([
        T.StructField("text", T.StringType(), False),
        T.StructField("label", T.IntegerType(), False),
        T.StructField("model", T.StringType(), False),
        T.StructField("source", T.StringType(), False),
    ])
    df = spark.createDataFrame(long_pd, schema=schema).cache()
    
    return df