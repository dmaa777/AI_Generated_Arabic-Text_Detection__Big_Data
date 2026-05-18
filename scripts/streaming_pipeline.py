"""
streaming_pipeline.py - Code for file-based stream simulation and Spark Structured Streaming
Handles real-time inference on streaming text data
"""

import json
import time
import os
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import PipelineModel


# ============================================================================
# Stream Configuration
# ============================================================================

def get_stream_schema() -> T.StructType:
    """
    Define schema for streaming input data.
    
    Returns:
        StructType schema for JSON input
    """
    return T.StructType([
        T.StructField("text", T.StringType(), True),
        T.StructField("label", T.IntegerType(), True),
    ])


# ============================================================================
# Stream Simulation Producer
# ============================================================================

def create_stream_batches(df_raw: DataFrame, incoming_path: str,
                          batch_size: int = 10, num_batches: int = 5,
                          seed: int = 9) -> int:
    """
    Create JSON batch files for stream simulation.
    
    Args:
        df_raw: Source DataFrame with 'text' and 'label' columns
        incoming_path: Directory to write JSON files
        batch_size: Number of records per batch
        num_batches: Number of batches to create
        seed: Random seed for sampling
        
    Returns:
        Number of records written
    """
    os.makedirs(incoming_path, exist_ok=True)
    
    # Clear existing files
    for f in os.listdir(incoming_path):
        os.remove(os.path.join(incoming_path, f))
    
    # Sample records
    sample = df_raw.sample(False, 0.005, seed=seed).limit(batch_size * num_batches).toPandas()
    
    for i, row in sample.iterrows():
        with open(os.path.join(incoming_path, f"abs_{i:03d}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump({
                "text": row["text"],
                "label": int(row["label"])
            }, fh, ensure_ascii=False)
    
    print(f"Wrote {len(sample)} JSON records to {incoming_path}")
    return len(sample)


# ============================================================================
# Stream Processing Consumer
# ============================================================================

def create_streaming_dataframe(spark: SparkSession, source_path: str,
                               max_files_per_trigger: int = 5) -> DataFrame:
    """
    Create a streaming DataFrame from a file source.
    
    Args:
        spark: SparkSession
        source_path: Path to directory containing JSON files
        max_files_per_trigger: Maximum files to process per trigger
        
    Returns:
        Streaming DataFrame
    """
    schema = get_stream_schema()
    
    stream_df = (spark.readStream
                 .schema(schema)
                 .option("maxFilesPerTrigger", max_files_per_trigger)
                 .json(source_path))
    
    return stream_df


def apply_stream_processing(stream_df: DataFrame,
                           preprocess_udf, tokenize_udf,
                           f023_udf, f046_udf,
                           prep_pipeline: PipelineModel,
                           model,
                           feature_f023: str, feature_f046: str,
                           min_tokens: int = 5) -> DataFrame:
    """
    Apply preprocessing and inference to streaming data.
    
    Args:
        stream_df: Streaming DataFrame
        preprocess_udf, tokenize_udf: Preprocessing UDFs
        f023_udf, f046_udf: Feature UDFs
        prep_pipeline: Fitted feature pipeline
        model: Trained classification model
        feature_f023, feature_f046: Feature column names
        min_tokens: Minimum tokens required
        
    Returns:
        Streaming DataFrame with predictions
    """
    # Clean and preprocess
    stream_clean = (stream_df
        .withColumn("clean_text_columns", preprocess_udf(F.col("text")))
        .withColumn("tokens", tokenize_udf(F.col("clean_text_columns")))
        .withColumn("n_tokens", F.size("tokens"))
        .filter(F.col("n_tokens") >= min_tokens)
        .withColumn(feature_f023, f023_udf(F.col("tokens")))
        .withColumn(feature_f046, f046_udf(F.col("tokens"))))
    
    # Apply feature pipeline
    stream_features = prep_pipeline.transform(stream_clean)
    
    # Apply model
    stream_predictions = (model.transform(stream_features)
                          .select("text", "label", "prediction"))
    
    return stream_predictions


# ============================================================================
# Stream Output Sinks
# ============================================================================

def write_stream_to_json(stream_df: DataFrame, output_path: str,
                         checkpoint_path: str,
                         trigger_interval: str = "3 seconds"):
    """
    Write streaming results to JSON files.
    
    Args:
        stream_df: Streaming DataFrame
        output_path: Output directory path
        checkpoint_path: Checkpoint directory path
        trigger_interval: Processing trigger interval
        
    Returns:
        StreamingQuery object
    """
    query = (stream_df.writeStream
             .outputMode("append")
             .format("json")
             .option("path", output_path)
             .option("checkpointLocation", checkpoint_path)
             .trigger(processingTime=trigger_interval)
             .start())
    
    return query


def write_stream_to_console(stream_df: DataFrame,
                            trigger_interval: str = "3 seconds"):
    """
    Write streaming results to console for debugging.
    
    Args:
        stream_df: Streaming DataFrame
        trigger_interval: Processing trigger interval
        
    Returns:
        StreamingQuery object
    """
    query = (stream_df.writeStream
             .outputMode("append")
             .format("console")
             .trigger(processingTime=trigger_interval)
             .start())
    
    return query


# ============================================================================
# Stream Evaluation
# ============================================================================

def evaluate_stream_output(output_path: str, spark: SparkSession) -> Tuple[DataFrame, float]:
    """
    Evaluate streaming output accuracy.
    
    Args:
        output_path: Path to streaming output JSON files
        spark: SparkSession
        
    Returns:
        Tuple of (DataFrame with results, accuracy)
    """
    import glob
    
    pred_files = glob.glob(os.path.join(output_path, "*.json"))
    print(f"{len(pred_files)} JSON parts written")
    
    if pred_files:
        sdf = spark.read.json(pred_files)
        sdf.show(5, truncate=70)
        
        pdf = sdf.select("label", "prediction").toPandas()
        accuracy = (pdf["label"] == pdf["prediction"]).mean()
        print(f"Streaming-mode accuracy on simulated batch: {accuracy:.4f}")
        
        return sdf, accuracy
    
    return None, 0.0


# ============================================================================
# Main Stream Execution
# ============================================================================

def run_streaming_pipeline(spark: SparkSession, paths, prep_pipeline, model,
                          preprocess_udf, tokenize_udf,
                          f023_udf,