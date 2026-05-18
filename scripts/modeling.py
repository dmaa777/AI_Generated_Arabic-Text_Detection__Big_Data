"""
modeling.py - Spark MLlib code for building, training, and evaluating models
Handles feature pipeline construction, model training, and evaluation
"""

import time
from typing import Dict, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, NaiveBayes, RandomForestClassifier, LinearSVC,
    LogisticRegressionModel, NaiveBayesModel,
    RandomForestClassificationModel, LinearSVCModel
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator
)


# ============================================================================
# Feature Pipeline
# ============================================================================

class FeaturePipelineBuilder:
    """
    Builds TF-IDF + stylometric feature pipeline.
    """
    
    def __init__(self, tf_dim: int = 4096):
        """
        Initialize pipeline builder.
        
        Args:
            tf_dim: Number of features for HashingTF
        """
        self.tf_dim = tf_dim

    def build(self, feature_f023: str, feature_f046: str) -> Pipeline:
        """
        Build the complete feature engineering pipeline.
        
        Args:
            feature_f023: Name of f023 feature column
            feature_f046: Name of f046 feature column
            
        Returns:
            Spark ML Pipeline object
        """
        tf = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=self.tf_dim)
        idf = IDF(inputCol="tf", outputCol="tfidf")
        
        stylo_assembler = VectorAssembler(
            inputCols=[feature_f023, feature_f046, "n_tokens"],
            outputCol="stylo_raw"
        )
        
        scaler = StandardScaler(
            inputCol="stylo_raw",
            outputCol="stylo",
            withMean=False,
            withStd=True
        )
        
        final_assembler = VectorAssembler(
            inputCols=["tfidf", "stylo"],
            outputCol="features"
        )
        
        return Pipeline(stages=[tf, idf, stylo_assembler, scaler, final_assembler])


def split_data(df: DataFrame, train_ratio: float = 0.70,
               val_ratio: float = 0.15, test_ratio: float = 0.15,
               seed: int = 2026) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        df: DataFrame with 'features' and 'label' columns
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df, temp_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = temp_df.randomSplit([val_relative, 1 - val_relative], seed=seed)
    
    return train_df.cache(), val_df.cache(), test_df.cache()


# ============================================================================
# Model Arena
# ============================================================================

class ModelArena:
    """
    Trains, evaluates and ranks multiple Spark MLlib classifiers.
    """
    
    # Default model configurations
    MODEL_CONFIGS = {
        "NaiveBayes": {
            "class": NaiveBayes,
            "params": {"smoothing": 1.0, "modelType": "multinomial"}
        },
        "LogReg": {
            "class": LogisticRegression,
            "params": {"maxIter": 30, "regParam": 0.01}
        },
        "RandomForest": {
            "class": RandomForestClassifier,
            "params": {"numTrees": 80, "maxDepth": 10, "seed": 2026}
        },
        "LinearSVC": {
            "class": LinearSVC,
            "params": {"maxIter": 30, "regParam": 0.05}
        }
    }

    def __init__(self, train_df: DataFrame, val_df: DataFrame):
        """
        Initialize ModelArena.
        
        Args:
            train_df: Training DataFrame with 'features' and 'label' columns
            val_df: Validation DataFrame with 'features' and 'label' columns
        """
        self.train = train_df
        self.val = val_df
        self.results = {}
        self.fitted = {}

    def _evaluate(self, model) -> Dict:
        """
        Evaluate a trained model on validation data.
        
        Args:
            model: Trained Spark ML model
            
        Returns:
            Dictionary with accuracy, f1, auc metrics
        """
        predictions = model.transform(self.val)
        
        acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(metricName="f1")
        auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
        
        accuracy = acc_eval.evaluate(predictions)
        f1_score = f1_eval.evaluate(predictions)
        
        try:
            auc = auc_eval.evaluate(predictions)
        except Exception:
            auc = float("nan")
        
        return {"acc": accuracy, "f1": f1_score, "auc": auc}

    def add(self, name: str, estimator) -> None:
        """
        Add and train a model.
        
        Args:
            name: Model name identifier
            estimator: Unfitted Spark ML estimator
        """
        t0 = time.time()
        model = estimator.fit(self.train)
        scores = self._evaluate(model)
        scores["train_sec"] = round(time.time() - t0, 2)
        
        self.results[name] = scores
        self.fitted[name] = model
        
        print(f"{name:14s}  acc={scores['acc']:.4f}  f1={scores['f1']:.4f}  "
              f"auc={scores['auc']:.4f}  ({scores['train_sec']}s)")

    def add_all(self) -> None:
        """Add and train all configured models."""
        for name, config in self.MODEL_CONFIGS.items():
            estimator = config["class"](**config["params"])
            self.add(name, estimator)

    def best(self, key: str = "f1") -> Tuple[str, object]:
        """
        Select the best model based on a metric.
        
        Args:
            key: Metric to optimize ('acc', 'f1', 'auc')
            
        Returns:
            Tuple of (model_name, trained_model)
        """
        best_name = max(self.results, key=lambda k: self.results[k][key])
        return best_name, self.fitted[best_name]

    def summary(self) -> 'pd.DataFrame':
        """
        Get summary DataFrame of all results.
        
        Returns:
            Pandas DataFrame with model results
        """
        import pandas as pd
        
        df = (pd.DataFrame(self.results).T
              .reset_index()
              .rename(columns={"index": "model"})
              .round(4)
              .sort_values("f1", ascending=False))
        return df


# ============================================================================
# Model Persistence
# ============================================================================

def save_pipeline(pipeline_model: Pipeline, pipeline_path: str) -> None:
    """
    Save feature pipeline to disk.
    
    Args:
        pipeline_model: Fitted PipelineModel
        pipeline_path: Destination path
    """
    import shutil
    shutil.rmtree(pipeline_path, ignore_errors=True)
    pipeline_model.write().overwrite().save(pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")


def save_model(model, model_path: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained Spark ML model
        model_path: Destination path
    """
    import shutil
    shutil.rmtree(model_path, ignore_errors=True)
    model.write().overwrite().save(model_path)
    print(f"Model saved to: {model_path}")


def load_pipeline(pipeline_path: str) -> Pipeline:
    """
    Load saved pipeline from disk.
    
    Args:
        pipeline_path: Path to saved pipeline
        
    Returns:
        Loaded PipelineModel
    """
    from pyspark.ml import PipelineModel
    return PipelineModel.load(pipeline_path)


def load_model(model_path: str):
    """
    Load saved model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model (type determined from saved metadata)
    """
    # Try each model type
    try:
        return LogisticRegressionModel.load(model_path)
    except Exception:
        pass
    
    try:
        return NaiveBayesModel.load(model_path)
    except Exception:
        pass
    
    try:
        return RandomForestClassificationModel.load(model_path)
    except Exception:
        pass
    
    try:
        return LinearSVCModel.load(model_path)
    except Exception:
        pass
    
    raise ValueError(f"Cannot load model from {model_path}")


# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_on_test(model, test_df: DataFrame, model_name: str = "") -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained Spark ML model
        test_df: Test DataFrame with 'features' and 'label' columns
        model_name: Name prefix for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.transform(test_df)
    
    acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(metricName="f1")
    auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
    
    accuracy = acc_eval.evaluate(predictions)
    f1_score = f1_eval.evaluate(predictions)
    auc = auc_eval.evaluate(predictions)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "auc": round(auc, 4),
        "predictions": predictions
    }
    
    if model_name:
        print(f"TEST · {model_name}: acc={accuracy:.4f} · f1={f1_score:.4f} · auc={auc:.4f}")
    
    return metrics


def get_confusion_matrix(predictions: DataFrame) -> 'pd.DataFrame':
    """
    Generate confusion matrix from model predictions.
    
    Args:
        predictions: DataFrame with 'label' and 'prediction' columns
        
    Returns:
        Pandas DataFrame with confusion matrix counts
    """
    cm = (predictions
          .groupBy("label", "prediction")
          .count()
          .toPandas()
          .pivot(index="label", columns="prediction", values="count")
          .fillna(0)
          .astype(int))
    
    return cm


def get_feature_importance(rf_model, feature_names: list) -> Dict[str, float]:
    """
    Extract feature importance from a trained Random Forest model.
    
    Args:
        rf_model: Trained RandomForestClassificationModel
        feature_names: List of feature names in order
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not isinstance(rf_model, RandomForestClassificationModel):
        raise ValueError("Feature importance is only available for RandomForest models")
    
    importances = rf_model.featureImportances.toArray()
    
    # For this pipeline, stylometric features are the last 3 dimensions
    stylo_importance = importances[-3:]
    
    return dict(zip(feature_names, [round(float(x), 5) for x in stylo_importance]))