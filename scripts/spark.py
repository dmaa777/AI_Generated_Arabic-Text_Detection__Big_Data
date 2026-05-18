import os
import sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk"  
os.environ["SPARK_HOME"] = "/home/centos/spark"        

sys.path.insert(0, os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.insert(0, os.path.join(os.environ["SPARK_HOME"], "python/lib/py4j-src.zip"))

from pyspark.sql import SparkSession
from pyspark import SparkConf