# Databricks notebook source
import pyspark.sql.functions as F
import pandas as pd

# COMMAND ----------

repository_path = "file:/Workspace/Repos/giovanichequim@gmail.com/Mastercloud-Trilha-03-Onda-2/src/azure_databricks/Machine Learning/Heart Attack Analysis & Prediction/"
data_path = repository_path + "data/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leitura dos dados

# COMMAND ----------

df = spark.read.table('hive_metastore.default.heart')

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Bronze

# COMMAND ----------

df.write.parquet(data_path+"bronze/heart", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Silver

# COMMAND ----------

bronze_df = spark.read.parquet(data_path+"bronze/heart")

# COMMAND ----------

bronze_df = bronze_df.withColumn('age', F.col('age').cast('int'))
bronze_df = bronze_df.withColumn('sex', F.col('sex').cast('int'))
bronze_df = bronze_df.withColumn('cp', F.col('cp').cast('int'))
bronze_df = bronze_df.withColumn('trtbps', F.col('trtbps').cast('int'))
bronze_df = bronze_df.withColumn('chol', F.col('chol').cast('int'))
bronze_df = bronze_df.withColumn('fbs', F.col('fbs').cast('int'))
bronze_df = bronze_df.withColumn('restecg', F.col('restecg').cast('int'))
bronze_df = bronze_df.withColumn('thalachh', F.col('thalachh').cast('int'))
bronze_df = bronze_df.withColumn('exng', F.col('exng').cast('int'))
bronze_df = bronze_df.withColumn('oldpeak', F.col('oldpeak').cast('float'))
bronze_df = bronze_df.withColumn('slp', F.col('slp').cast('int'))
bronze_df = bronze_df.withColumn('caa', F.col('caa').cast('int'))
bronze_df = bronze_df.withColumn('thall', F.col('thall').cast('int'))
bronze_df = bronze_df.withColumn('output', F.col('output').cast('int'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### O que deveria ser feito na etapa Bronze-Silver:
# MAGIC   - Limpeza e correção de tipos
# MAGIC   - Enriquecimento com possíveis novas variáveis
# MAGIC   - Remoção de valores faltantes
# MAGIC   - Analise exploratória

# COMMAND ----------

bronze_df.write.parquet(data_path+"silver/heart", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Gold

# COMMAND ----------

silver_df = spark.read.parquet(data_path+"silver/heart")

# COMMAND ----------

silver_df.write.parquet(data_path+"gold/heart", mode="overwrite")
