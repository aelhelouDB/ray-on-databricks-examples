# Databricks notebook source
# MAGIC %pip install dbldatagen -qU
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "amine_elhelou" # "mlops_pj" Change This
schema = "tko_fy25_hpo"
feature_table_name = "features_synthetic"
labels_table_name = "labels_synthetic"
label="income"
primary_key = "id"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T


table_schema = """`age` DOUBLE,
`workclass` STRING,
`fnlwgt` DOUBLE,
`education` STRING,
`education_num` DOUBLE,
`marital_status` STRING,
`occupation` STRING,
`relationship` STRING,
`race` STRING,
`sex` STRING,
`capital_gain` DOUBLE,
`capital_loss` DOUBLE,
`hours_per_week` DOUBLE,
`native_country` STRING,
`income` STRING"""

# Read data and add a unique id column to use as primary key
raw_df = (
  spark.read.csv("/databricks-datasets/adult/adult.data", schema=table_schema)
  .withColumn(primary_key, F.expr("uuid()"))
)

# Trim all strings and replace "?" with None/Null value
clean_df = raw_df
for field in raw_df.schema.fields:
    if isinstance(field.dataType, T.StringType):
        clean_df = clean_df.withColumn(field.name, F.trim(F.col(field.name)))

clean_df = clean_df.na.replace("?", None)

# COMMAND ----------

import dbldatagen as dg


analyzer = dg.DataAnalyzer(sparkSession=spark, df=clean_df)
code =  dg.DataAnalyzer.scriptDataGeneratorFromSchema(clean_df.schema)

# COMMAND ----------

clean_df.summary().display()

# COMMAND ----------

unique_occupations = clean_df.select("native_country").distinct().collect()
country_list = [row.native_country for row in unique_occupations]

# COMMAND ----------

# Column definitions are stubs only - modify to generate correct data  
#
generation_spec = (
    dg.DataGenerator(sparkSession=spark, 
                     name='synthetic_data', 
                     rows=100000,
                     random=True,
                     )
    .withColumn('age', 'double', minValue=17, maxValue=95, step=1)
    .withColumn('workclass', 'string', values=['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Without-pay', 'Never-worked'])
    .withColumn('fnlwgt', 'double', minValue=12285.0, maxValue=1484705.0, step=2)
    .withColumn('education', 'string', values=['Masters', 'HS-grad', 'Bachelors', 'Some-college', 'Assoc-voc', 'HS-grad', 'Masters', 'Bachelors', 'Some-college', 'Prof-school', 'Doctorate', 'Assoc-acdm', 'Preschool', '10th', '12th', '5th-6th', '7th-8th', '11th'])
    .withColumn('education_num', 'double', minValue=1, maxValue=16, step=1)
    .withColumn('marital_status', 'string', values=['Married-spouse-absent','Married-civ-spouse','Divorced','Separated','Never-married', 'Married-AF-spouse'])
    .withColumn('occupation', 'string', values=occupation_list)
    .withColumn('relationship', 'string', values=relation_list)
    .withColumn('race', 'string', values=race_list)
    .withColumn('sex', 'string', values=gender_list)
    .withColumn('capital_gain', 'double', minValue=0.0, maxValue=99999.0, step=100)
    .withColumn('capital_loss', 'double', minValue=0.0, maxValue=5000, step=100)
    .withColumn('hours_per_week', 'double', minValue=1.0, maxValue=100.0, step=1)
    .withColumn('native_country', 'string', values=country_list)
    .withColumn('income', 'string', values=['<=50K', '>50K'])
    )

# COMMAND ----------

df_synthetic_data = generation_spec.build()

# COMMAND ----------

from pyspark.sql import functions as F

df_synthetic_data = df_synthetic_data \
  .withColumn("id", F.expr("uuid()")) \
  .selectExpr("id", "* EXCEPT (id)")

# COMMAND ----------

# df_synthetic_data = fe.score_batch(df=df_synthetic_data, model_uri="models:/amine_elhelou.tko_fy25_hpo.hpo_model_ray_tune_optuna/2", result_type="string")

# COMMAND ----------

df_full = clean_df.union(df_synthetic_data)

# COMMAND ----------

df_full.count()

# COMMAND ----------

df_full.write.format("delta").save(f"/Volumes/{catalog}/{schema}/synthetic-dataset/adult_synthetic")

# COMMAND ----------

df_full.write.format("parquet").save(f"/Volumes/{catalog}/{schema}/synthetic-dataset/adult_synthetic.parquet")
