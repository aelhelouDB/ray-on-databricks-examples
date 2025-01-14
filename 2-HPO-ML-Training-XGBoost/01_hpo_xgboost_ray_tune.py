# Databricks notebook source
# MAGIC %md
# MAGIC # HPO for Traditional ML/Boosting algo
# MAGIC
# MAGIC In this example we'll cover the new recommended ways for performing distributed hyperparameter optimization for boosting algorithms (e.g. xgboost) on databricks, while also leveraging MLflow for tracking experiments and the feature engineering client to ensure lineage between models and feature tables.
# MAGIC
# MAGIC We'll use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) on top of [ray on spark](https://docs.databricks.com/en/machine-learning/ray/index.html) and leverage specific [early stopping]() callbacks.
# MAGIC
# MAGIC **WORK-IN-PROGRESS**

# COMMAND ----------

# MAGIC %pip install -qU databricks-feature-engineering mlflow optuna
# MAGIC
# MAGIC %pip install ucimlrepo
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "jon_cheung" # Change This
schema = "ray_gtm_examples"
table = "adult_synthetic_raw"
feature_table_name = "features_synthetic"
labels_table_name = "labels_synthetic"
label="income"
primary_key = "id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Create feature table

# COMMAND ----------

# MAGIC %md
# MAGIC For this demo/lab we'll be using an augmented version of the [UCI's Adult Census](https://archive.ics.uci.edu/dataset/2/adult) classification dataset

# COMMAND ----------

import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
X = adult.data.features 
y = adult.data.targets 
y['income'].replace({"<=50K": 0, "<=50K.": 0,
                     ">50K": 1, ">50K.": 1}, inplace=True)

# build sdf 
adult_synthetic_df = pd.concat([X, y], axis=1)
adult_synthetic_df.reset_index(drop=False, inplace=True, names='id')
adult_synthetic_sdf = spark.createDataFrame(adult_synthetic_df)

fe = FeatureEngineeringClient()

# Read dataset and remove duplicate keys
# adult_synthetic_df = spark.read.table(f"{catalog}.{schema}.{table}").dropDuplicates([primary_key])

# Do this ONCE
try:
  print(f"Creating features and labels tables...")
  fe.create_table(
    name=f"{catalog}.{schema}.{feature_table_name}",
    primary_keys=[primary_key],
    df=adult_synthetic_sdf.drop(label),
    description="Adult Census feature"
  )

  # Extract ground-truth labels and primary key into separate dataframe
  training_ids_df = adult_synthetic_sdf.select(primary_key, label)
  training_ids_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{labels_table_name}")
  print(f"... OK!")

# will need to patch this exception
except Exception as e:
  print(f"Table {feature_table_name} and {labels_table_name} already exists")

# COMMAND ----------

print(f"Created following catalog/schema/tables and variables:\n catalog = {catalog}\n schema = {schema}\n labels_table_name = {labels_table_name}\n feature_table_name = {feature_table_name}\n label = {label}\n primary_key = {primary_key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load training dataset
# MAGIC
# MAGIC First create feature lookups and training dataset specs

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup


feature_lookups = [
  FeatureLookup(
    table_name=f"{catalog}.{schema}.{feature_table_name}",
    lookup_key=primary_key
  )
]

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Read labels and ids
training_ids_df = spark.table(f"{catalog}.{schema}.{labels_table_name}")

fe = FeatureEngineeringClient()
# Create training set
training_set = fe.create_training_set(
    df=training_ids_df,
    feature_lookups=feature_lookups,
    label=label,
    exclude_columns=[primary_key]
  )

# Get raw features and label
training_pdf = training_set.load_df().toPandas()
X, y = (training_pdf.drop(label, axis=1), training_pdf[label])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Universal preprocessing pipeline

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def initialize_preprocessing_pipeline(X_train_in:pd.DataFrame, Y_train_in:pd.Series):
  """
  Helper function to create pre-processing pipeline
  """
  # Universal preprocessing pipeline
  categorical_cols = [col for col in X_train_in if X_train_in[col].dtype == "object"]
  numerical_cols = [col for col in X_train_in if X_train_in[col].dtype != "object"]
  cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')),("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
  num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='median'))])                        

  preprocessor = ColumnTransformer(
    [
      ("cat", cat_pipeline, categorical_cols),
      ("num", num_pipeline, numerical_cols)
    ],
    remainder="passthrough",
    sparse_threshold=0
  )
  
  return preprocessor

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Optional: Train one pipeline

# COMMAND ----------

import xgboost as xgb
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Get raw features and label
training_pdf = training_set.load_df().toPandas()
X, y = (training_pdf.drop(label, axis=1), training_pdf[label])

# XGB params
params = {"objective": "binary:logistic",
          "n_estimators": 100,
          "max_depth": 8, 
          "learning_rate": .01,
          "max_bin": 256}

# Pipeline
preprocessor = initialize_preprocessing_pipeline(X, y)
xgb_pipeline = Pipeline(steps=[("preprocessor", preprocessor), 
                              ("classifier", xgb.XGBClassifier(**params))])

# Create validation splits from the training data
train_X, val_X, train_y, val_y = train_test_split(X, 
                                                    y,
                                                      test_size=0.2,
                                                      random_state=42)

# Disable mlflow autologging to avoid logging artifacts for every run
mlflow.sklearn.autolog(disable=True) 

# Train one XGB model
xgb_pipeline.fit(train_X, train_y)

# Predict on validation set
y_val_pred = xgb_pipeline.predict(val_X)

# Calculate and return validation F1-Score
f1_score_binary= f1_score(val_y, y_val_pred, average="binary", pos_label=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scaling HPO with Ray Tune
# MAGIC
# MAGIC Ray Tune allows hyperparameter optimization through several key capabilities:
# MAGIC 1. **Search space definition**: defining a search space for hyperparameters using Ray Tune's API, specifying ranges and distributions for parameters to explore.
# MAGIC 2. **Trainable functions**: wrapping the model training code in a "trainable" function that Ray Tune can call with different hyperparameter configurations.
# MAGIC 3. **Search algorithms**: Ray Tune provides search algorithms like grid, random, and Bayesian. Ray Tune also integrates well with other frameworks like Optuna and Hyperopt. 
# MAGIC 4. **Parallel execution**: running multiple trials (hyperparameter configurations) in parallel across a cluster of machines.
# MAGIC 5. **Early stopping**: Ray Tune supports early stopping of poorly performing trials to save compute resources
# MAGIC
# MAGIC To get started with Ray, we'll set up a Ray cluster and then Ray Tune workflow, encapsulating all the above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Setting up your Ray cluster. 
# MAGIC
# MAGIC Recommended Cluster Size for lab:
# MAGIC * Driver node with 4 cores & min 30GB of RAM
# MAGIC * 2 worker nodes with 4 cores each & min 30GB of RAM

# COMMAND ----------

num_cpu_cores_per_worker = 4 # total cpu to use in each worker node
max_worker_nodes = 2
# Define high-level experiments parameters
n_trials = 32 # Number of trials (arbitrary for demo purposes but has to be at least > 30 to see benefits of multinode)
rng_seed = 2024 # Random Number Generation Seed for random states

# COMMAND ----------

# DBTITLE 1,Initialize Ray on spark cluster
import os
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

# Cluster cleanup
restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass

  try:
    ray.shutdown()
  except:
    pass

# Set configs based on your cluster size
num_cpu_cores_per_worker = 16 # total cpu to use in each worker node (total_cores - 1 to leave one core for spark)
num_cpus_head_node = 8 # Cores to use in driver node (total_cores - 1)
max_worker_nodes = 2

ray_conf = setup_ray_cluster(
  min_worker_nodes=max_worker_nodes,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node=num_cpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_head_node=0,
  num_gpus_worker_node=0
)
os.environ['RAY_ADDRESS'] = ray_conf[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Ray Tune workflow:
# MAGIC 1. Write your trainable (i.e. objective) function --> `trainable_with_resources`
# MAGIC 2. Configure your search space and algorithm --> `searcher_with_concurrency`
# MAGIC 3. Run an HPO search using `Ray Tune` + final training pipeline. This will:
# MAGIC     * a. Launch the tuning job using the `Tuner` class, searching across `n_trials`
# MAGIC     * b. Analyze results for best hyperparameters and then fit, log, and register the final model.

# COMMAND ----------

# DBTITLE 1,Write your trainable function and configure resources
import pandas as pd
import mlflow
import os
import ray
from ray import train
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Any
from ray import tune


def xgb_ray_trainable(config: dict, parent_run_id:str, X_train_in:pd.DataFrame, Y_train_in:pd.Series, experiment_name_in:str, mlflow_db_creds_in:dict):
    """
    Wrapper training/objective function for ray tune
    """
    
    # Set mlflow credentials and active MLflow experiment for each Ray task
    os.environ.update(mlflow_db_creds_in)
    mlflow.set_experiment(experiment_name_in)
    
    # Pipeline
    preprocessor = initialize_preprocessing_pipeline(X_train_in, Y_train_in)
    xgb_pipeline = Pipeline(steps=[("preprocessor", preprocessor), 
                                ("classifier", xgb.XGBClassifier(**config))])
    
    # Create validation splits from the training data
    train_X, val_X, train_y, val_y = train_test_split(X_train_in, 
                                                        Y_train_in,
                                                         test_size=0.2,
                                                          random_state=rng_seed)
    
    # Disable mlflow autologging to avoid logging artifacts for every child run
    mlflow.sklearn.autolog(disable=True) 
    # Start a nested child run for HPO tuning
    with mlflow.start_run(run_name="hpo_run",
                          parent_run_id=parent_run_id) as child_run:

        xgb_pipeline.fit(train_X, train_y)

        # Predict on validation set
        y_val_pred = xgb_pipeline.predict(val_X)

        # Calculate and return F1-Score
        f1_score_binary= f1_score(val_y, y_val_pred, average="binary", pos_label=1)
        # Log
        mlflow.log_metrics({"f1_score_val": f1_score_binary}) # to mlfl ow
     
    train.report({"f1_score_val": f1_score_binary}) # [OPTIONAL] to view in ray logs

# XGBoost benefits from multi-threading (i.e. leveraging parallel threads to speed up the construction of weak learners). Here you can allocate the number of CPUs to use per XGboost model. We suggest setting this to the number of CPUs equal to or less than the number of CPUs available on the worker (i.e. <16 in this tutorial)
trainable_with_resources = tune.with_resources(xgb_ray_trainable, 
                                               {"cpu": 8})


# COMMAND ----------

# DBTITLE 1,Configure your search space and algorithm
import warnings
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
from mlflow.utils.databricks_utils import get_databricks_env_vars
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


# The search space sampling can be defined by 
# 1) a "define by run function" (see below) which uses Optuna's native sampling OR
# 2) a dictionary with a different sampling function (i.e. Ray Tune's sampler)
def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space. Trial suggestions automatically return a dictionary that's passed to the trainable function.
    """
    trial.suggest_categorical("objective", ["binary:logistic"])
    trial.suggest_int("n_estimators", 10, 200, log=True)
    trial.suggest_int("max_depth", 3, 10)
    trial.suggest_float("learning_rate", 1e-2, 0.9)
    trial.suggest_int("max_bin", 2, 256)

# Define Optuna search algo with concurrency limiter.
# Note: A concurrency limiter, limits the number of parallel trials. This is important for Bayesian search (inherently sequential) as too many parallel trials reduces the benefits of priors to inform the next search round.
searcher = OptunaSearch(space=define_by_run_func, metric="f1_score_val", mode="max")
# This is the object of our search algorithm
searcher_with_concurrency = ConcurrencyLimiter(searcher,
                                               max_concurrent=num_cpu_cores_per_worker*max_worker_nodes+num_cpus_head_node
                        )


# COMMAND ----------

# DBTITLE 1,Run HPO search + train final model
# Define your experiment name
experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(experiment_name)

# Since we want to create nested child runs to organize our HPO experiment, we need to pass in the mlflow credentials to each Ray actor.
mlflow_db_creds = get_databricks_env_vars("databricks")
model_name=f"{catalog}.{schema}.hpo_model_ray_tune_optuna"

# Load the X, y dataset for training
X, y = (training_pdf.drop(label, axis=1), training_pdf[label])

# Start the parent mlflow run that will also house the best model 
with mlflow.start_run(run_name ='ray_tune') as parent_run:
    # Run our Tuner job
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            trainable_with_resources,
            parent_run_id = parent_run.info.run_id,
            X_train_in = X, Y_train_in = y,
            experiment_name_in=experiment_name,
            mlflow_db_creds_in=mlflow_db_creds),
        tune_config=tune.TuneConfig(
            search_alg=searcher_with_concurrency,
            num_samples=n_trials,
            reuse_actors = True # Highly recommended for short training jobs (NOT RECOMMENDED FOR GPU AND LONG TRAINING JOBS)
            )
    )

    multinode_results = tuner.fit()

    # Extract best trial info
    best_model_params = multinode_results.get_best_result(metric="f1_score_val",
                                                           mode="max",
                                                            scope='last').config
    best_model_params["random_state"] = rng_seed
    
    # Reproduce best classifier
    best_model = xgb.XGBClassifier(**best_model_params)

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(log_input_examples=True,
                            log_models=False,
                             silent=True)
    
    # Fit best model and log using FE client in parent run.
    # Note that since this is our final model, we will train using all the data
    preprocessor = initialize_preprocessing_pipeline(X, y)
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                      ("classifier", best_model)])
    model_pipeline.fit(X, y)

    # Infer output schema
    try:
        output_schema = _infer_schema(y)
    
    except Exception as e:
        warnings.warn(f"Could not infer model output schema: {e}")
        output_schema = None
    
    fe.log_model(
        model=model_pipeline,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        output_schema=output_schema,
        registered_model_name=model_name
    )

    # Evaluate model and log into experiment
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)
    
    # Log metrics for the training set
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X.assign(**{str(label):y}),
        targets=label,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": 1}
    )

    # # Log metrics for the test set
    # val_eval_result = mlflow.evaluate(
    #     model=pyfunc_model,
    #     data=X_test.assign(**{str(label):Y_test}),
    #     targets=label,
    #     model_type="classifier",
    #     evaluator_config = {"log_model_explainability": False,
    #                         "metric_prefix": "test_" , "pos_label": ">50K" }
    # )

    mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Elapsed time excluding best model fitting, logging and evaluation
rt_trials_pdf = multinode_results.get_dataframe()
print(f"Elapsed time for multinode HPO with ray tune for {n_trials} experiments:: {(rt_trials_pdf['timestamp'].iloc[-1] - rt_trials_pdf['timestamp'].iloc[0] + rt_trials_pdf['time_total_s'].iloc[-1])/60} min")
