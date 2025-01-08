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

# MAGIC %pip install -qU databricks-feature-engineering mlflow
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "amine_elhelou" # Change This
schema = "ray_gtm_examples"
table = "adult_synthetic_raw"
feature_table_name = "features_synthetic"
labels_table_name = "labels_synthetic"
label="income"
primary_key = "id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create feature table

# COMMAND ----------

# MAGIC %md
# MAGIC For this demo/lab we'll be using an augmented version of the [UCI's Adult Census](https://archive.ics.uci.edu/dataset/2/adult) classification dataset

# COMMAND ----------

# DBTITLE 1,Read synthetic dataset and write as feature and labels tables
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

# Read dataset and remove duplicate keys
adult_synthetic_df = spark.read.table(f"{catalog}.{schema}.{table}").dropDuplicates([primary_key])

# Do this ONCE
try:
  print(f"Creating features and labels tables...")
  fe.create_table(
    name=f"{catalog}.{schema}.{feature_table_name}",
    primary_keys=[primary_key],
    df=adult_synthetic_df.drop(label),
    description="Adult Census feature"
  )

  # Extract ground-truth labels and primary key into separate dataframe
  training_ids_df = adult_synthetic_df.select(primary_key, label)
  training_ids_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{labels_table_name}")
  print(f"... OK!")

except Exception as e:
  print(f"Table {feature_table_name} and {labels_table_name} already exists")

# COMMAND ----------

print(f"Created following catalog/schema/tables and variables:\n catalog = {catalog}\n schema = {schema}\n labels_table_name = {labels_table_name}\n feature_table_name = {feature_table_name}\n label = {label}\n primary_key = {primary_key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load training dataset
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
X, Y = (training_pdf.drop(label, axis=1), training_pdf[label])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Universal preprocessing pipeline

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
# MAGIC ## Define high-level experiments parameters
# MAGIC
# MAGIC Recommended Cluster Size for lab:
# MAGIC * Driver node with 4 cores & min 30GB of RAM
# MAGIC * 2 worker nodes with 4 cores each & min 30GB of RAM

# COMMAND ----------

num_cpu_cores_per_worker = 4 # total cpu to use in each worker node
max_worker_nodes = 2
n_trials = 32 # Number of trials (arbitrary for demo purposes but has to be at least > 30 to see benefits of multinode)
rng_seed = 2024 # Random Number Generation Seed for random states

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define objective/loss function and search space to optimize
# MAGIC **TO-DO**

# COMMAND ----------

import mlflow
...
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class ObjectiveBoosting(object):
  """
  a callable class for implementing the objective function. It takes the training dataset by a constructor's argument
  instead of loading it in each trial execution. This will speed up the execution of each trial
  """
  def __init__(self, X_train_in:pd.DataFrame, Y_train_in:pd.Series):
    """
    X_train_in: features
    Y_train_in: label
    """

    # Create pre-processing pipeline
    self.preprocessor = initialize_preprocessing_pipeline(X_train_in, Y_train_in)

    # Split into training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed)

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_val = X_val
    self.Y_val = Y_val
    
  def __call__(self, trial):
    """
    Wrapper call containing data processing pipeline, training and hyperparameter tuning code.
    The function returns the weighted F1 accuracy metric to maximize in this case.
    """

    # TO-DO 
    
    # Assemble the pipeline
    this_model = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", classifier_obj)])

    # Fit the model
    mlflow.sklearn.autolog(disable=True) # Disable mlflow autologging to avoid logging artifacts for every run
    this_model.fit(self.X_train, self.Y_train)

    # Predict on validation set
    y_val_pred = this_model.predict(self.X_val)

    # Calculate and return F1-Score
    f1_score_binary= f1_score(self.Y_val, y_val_pred, average="binary", pos_label='>50K')

    return f1_score_binary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test locally on driver node

# COMMAND ----------

# DBTITLE 1,Quick test/debug
objective_fn = ObjectiveBoosting(X, Y)
...

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Scaling HPO with Ray Tune
# MAGIC
# MAGIC Ray Tune allows hyperparameter optimization through several key capabilities:
# MAGIC 1. Search space definition: defining a search space for hyperparameters using Ray Tune's API, specifying ranges and distributions for parameters to explore.
# MAGIC 2. Trainable functions: wrapping the model training code in a "trainable" function that Ray Tune can call with different hyperparameter configurations.
# MAGIC 3. Search algorithms: Ray Tune provides various search algorithms like random search, Bayesian optimization, etc.
# MAGIC 4. Parallel execution: running multiple trials (hyperparameter configurations) in parallel across a cluster of machines.
# MAGIC 5. **Early stopping**: Ray Tune supports early stopping of poorly performing trials to save compute resources

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
num_cpu_cores_per_worker = 3 # total cpu to use in each worker node (total_cores - 1 to leave one core for spark)
num_cpus_head_node = 3 # Cores to use in driver node (total_cores - 1)
max_worker_nodes = 2


ray_conf = setup_ray_cluster(
  min_worker_nodes=max_worker_nodes,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_head_node=0,
  num_gpus_worker_node=0
)
os.environ['RAY_ADDRESS'] = ray_conf[0]

# COMMAND ----------

# MAGIC %md
# MAGIC **Ray Tune workflow:**
# MAGIC 1. Define search space by wrapping training code in a trainable function
# MAGIC 2. Configure the tuning process (algorithm, resources, etc.): we'll use `ASHA` as the scheduler.
# MAGIC 3. Launch the tuning job using the `Tuner` class
# MAGIC 4. Analyze results to find the best hyperparameters and fit the best model using the `fe()` client

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define objective/loss function and search space

# COMMAND ----------

import pandas as pd
import mlflow
import os
import ray
from ray import train
from sklearn.metrics import f1_score
...
from typing import Dict, Optional, Any


def objective_ray(config: dict, parent_run_id:str, X_train_in:pd.DataFrame, Y_train_in:pd.Series, experiment_name_in:str, mlflow_db_creds_in:dict):
    """
    Wrapper training/objective function for ray tune
    """

    #TO-DO
        # Fit the model
        # mlflow.sklearn.autolog(disable=True) # Disable mlflow autologging to minimize overhead
        this_model.fit(X_train, Y_train)

        # Predict on validation set
        y_val_pred = this_model.predict(X_val)

        # Calculate and return F1-Score
        f1_score_binary= f1_score(Y_val, y_val_pred, average="binary", pos_label='>50K')

        # Log
        train.report({"f1_score_val": f1_score_binary}) # [OPTIONAL] to view in ray logs
        mlflow.log_metrics({"f1_score_val": f1_score_binary}) # to mlflow


def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space.
    """

    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "LightGBM"]) #, "XGBoost"])

    # Define-by-run allows for conditional search spaces.
    if classifier_name == "LogisticRegression":
        trial.suggest_float("C", 1e-2, 1, log=True)
        trial.suggest_float('tol' , 1e-6 , 1e-3, step=1e-6)
    elif classifier_name == "RandomForest":
        trial.suggest_int("n_estimators", 10, 200, log=True)
        trial.suggest_int("max_depth", 3, 10)
        trial.suggest_int("min_samples_split", 2, 10)
        trial.suggest_int("min_samples_leaf", 1, 10)
    else:
        trial.suggest_int("n_estimators", 10, 200, log=True)
        trial.suggest_int("max_depth", 3, 10)
        trial.suggest_float("learning_rate", 1e-2, 0.9)
        trial.suggest_int("max_bin", 2, 256)
        trial.suggest_int("num_leaves", 2, 256),
        
    # Return all constants in a dictionary.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure ray tune's algo and concurrency
# MAGIC
# MAGIC * TO-DO 
# MAGIC * **TO-DO:** Update mlflow callbacks section

# COMMAND ----------

import warnings
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
from mlflow.utils.databricks_utils import get_databricks_env_vars
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


# Grab and set experiment
mlflow_db_creds = get_databricks_env_vars("databricks")
experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
mlflow.set_experiment(experiment_name)
model_name=f"{catalog}.{schema}.hpo_model_ray_tune_optuna"

# Hold-out Test/Train set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rng_seed)

# Define Optuna search algo
searcher = OptunaSearch(space=define_by_run_func, metric="f1_score_val", mode="max")
algo = ConcurrencyLimiter(searcher,
                          max_concurrent=num_cpu_cores_per_worker*max_worker_nodes+num_cpus_head_node
                        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute

# COMMAND ----------

with mlflow.start_run(run_name ='ray_tune', experiment_id=experiment_id) as parent_run:
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    os.environ.update(mlflow_db_creds)
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            objective_ray,
            parent_run_id = parent_run.info.run_id,
            X_train_in = X_train, Y_train_in = Y_train,
            experiment_name_in=experiment_name,
            mlflow_db_creds_in=mlflow_db_creds),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=n_trials,
            reuse_actors = True # Highly recommended for short training jobs (NOT RECOMMENDED FOR GPU AND LONG TRAINING JOBS)
            )
    )

    multinode_results = tuner.fit()

    # Extract best trial info
    best_model_params = multinode_results.get_best_result(metric="f1_score_val", mode="max", scope='last').config
    best_model_params["random_state"] = rng_seed
    classifier_type = best_model_params.pop('classifier')
    
    # Reproduce best classifier
    if classifier_type  == "LogisticRegression":
        best_model = LogisticRegression(**best_model_params)
    elif classifier_type == "RandomForestClassifier":
        best_model = RandomForestClassifier(**best_model_params)
    elif classifier_type == "LightGBM":
        best_model = LGBMClassifier(force_row_wise=True, verbose=-1, **best_model_params)

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False, silent=True)
    
    # Fit best model and log using FE client in parent run
    preprocessor = initialize_preprocessing_pipeline(X_train, Y_train)
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_model)])
    model_pipeline.fit(X_train, Y_train)

    # Infer output schema
    try:
        output_schema = _infer_schema(Y_train)
    
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
        data=X_train.assign(**{str(label):Y_train}),
        targets=label,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": ">50K" }
    )

    # Log metrics for the test set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(label):Y_test}),
        targets=label,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": ">50K" }
    )

    mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Elapsed time excluding best model fitting, logging and evaluation
rt_trials_pdf = multinode_results.get_dataframe()
print(f"Elapsed time for multinode HPO with ray tune for {n_trials} experiments:: {(rt_trials_pdf['timestamp'].iloc[-1] - rt_trials_pdf['timestamp'].iloc[0] + rt_trials_pdf['time_total_s'].iloc[-1])/60} min")

# COMMAND ----------


