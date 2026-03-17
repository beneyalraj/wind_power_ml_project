from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="wind_data_pipeline",
    start_date=datetime(2026, 3, 11),
    schedule_interval=timedelta(days=14), # Runs exactly once every two weeks
    catchup=False,
) as dag:

    # Task 1: Trigger DVC Data Ingestion
    ingest_data = BashOperator(
        task_id="data_ingestion",
        bash_command="cd /opt/project && dvc repro data_ingestion"
    )

    # Task 2: Trigger DVC Raw Validation
    validate_raw_data = BashOperator(
        task_id="validate_raw",
        bash_command="cd /opt/project && dvc repro validate_raw"
    )

    # Task 3: Trigger DVC Parquet Extraction
    extract_features = BashOperator(
        task_id="extract_scenario_dataset",
        bash_command="cd /opt/project && dvc repro extract_scenario_dataset"
    )

    # Task 4: Trigger DVC Processed Validation
    validate_processed_data = BashOperator(
        task_id="validate_processed",
        bash_command="cd /opt/project && dvc repro validate_processed"
    )

    train_model = BashOperator(
    task_id="model_training",
    bash_command="cd /opt/project && dvc repro train_model"
    )

    # Set the execution pipeline order
    ingest_data >> validate_raw_data >> extract_features >> validate_processed_data >> train_model