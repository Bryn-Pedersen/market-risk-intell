"""
Export predictions to database for BI tools (Tableau, PowerBI, etc.)

Supports:
- PostgreSQL
- BigQuery
- CSV (already handled by predict.py, but can be used as fallback)
"""
import os
import argparse
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def export_to_postgres(df, connection_string, table_name="risk_predictions", if_exists="append"):
    """
    Export predictions DataFrame to PostgreSQL
    
    Args:
        df: DataFrame with predictions
        connection_string: PostgreSQL connection string
                          e.g., "postgresql://user:password@host:5432/dbname"
        table_name: Name of the table to write to
        if_exists: What to do if table exists - "append", "replace", or "fail"
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("sqlalchemy and psycopg2-binary required for PostgreSQL export. "
                         "Install with: pip install sqlalchemy psycopg2-binary")
    
    engine = create_engine(connection_string)
    
    print(f"Exporting {len(df)} rows to PostgreSQL table '{table_name}'...")
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=5000,
    )
    print(f"✓ Successfully exported to PostgreSQL")


def export_to_bigquery(df, project_id, dataset_id, table_name="risk_predictions", if_exists="append"):
    """
    Export predictions DataFrame to BigQuery
    
    Args:
        df: DataFrame with predictions
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Name of the table to write to
        if_exists: What to do if table exists - "append" or "replace"
    """
    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
    except ImportError:
        raise ImportError("google-cloud-bigquery required for BigQuery export. "
                         "Install with: pip install google-cloud-bigquery")
    
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    
    # Create dataset if it doesn't exist
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        print(f"Creating dataset {dataset_id}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Change as needed
        client.create_dataset(dataset, exists_ok=True)
    
    print(f"Exporting {len(df)} rows to BigQuery table '{table_id}'...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND" if if_exists == "append" else "WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.PARQUET if if_exists == "replace" else None,
    )
    
    if if_exists == "replace":
        # For replace, convert to parquet and upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            with open(tmp.name, "rb") as f:
                job = client.load_table_from_file(f, table_id, job_config=job_config)
        os.unlink(tmp.name)
    else:
        # For append, use pandas to_sql equivalent
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    
    job.result()  # Wait for job to complete
    print(f"✓ Successfully exported to BigQuery")


def main():
    parser = argparse.ArgumentParser(description="Export predictions to database")
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/processed/predictions.csv",
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["postgres", "bigquery"],
        required=True,
        help="Database type"
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        help="PostgreSQL connection string (for postgres) or GCP project ID (for bigquery)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="BigQuery dataset ID (required for bigquery)"
    )
    parser.add_argument(
        "--table",
        type=str,
        default="risk_predictions",
        help="Table name (default: risk_predictions)"
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["append", "replace"],
        default="append",
        help="What to do if table exists (default: append)"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    print(f"Loading predictions from {pred_path}...")
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} rows")
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Export based on database type
    if args.db_type == "postgres":
        if not args.connection_string:
            raise ValueError("--connection-string required for PostgreSQL")
        export_to_postgres(df, args.connection_string, args.table, args.if_exists)
    
    elif args.db_type == "bigquery":
        if not args.connection_string:
            raise ValueError("--connection-string (project ID) required for BigQuery")
        if not args.dataset:
            raise ValueError("--dataset required for BigQuery")
        export_to_bigquery(df, args.connection_string, args.dataset, args.table, args.if_exists)


if __name__ == "__main__":
    main()

