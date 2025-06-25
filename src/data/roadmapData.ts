import { Database, Cloud, Cog, Award } from 'lucide-react';

export const phases = [
  {
    number: 1,
    title: "Foundation Strengthening",
    weeks: "Weeks 1-4",
    description: "Build solid fundamentals in Python, SQL, and cloud basics. Master data manipulation, file handling, and basic cloud services.",
    color: "blue",
    icon: <Database className="h-6 w-6" />,
    weeks_data: [
      {
        title: "Week 1: Python Fundamentals",
        days: [
          "Day 1-2: Variables, data types, control structures, functions and modules, file handling",
          "Day 3-4: Lists, dictionaries, sets, tuples, list comprehensions, error handling", 
          "Day 5-7: pandas (DataFrames, data manipulation), numpy (numerical operations), basic matplotlib/seaborn for visualization"
        ],
        codeExample: `# Basic Python for Data\n# Variables and data types\nname = "Data Engineer"\nage = 25\ndata_types = ['int', 'float', 'str', 'bool']\n\n# List comprehension\nsquares = [x**2 for x in range(10)]\n\n# Function definition\ndef process_data(data):\n    return [item.upper() for item in data if isinstance(item, str)]`,
        language: "python"
      },
      {
        title: "Week 2: Advanced Python for Data",
        days: [
          "Day 1-3: Pandas deep dive - data cleaning, transformation, groupby operations, merging datasets",
          "Day 4-5: File formats - JSON, CSV, Parquet handling, reading from databases, API interactions",
          "Day 6-7: Practice projects - clean messy datasets, build data transformation pipelines"
        ],
        codeExample: `import pandas as pd\nimport numpy as np\n\n# Advanced pandas operations\ndf = pd.read_csv('data.csv')\n\n# Data cleaning\ndf_clean = df.dropna().drop_duplicates()\n\n# Groupby operations\nsummary = df.groupby('category').agg({\n    'sales': ['sum', 'mean'],\n    'quantity': 'count'\n})\n\n# Date handling\ndf['date'] = pd.to_datetime(df['date'])\ndf['month'] = df['date'].dt.month`,
        language: "python"
      },
      {
        title: "Week 3: SQL Enhancement", 
        days: [
          "Day 1-2: Window functions, ROW_NUMBER(), PARTITION BY, ranking functions",
          "Day 3-4: CTEs and subqueries, performance tuning, indexing strategies, query optimization",
          "Day 5-7: Practice with HackerRank SQL challenges and LeetCode database problems"
        ],
        codeExample: `-- Advanced SQL with Window Functions\nSELECT \n    name, \n    salary,\n    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rank,\n    AVG(salary) OVER (PARTITION BY dept) as dept_avg\nFROM employees;\n\n-- CTEs for complex queries\nWITH dept_stats AS (\n    SELECT dept, AVG(salary) as avg_sal, COUNT(*) as emp_count\n    FROM employees \n    GROUP BY dept\n)\nSELECT e.*, d.avg_sal, d.emp_count\nFROM employees e\nJOIN dept_stats d ON e.dept = d.dept;`,
        language: "sql"
      },
      {
        title: "Week 4: Cloud Fundamentals",
        days: [
          "Day 1-3: AWS basics - create account, learn S3, EC2, IAM, AWS CLI setup and commands",
          "Day 4-5: Azure basics - create account, Blob Storage, Resource Groups, Azure CLI",
          "Day 6-7: Choose primary cloud platform based on job market, focus on deeper learning"
        ],
        codeExample: `# AWS CLI Commands\n# Configure AWS CLI\naws configure\n\n# S3 Operations\naws s3 ls\naws s3 cp data.csv s3://my-bucket/data/\naws s3 sync ./local-folder s3://my-bucket/folder/\n\n# EC2 Operations\naws ec2 describe-instances\naws ec2 start-instances --instance-ids i-1234567890abcdef0`,
        language: "bash"
      }
    ]
  },
  {
    number: 2,
    title: "Modern Data Stack",
    weeks: "Weeks 5-8", 
    description: "Learn industry-standard tools like Apache Airflow, Docker, cloud data services, and Databricks. Build production-ready data pipelines.",
    color: "purple",
    icon: <Cog className="h-6 w-6" />,
    weeks_data: [
      {
        title: "Week 5: Apache Airflow",
        days: [
          "Day 1-2: Installation and setup, understand DAGs, operators, tasks",
          "Day 3-4: Building DAGs with PythonOperator, scheduling, dependencies",
          "Day 5-7: Advanced features - sensors, XComs, SubDAGs, error handling, monitoring"
        ],
        codeExample: `from airflow import DAG\nfrom airflow.operators.python_operator import PythonOperator\nfrom datetime import datetime, timedelta\n\ndef extract_data():\n    # Data extraction logic\n    return "Data extracted successfully"\n\ndef transform_data(**context):\n    # Get data from previous task\n    data = context['task_instance'].xcom_pull(task_ids='extract')\n    return f"Transformed: {data}"\n\ndag = DAG(\n    'data_pipeline',\n    default_args={\n        'start_date': datetime(2024, 1, 1),\n        'retries': 1,\n        'retry_delay': timedelta(minutes=5)\n    },\n    schedule_interval='@daily'\n)\n\nextract_task = PythonOperator(\n    task_id='extract',\n    python_callable=extract_data,\n    dag=dag\n)\n\ntransform_task = PythonOperator(\n    task_id='transform', \n    python_callable=transform_data,\n    dag=dag\n)\n\nextract_task >> transform_task`,
        language: "python"
      },
      {
        title: "Week 6: Docker & Containerization",
        days: [
          "Day 1-2: Docker basics - install, learn images, containers, Dockerfile creation",
          "Day 3-4: Docker Compose for multi-container applications, networking between containers",
          "Day 5-7: Practice - containerize data pipelines, push images to Docker Hub"
        ],
        codeExample: `# Dockerfile for Data Pipeline\nFROM python:3.9-slim\n\nWORKDIR /app\n\n# Install dependencies\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\n# Copy source code\nCOPY src/ ./src/\nCOPY config/ ./config/\n\n# Set environment variables\nENV PYTHONPATH=/app\nENV DATA_PATH=/app/data\n\n# Create data directory\nRUN mkdir -p /app/data\n\n# Run the pipeline\nCMD ["python", "src/pipeline.py"]`,
        language: "dockerfile"
      },
      {
        title: "Week 7: Cloud Data Services",
        days: [
          "Day 1-2: AWS S3 deep dive, boto3 for Python, upload/download/manage buckets",
          "Day 3-4: AWS Glue for ETL, crawlers, data catalog, Spark on EMR",
          "Day 5-7: Redshift & Athena for data warehousing, querying S3 data"
        ],
        codeExample: `import boto3\nimport pandas as pd\nfrom io import StringIO\n\n# AWS S3 Operations\ns3_client = boto3.client('s3')\n\n# Upload DataFrame to S3\ndef upload_dataframe_to_s3(df, bucket, key):\n    csv_buffer = StringIO()\n    df.to_csv(csv_buffer, index=False)\n    s3_client.put_object(\n        Bucket=bucket,\n        Key=key,\n        Body=csv_buffer.getvalue()\n    )\n\n# Read from S3 into DataFrame\ndef read_s3_to_dataframe(bucket, key):\n    obj = s3_client.get_object(Bucket=bucket, Key=key)\n    return pd.read_csv(obj['Body'])\n\n# List objects in bucket\nobjects = s3_client.list_objects_v2(Bucket='my-data-bucket')\nfor obj in objects.get('Contents', []):\n    print(obj['Key'])`,
        language: "python"
      },
      {
        title: "Week 8: Databricks Fundamentals", 
        days: [
          "Day 1-2: Databricks setup, community edition signup, notebooks and clusters",
          "Day 3-4: PySpark on Databricks, data transformations, DataFrame operations",
          "Day 5-7: Delta Lake - ACID transactions, time travel, merge operations"
        ],
        codeExample: `from pyspark.sql import SparkSession\nfrom pyspark.sql.functions import col, when, avg, count\nfrom delta.tables import DeltaTable\n\n# Initialize Spark Session\nspark = SparkSession.builder \\\n    .appName("DataPipeline") \\\n    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\\n    .getOrCreate()\n\n# Read data\ndf = spark.read.format("csv") \\\n    .option("header", "true") \\\n    .option("inferSchema", "true") \\\n    .load("/path/to/data.csv")\n\n# Data transformations\ncleaned_df = df.filter(col("age") > 18) \\\n    .withColumn("age_group", \n        when(col("age") < 30, "Young")\n        .when(col("age") < 50, "Middle")\n        .otherwise("Senior")\n    )\n\n# Write to Delta Lake\ncleaned_df.write.format("delta").mode("overwrite").save("/delta/customer_data")\n\n# Delta Lake operations\ndelta_table = DeltaTable.forPath(spark, "/delta/customer_data")\ndelta_table.vacuum()  # Clean up old files`,
        language: "python"
      }
    ]
  },
  {
    number: 3,
    title: "Advanced Topics",
    weeks: "Weeks 9-12",
    description: "Master streaming data processing, ML pipelines, data quality, and infrastructure as code. Learn Kafka, MLflow, and Terraform.",
    color: "green", 
    icon: <Cloud className="h-6 w-6" />,
    weeks_data: [
      {
        title: "Week 9: Streaming Data",
        days: [
          "Day 1-3: Kafka fundamentals - setup local Kafka, producers and consumers, topic management",
          "Day 4-5: Spark Streaming basics, structured streaming, readStream and writeStream",
          "Day 6-7: Real-time processing - window operations, watermarking, output modes"
        ],
        codeExample: `from pyspark.sql import SparkSession\nfrom pyspark.sql.functions import from_json, col, window\nfrom pyspark.sql.types import StructType, StructField, StringType, IntegerType\n\n# Initialize Spark for streaming\nspark = SparkSession.builder \\\n    .appName("StreamingApp") \\\n    .getOrCreate()\n\n# Define schema for incoming data\nschema = StructType([\n    StructField("user_id", StringType(), True),\n    StructField("event_type", StringType(), True),\n    StructField("timestamp", StringType(), True),\n    StructField("value", IntegerType(), True)\n])\n\n# Read from Kafka\ndf = spark.readStream \\\n    .format("kafka") \\\n    .option("kafka.bootstrap.servers", "localhost:9092") \\\n    .option("subscribe", "user_events") \\\n    .load()\n\n# Parse JSON and apply transformations\nparsed_df = df.select(\n    from_json(col("value").cast("string"), schema).alias("data")\n).select("data.*")\n\n# Windowed aggregation\nwindowed_counts = parsed_df \\\n    .withWatermark("timestamp", "10 minutes") \\\n    .groupBy(\n        window(col("timestamp"), "5 minutes"),\n        col("event_type")\n    ).count()\n\n# Write to console\nquery = windowed_counts.writeStream \\\n    .outputMode("update") \\\n    .format("console") \\\n    .trigger(processingTime="30 seconds") \\\n    .start()\n\nquery.awaitTermination()`,
        language: "python"
      },
      {
        title: "Week 10: Machine Learning Pipeline",
        days: [
          "Day 1-2: MLflow setup, experiment tracking, model registry, versioning",
          "Day 3-4: Feature engineering, feature stores, data versioning for ML",
          "Day 5-7: Model deployment - batch predictions, real-time inference, model serving"
        ],
        codeExample: `import mlflow\nimport mlflow.sklearn\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, classification_report\nimport pandas as pd\n\n# Set tracking URI\nmlflow.set_tracking_uri("http://localhost:5000")\nmlflow.set_experiment("customer_churn_prediction")\n\n# Load and prepare data\ndf = pd.read_csv("customer_data.csv")\nX = df.drop(['churn', 'customer_id'], axis=1)\ny = df['churn']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Start MLflow run\nwith mlflow.start_run(run_name="rf_model_v1"):\n    # Log parameters\n    n_estimators = 100\n    max_depth = 10\n    mlflow.log_param("n_estimators", n_estimators)\n    mlflow.log_param("max_depth", max_depth)\n    \n    # Train model\n    model = RandomForestClassifier(\n        n_estimators=n_estimators,\n        max_depth=max_depth,\n        random_state=42\n    )\n    model.fit(X_train, y_train)\n    \n    # Make predictions and log metrics\n    y_pred = model.predict(X_test)\n    accuracy = accuracy_score(y_test, y_pred)\n    mlflow.log_metric("accuracy", accuracy)\n    \n    # Log model\n    mlflow.sklearn.log_model(\n        model, \n        "model",\n        registered_model_name="ChurnPredictor"\n    )\n    \n    print(f"Model accuracy: {accuracy:.4f}")`,
        language: "python"
      },
      {
        title: "Week 11: Data Quality & Governance",
        days: [
          "Day 1-2: Great Expectations - data validation, expectations suites, checkpoints",
          "Day 3-4: Data lineage with Apache Atlas or DataHub, dbt documentation",
          "Day 5-7: Monitoring - data quality checks, pipeline monitoring, alerting systems"
        ],
        codeExample: `import great_expectations as ge\nfrom great_expectations.dataset import PandasDataset\nimport pandas as pd\n\n# Create expectation suite\ndf = pd.read_csv("customer_data.csv")\nge_df = PandasDataset(df)\n\n# Define expectations\nge_df.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)\nge_df.expect_column_to_exist("customer_id")\nge_df.expect_column_values_to_not_be_null("customer_id")\nge_df.expect_column_values_to_be_unique("customer_id")\nge_df.expect_column_values_to_be_in_set("status", ["active", "inactive", "pending"])\nge_df.expect_column_values_to_be_between("age", min_value=18, max_value=120)\n\n# Email validation\nge_df.expect_column_values_to_match_regex(\n    "email", \n    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n)\n\n# Run validation\nresult = ge_df.validate()\n\n# Check if validation passed\nif result.success:\n    print("All data quality checks passed!")\nelse:\n    print("Data quality issues found:")\n    for failure in result.results:\n        if not failure.success:\n            print(f"- {failure.expectation_config.expectation_type}: {failure.result}")`,
        language: "python"
      },
      {
        title: "Week 12: Infrastructure as Code",
        days: [
          "Day 1-3: Terraform basics - HCL syntax, providers, resources, state management",
          "Day 4-5: CI/CD for data pipelines - GitHub Actions, automated testing, deployments",
          "Day 6-7: Project integration - combine all learned technologies, build end-to-end pipeline"
        ],
        codeExample: `# Terraform for AWS Data Infrastructure\n# main.tf\nterraform {\n  required_providers {\n    aws = {\n      source  = "hashicorp/aws"\n      version = "~> 5.0"\n    }\n  }\n}\n\nprovider "aws" {\n  region = var.aws_region\n}\n\n# S3 bucket for data lake\nresource "aws_s3_bucket" "data_lake" {\n  bucket = "\${var.project_name}-data-lake-\${random_string.suffix.result}"\n}\n\nresource "aws_s3_bucket_versioning" "data_lake_versioning" {\n  bucket = aws_s3_bucket.data_lake.id\n  versioning_configuration {\n    status = "Enabled"\n  }\n}\n\n# IAM role for data processing\nresource "aws_iam_role" "data_processing_role" {\n  name = "\${var.project_name}-data-processing-role"\n  \n  assume_role_policy = jsonencode({\n    Version = "2012-10-17"\n    Statement = [\n      {\n        Action = "sts:AssumeRole"\n        Effect = "Allow"\n        Principal = {\n          Service = "glue.amazonaws.com"\n        }\n      }\n    ]\n  })\n}\n\n# Glue job for ETL\nresource "aws_glue_job" "etl_job" {\n  name     = "\${var.project_name}-etl-job"\n  role_arn = aws_iam_role.data_processing_role.arn\n  \n  command {\n    script_location = "s3://\${aws_s3_bucket.data_lake.bucket}/scripts/etl_job.py"\n    python_version  = "3"\n  }\n  \n  default_arguments = {\n    "--TempDir" = "s3://\${aws_s3_bucket.data_lake.bucket}/temp/"\n    "--job-language" = "python"\n  }\n}`,
        language: "terraform"
      }
    ]
  },
  {
    number: 4,
    title: "Specialization & Certification",
    weeks: "Weeks 13-16",
    description: "Choose your specialization path, prepare for certifications, and build a comprehensive portfolio to showcase your skills.",
    color: "orange",
    icon: <Award className="h-6 w-6" />,
    weeks_data: [
      {
        title: "Week 13-14: Choose Specialization",
        days: [
          "Option A: Cloud-Native Data Engineering - AWS Data Analytics Specialty prep, serverless processing",
          "Option B: Real-time Analytics - Advanced Kafka and streaming, event-driven architectures, real-time ML",
          "Option C: Data Architecture - Data mesh principles, system design for data platforms, scalability patterns"
        ]
      },
      {
        title: "Week 15-16: Certification & Portfolio",
        days: [
          "Complete chosen certification exam preparation",
          "Build 2-3 comprehensive end-to-end projects",
          "Create technical blog posts documenting your learning journey",
          "Update LinkedIn profile and resume with new skills",
          "Prepare for data engineering interviews with system design practice"
        ]
      }
    ]
  }
];