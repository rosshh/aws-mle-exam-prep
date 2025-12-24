# AWS Certified Machine Learning Engineer Study Guide

## 📘 Domain 1: Data Preparation for Machine Learning (28%)

### Data Ingestion

Key goals: Collect, store, and preprocess raw data for ML workloads.

| Task | AWS Service | Key Concepts |
|------|------------|--------------|
| Batch data | S3, Glue, DataBrew, Athena | Store structured/unstructured data; catalog with Glue Data Catalog. |
| Streaming data | Kinesis Data Streams / Firehose, MSK (Managed Kafka) | Near real-time ingestion and transformation. |
| Database sources | RDS, Aurora, DynamoDB, Redshift | Use JDBC or connectors with SageMaker Data Wrangler. |
| Automation | Step Functions, Lambda | Event-driven orchestration for ETL jobs. |

**Best Practices:**
- Use Parquet/ORC for analytics (columnar format → faster queries).
- Validate schema with Glue crawlers.
- Store data in S3 buckets with partitioning (e.g., s3://data/year=2025/month=11/).

### Data Cleaning & Transformation

Objective: Ensure data quality, consistency, and usability.

**Common transformations:**
- Handle missing values (fillna, interpolation, deletion).
- Scale numeric data (standardization or normalization).
- Encode categorical data (label encoding, one-hot).
- Detect and remove outliers (Z-score, IQR).
- Transform skewed data with log or Box-Cox transforms.

**Tools:**
- AWS Glue DataBrew – GUI-based cleaning.
- SageMaker Data Wrangler – End-to-end preprocessing pipelines.
- Athena / SQL / PySpark – Custom transformations.

### Feature Engineering

Goal: Create relevant features that improve model performance.

| Technique | Purpose |
|-----------|---------|
| Feature scaling | Ensure uniform influence on model (MinMaxScaler, StandardScaler). |
| Feature selection | Reduce dimensionality using correlation, mutual info, or PCA. |
| Feature creation | Combine or decompose features (ratios, polynomial terms, time lags). |
| Embedding generation | Text (Word2Vec/BERT), Categorical (entity embeddings). |

**Tools:** SageMaker Feature Store (for storing & reusing features).

### Data Quality, Bias & Governance

**Bias Detection:** Sampling bias, label bias, measurement bias.

**Mitigation:** Balanced datasets, weighting, resampling (SMOTE).

**Security:**
- Encrypt data at rest (KMS) and in transit (TLS).
- Use IAM roles for least privilege.
- Track lineage with Glue Catalog and CloudTrail logs.

## 📗 Domain 2: Model Development (26%)

### Algorithm Selection

| Problem Type | Algorithm | AWS Tool |
|--------------|-----------|----------|
| Classification | XGBoost, Linear Learner, BlazingText | SageMaker built-ins |
| Regression | XGBoost, Linear Learner | SageMaker |
| Clustering | K-Means | SageMaker |
| Recommendation | Factorization Machines | SageMaker |
| Time Series | DeepAR Forecasting | SageMaker |
| NLP / Vision | BERT, Comprehend, Rekognition | SageMaker / AI APIs |

**Tips:**
- Structured tabular → XGBoost
- Text → BlazingText or BERT
- Images → CNN or Rekognition
- Sequential/time → RNN/LSTM/DeepAR

### Model Training

**AWS Tools:**
- SageMaker Training Jobs (single or distributed training)
- SageMaker Experiments (track metrics & versions)
- Spot Training – cost optimization.
- Managed Frameworks: TensorFlow, PyTorch, Scikit-learn, MXNet.

**Training Workflow:**
1. Load data from S3.
2. Define estimator and hyperparameters.
3. Launch training job.
4. Output model artifacts to S3.

**Example:**
```python
estimator = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/model-output'
)
estimator.fit({'train': train_path, 'validation': val_path})
```

### Hyperparameter Tuning

**Methods:** Grid Search, Random Search, Bayesian Optimization.

**SageMaker Automatic Model Tuning:** Launches multiple jobs with different parameter sets.

**Key Terms:**
- Objective metric (e.g., validation accuracy)
- Tuning ranges (continuous, integer, categorical)

### Model Evaluation

**Regression Metrics:** RMSE, MAE, R²

**Classification Metrics:** Precision, Recall, F1, AUC

**Ranking Metrics:** NDCG, Precision@K

**Confusion Matrix:**
- TP = correctly predicted positive
- FP = wrongly predicted positive
- FN = missed positive → reduce with recall optimization

**Cross-validation:** Use stratified K-fold for imbalanced datasets.

### Bias & Interpretability

**Use SageMaker Clarify for:**
- Bias detection (pre/post-training)
- Explainability (SHAP values)

**Use SHAP / LIME locally for model transparency.**

## 📙 Domain 3: Deployment and Orchestration of ML Workflows (22%)

### Deployment Options

| Type | Description | AWS Tool |
|------|-------------|----------|
| Real-time | Low-latency prediction via endpoint | SageMaker Endpoint |
| Batch | Large offline predictions | SageMaker Batch Transform |
| Async Inference | Long-running requests | SageMaker Async Endpoint |
| Edge / IoT | Model optimized for devices | SageMaker Neo |
| Multi-model Endpoint | Serve multiple models | SageMaker Multi-Model Endpoint |

**A/B testing:** Deploy multiple versions with production variants.

### Orchestration & Automation

- **SageMaker Pipelines:** Automate steps (ingest → train → evaluate → deploy).
- **Step Functions:** Combine multiple AWS services into workflows.
- **EventBridge:** Trigger retraining or deployment.
- **CI/CD:** Use CodeCommit + CodeBuild + CodePipeline.

**Pipeline Steps:**
Data processing → Training → Evaluation → Model registration → Deployment.

### Infrastructure as Code (IaC)

- **AWS CDK / CloudFormation:** Define reproducible ML infra.
- **ECR:** Store Docker images for custom containers.
- **ECS/EKS:** Containerized training environments.

### Scaling & Optimization

- Use AutoScaling for endpoints (SageMaker).
- Spot instances for cheap training.
- Model compression via SageMaker Neo.

## 📕 Domain 4: Monitoring, Maintenance & Security (24%)

### Monitoring

| Aspect | Tool | Purpose |
|--------|------|---------|
| Data & model drift | SageMaker Model Monitor | Detect input/output drift |
| Logs | CloudWatch Logs | Capture metrics, latency |
| Tracing | X-Ray | End-to-end request tracing |
| Metrics visualization | CloudWatch Dashboards, QuickSight | Trend analysis |

**Retraining strategies:**
- Periodic (e.g., monthly).
- Event-based (new data triggers).

### Maintenance

- Version control with SageMaker Model Registry.
- Use CI/CD retraining pipelines.
- Use Shadow deployments before production rollout.

### Security

**Data Protection:**
- Encrypt with KMS.
- Enforce VPC isolation for endpoints.
- Use PrivateLink for SageMaker Studio.
- Restrict roles with IAM policies (least privilege).

**Compliance:**
- GDPR, HIPAA, SOC, ISO 27001 readiness.
- Monitor via CloudTrail and Access Analyzer.

## Deep Learning & SageMaker Concepts

### Tuning neural networks
- Small batch sizes tend to provide better generalization and avoid local minima
- Large batch sizes can converge faster but may settle on suboptimal solutions
- Large learning rates can overshoot the correct solution or cause divergence
- Small learning rates increase training time but provide more stable convergence

### Model Training, Tuning & Evaluation
- Training: optimize loss with data.
- Tuning: adjust hyperparameters (LR, layers, batch size).
- Evaluation: use validation/test sets + metrics.

### Deep Learning Basics
- Neural networks with many layers learn hierarchical patterns.
- Used in vision, NLP, speech.

### Activation Functions
- ReLU (most common), Leaky ReLU, GELU, SiLU/Swish
- Sigmoid, Tanh (older, prone to vanishing gradients)
- Softmax for multi-class classification output probabilities

### CNNs
- Convolutions extract spatial features (edges → textures → objects).
- Pooling reduces dimension.
- Used for images, vision tasks.

### RNNs
- For sequences (text/time series).
- Maintain hidden state.
- LSTM/GRU solve vanishing gradient issues.

### Neural Network Tuning
- Key hyperparameters: LR, optimizer, depth, width, batch size.
- Methods: grid search, random search, Bayesian tuning.

### Regularization
- Dropout: randomly disable neurons.
- Early stopping: stop when val loss rises.
- L1: sparsity; L2: smooth weights.

### Vanishing/Exploding Gradient Problems
- Vanishing: Gradients become too small in deep networks → slow/no learning
- Exploding: Gradients become too large → unstable training
- Solutions: ReLU family activations, BatchNorm, residual connections, gradient clipping, LSTM/GRU

### Evaluation Metrics
- Confusion Matrix → TP, FP, FN, TN.
- Precision = TP/(TP+FP), Recall = TP/(TP+FN).
- F1 = harmonic mean.
- AUC = ranking quality.
- RMSE/MAE = error size.
- R² = variance explained.

### Ensemble Methods
- Bagging: parallel models (Random Forest).
- Boosting: sequential models reducing bias (XGBoost).

## AWS SageMaker Concepts

### Automatic Model Tuning (AMT)
- Automated hyperparameter optimization via parallel training jobs.

### Hyperparameter Tuning
- Define objective + param ranges → AMT searches best configuration.

### Autopilot (AutoML)
- Auto feature engineering, algorithm selection, tuning, deployment.

### SageMaker Studio & Experiments
- Studio = ML IDE; Experiments = track runs, params, metrics.

### SageMaker Debugger
- Monitors training; detects overfitting, vanishing gradients, dead neurons.

### Model Registry
- Stores, versions, and manages model approval/deployment workflow.

### TensorBoard Integration
- Visualize training metrics, weights, gradients.

### Large-Scale Training
- Training Compiler: faster model training via graph optimization.
- Warm Pools: reuse instances → faster startup.

### Checkpointing & Recovery
- Save training state; auto-restart on node failures; cluster health checks.

### Distributed Training
- Data Parallelism: each worker processes part of batch; gradients synced.
- Model Parallelism: split large models across GPUs (tensor/pipeline division).

### EFA & MiCS
- Elastic Fabric Adapter: high-speed communication for multi-node GPU training.
- MiCS: efficient mixed-precision + optimized communication scaling.

## 🧩 AWS Service Quick Reference

| Category | Service | Purpose |
|----------|---------|---------|
| Data Ingestion | S3, Kinesis, Glue | Store, stream, transform |
| Feature Eng. | Data Wrangler, Feature Store | Transform & share features |
| Model Training | SageMaker | Build, train, tune models |
| Deployment | SageMaker Endpoints, Batch Transform | Serve predictions |
| Monitoring | Model Monitor, CloudWatch | Detect drift, latency |
| Security | IAM, KMS, VPC | Access control, encryption |

## 📏 Key Formulas

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) |
| RMSE | √Σ(y − ŷ)² / n |
| R² | 1 − (SSres / SStot) |

## Summary of AWS tools

### Analytics
- **Amazon Athena:** Serverless SQL query service for analyzing data in S3.
- **Amazon Data Firehose:** Real-time data streaming into storage and analytics tools.
- **Amazon EMR:** Managed big data processing using Hadoop, Spark, and more.
- **AWS Glue:** Serverless data integration and ETL service.
- **AWS Glue DataBrew:** Visual data preparation for analytics and ML.
- **AWS Glue Data Quality:** Monitors and validates data reliability and consistency.
- **Amazon Kinesis:** Real-time data streaming and analytics platform.
- **AWS Lake Formation:** Simplifies building and securing data lakes on AWS.
- **Amazon Managed Service for Apache Flink:** Real-time data processing with Flink.
- **Amazon OpenSearch Service:** Managed search and analytics engine.
- **Amazon QuickSight:** Cloud-scale business intelligence and visualization.
- **Amazon Redshift:** Fully managed data warehouse for large-scale analytics.

### Application Integration
- **Amazon EventBridge:** Event bus for building event-driven applications.
- **Amazon MWAA:** Managed Apache Airflow for workflow orchestration.
- **Amazon SNS:** Pub/sub messaging for event notifications.
- **Amazon SQS:** Fully managed message queuing service.
- **AWS Step Functions:** Workflow automation with visual orchestration of services.

### Cloud Financial Management
- **AWS Billing and Cost Management:** Centralized billing and payment dashboard.
- **AWS Budgets:** Set and track spending limits for AWS usage.
- **AWS Cost Explorer:** Analyze and visualize AWS cost and usage data.

### Compute
- **AWS Batch:** Batch job scheduling and execution at scale.
- **Amazon EC2:** Resizable virtual server instances in the cloud.
- **AWS Lambda:** Run code serverless in response to events.
- **AWS Serverless Application Repository:** Deploy pre-built serverless applications.

### Containers
- **Amazon ECR:** Managed container image registry.
- **Amazon ECS:** Scalable container orchestration service.
- **Amazon EKS:** Managed Kubernetes service for containerized apps.

### Database
- **Amazon DocumentDB:** Managed document database compatible with MongoDB.
- **Amazon DynamoDB:** Serverless NoSQL database for key-value and document data.
- **Amazon ElastiCache:** In-memory caching for high-performance apps.
- **Amazon Neptune:** Managed graph database for connected data.
- **Amazon RDS:** Managed relational database service for multiple engines.

### Developer Tools
- **AWS CDK:** Define cloud infrastructure using familiar programming languages.
- **AWS CodeArtifact:** Secure artifact repository for software dependencies.
- **AWS CodeBuild:** Continuous integration and build automation service.
- **AWS CodeDeploy:** Automate application deployments across environments.
- **AWS CodePipeline:** CI/CD orchestration for rapid release cycles.
- **AWS X-Ray:** Trace and debug distributed applications.

### Machine Learning
- **Amazon A2I:** Human review for AI model predictions.
- **Amazon Bedrock:** Build generative AI applications using foundation models.
- **Amazon CodeGuru:** AI-driven code quality and performance insights.
- **Amazon Comprehend:** Natural language processing for text analysis.
- **Amazon Comprehend Medical:** NLP for extracting medical insights from text.
- **Amazon DevOps Guru:** ML-powered application performance monitoring.
- **Amazon Fraud Detector:** Detect fraudulent activities using ML.
- **AWS HealthLake:** Centralized and structured storage for health data.
- **Amazon Kendra:** Intelligent enterprise search service.
- **Amazon Lex:** Build conversational interfaces with speech and text.
- **Amazon Lookout for Equipment:** Detect equipment anomalies using ML.
- **Amazon Lookout for Metrics:** Identify anomalies in business metrics.
- **Amazon Lookout for Vision:** Detect defects in visual data via ML.
- **Amazon Mechanical Turk:** Crowdsourcing marketplace for human tasks.
- **Amazon Personalize:** Real-time personalized recommendations.
- **Amazon Polly:** Text-to-speech service with lifelike voices.
- **Amazon Q:** Generative AI assistant for enterprise productivity.
- **Amazon Rekognition:** Image and video analysis with deep learning.
- **Amazon SageMaker:** Build, train, and deploy ML models at scale.
- **Amazon Textract:** Extract text and data from documents using AI.
- **Amazon Transcribe:** Speech-to-text transcription service.
- **Amazon Translate:** Neural machine translation for multiple languages.

### Management and Governance
- **AWS Auto Scaling:** Automatically adjust compute capacity based on demand.
- **AWS Chatbot:** Interact with AWS services via chat platforms.
- **AWS CloudFormation:** Infrastructure as code for AWS resources.
- **AWS CloudTrail:** Log and monitor API activity across AWS accounts.
- **Amazon CloudWatch:** Monitor resources and applications in real-time.
- **Amazon CloudWatch Logs:** Collect and analyze log data from AWS resources.
- **AWS Compute Optimizer:** Recommend resource optimization opportunities.
- **AWS Config:** Track configuration changes and compliance.
- **AWS Organizations:** Manage multiple AWS accounts centrally.
- **AWS Service Catalog:** Curate and manage approved AWS products.
- **AWS Systems Manager:** Central management for AWS infrastructure.
- **AWS Trusted Advisor:** Provides best-practice checks for cost, security, and performance.

### Media
- **Amazon Kinesis Video Streams:** Stream video data for analytics and machine learning.

### Migration and Transfer
- **AWS DataSync:** Automate data transfer between on-premises and AWS storage.

### Networking and Content Delivery
- **Amazon API Gateway:** Create and manage APIs at scale.
- **Amazon CloudFront:** Global CDN for fast, secure content delivery.
- **AWS Direct Connect:** Dedicated network link between data centers and AWS.
- **Amazon VPC:** Isolated virtual network environment in the cloud.

### Security, Identity, and Compliance
- **AWS IAM:** Manage user access and permissions securely.
- **AWS KMS:** Create and manage encryption keys for data protection.
- **Amazon Macie:** Discover and protect sensitive data in S3.
- **AWS Secrets Manager:** Securely store and rotate application secrets.

### Storage
- **Amazon EBS:** Block storage for EC2 instances.
- **Amazon EFS:** Scalable file storage for Linux workloads.
- **Amazon FSx:** Managed file systems for Windows and Lustre.
- **Amazon S3:** Object storage for scalable data storage.
- **Amazon S3 Glacier:** Archival storage for long-term data backup.
- **AWS Storage Gateway:** Hybrid cloud storage integration for on-prem systems.

## Built-in Amazon SageMaker algorithms

### 🔢 Supervised Learning
- **Linear Learner** – Fast linear regression/classification optimized for large-scale sparse data.
- **XGBoost** – Gradient-boosted decision trees for high-accuracy tabular prediction tasks – eta.
- **Random Cut Forest (RCF)** – Detects anomalies in unlabeled datasets using unsupervised forest models.
- **K-NN** – Finds nearest neighbors for classification or regression using L2 distance.
- **Factorization Machines** – Performs sparse prediction tasks such as recommendations with interactions.
- **Seq2Seq** – Neural translation and sequence-to-sequence tasks using encoder–decoder architectures.
- **BlazingText** – Fast and scalable word embedding + text classification.
- **Image Classification** – Trains deep CNNs for multi-class image classification.
- **Object Detection** – Detects and localizes objects in images using SSD-based architecture.
- **Semantic Segmentation** – Produces pixel-wise image classifications for segmentation tasks.
- **Neural Topic Model (NTM)** – Discovers topics in large document corpora using variational inference.
- **DeepAR Forecasting** – Probabilistic time-series forecasting using autoregressive RNNs.
- **LDA (Latent Dirichlet Allocation)** – Unsupervised topic modeling for text documents.

### 📊 Unsupervised Learning
- **K-Means** – Clusters unlabeled data by minimizing within-cluster distance.
- **PCA** – Performs dimensionality reduction via principal component transformation.
- **IP Insights** – Detects suspicious entities (e.g., fraudulent IP addresses) using embedding models.

### 🖼️ Computer Vision (Specialized)
- **Object2Vec** – Learns embeddings for paired entities (e.g., queries/products).
- **Reinforcement Learning (RLEstimator)** – Provides frameworks to train RL agents using SageMaker containers.

### 📈 Time Series / Forecasting
- **DeepAR** – Probabilistic multivariate forecasting using autoregressive RNNs.
- **Forecasting with XGBoost containers** – Tabular models adapted for time-series using tree ensembles.

### 💬 NLP (Specialized)
- **BlazingText** – Word2Vec embeddings and text classification at large scale.
- **Seq2Seq** – Neural translation and sequence transformation.
- **NTM** – Topic modeling for unlabeled text.

### 🧪 Anomaly Detection
- **Random Cut Forest** – Detects unusual patterns in numerical or streaming data.
- **IP Insights** – Identifies anomalous entity behaviors like fraud or misuse.