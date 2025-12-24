# AWS Certified Machine Learning Engineer - Study Notes

## Domain 1: Data Preparation for ML (28%)

### 1.1 Ingest & Store Data

**Data Formats & Ingestion**
- **Parquet**: Columnar, best compression, analytics workloads
- **ORC**: Optimized Row Columnar, Hive integration
- **Avro**: Schema evolution, streaming data
- **CSV**: Simple, human-readable, larger file sizes
- **Validated vs Non-validated**: Schema enforcement vs flexible ingestion

**AWS Data Sources**
- **S3**: Object storage, data lakes, lifecycle policies
- **EFS**: Network file system, concurrent access
- **FSx for NetApp ONTAP**: High-performance, enterprise features
- **Kinesis Data Streams**: Real-time streaming ingestion
- **Kinesis Data Firehose**: Managed delivery to destinations
- **Kafka/Flink**: Open-source streaming platforms

**Storage Trade-offs**
- **Cost**: S3 Standard > S3 IA > S3 Glacier > S3 Glacier Deep Archive
- **Performance**: EBS PIOPS > EBS GP3 > EFS > S3
- **Durability**: All AWS storage designed for 99.999999999% (11 9's)

**Data Extraction Features**
- **S3 Transfer Acceleration**: CloudFront edge locations for faster uploads
- **EBS PIOPS**: Provisioned IOPS for consistent performance
- **Multi-part uploads**: Large file handling
- **S3 Select**: Query data without downloading entire objects

**Integration Tools**
- **AWS Glue**: Serverless ETL, data catalog
- **Spark on EMR**: Big data processing
- **SageMaker Data Wrangler**: Visual data prep
- **SageMaker Feature Store**: Centralized feature management

### 1.2 Transform Data & Feature Engineering

**Data Cleaning**
- **Outlier handling**: Statistical methods, domain knowledge
- **Missing value imputation**: Mean/median/mode, forward fill, interpolation
- **Deduplication**: Exact matches, fuzzy matching
- **Data combination**: Joins, unions, concatenation

**Feature Engineering Techniques**
- **Scaling/Standardization**: MinMax, StandardScaler, RobustScaler
- **Normalization**: L1, L2 normalization
- **Binning**: Equal width, equal frequency, custom bins
- **Log transforms**: Reduce skewness, handle exponential relationships

**Encoding Methods**
- **One-hot encoding**: Categorical with no order
- **Label encoding**: Ordinal categories
- **Binary encoding**: Reduce dimensionality vs one-hot
- **Tokenization**: Text to numerical representation

**AWS Tools**
- **SageMaker Data Wrangler**: 300+ transforms, visual interface
- **AWS Glue DataBrew**: Visual data preparation
- **Lambda**: Real-time stream processing
- **EMR/Spark**: Large-scale transformations

**Data Annotation Services**
- **SageMaker Ground Truth**: Human labeling workflows
- **Mechanical Turk**: Crowdsourced labeling
- **Ground Truth Plus**: Managed labeling teams

### 1.3 Data Integrity & Modeling Preparation

**Pre-training Bias Detection**
- **Class imbalance**: Unequal sample distribution
- **Difference in proportions**: Statistical measures
- **Demographic parity**: Equal outcomes across groups
- **Equalized odds**: Equal TPR/FPR across groups

**Bias Mitigation**
- **Resampling**: Oversampling minority, undersampling majority
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Synthetic data generation**: GAN-based approaches
- **Stratified sampling**: Maintain class proportions

**Data Security & Compliance**
- **Encryption at rest**: KMS, S3 default encryption
- **Encryption in transit**: TLS, HTTPS
- **Data classification**: Sensitive, confidential, public
- **PII/PHI handling**: Anonymization, pseudonymization, masking
- **Data residency**: Regional compliance requirements

**Data Quality & Validation**
- **AWS Glue Data Quality**: Automated quality checks
- **SageMaker Clarify**: Bias and explainability analysis
- **Data profiling**: Statistical summaries, distributions
- **Schema validation**: Type checking, format verification

**Training Data Preparation**
- **Data splits**: Train/validation/test (70/15/15 typical)
- **Stratification**: Maintain class distribution
- **Cross-validation**: K-fold, time series splits
- **Data augmentation**: Increase dataset diversity
- **EFS/FSx configuration**: High-throughput training data access

---

## Domain 2: ML Model Development (26%)

### 2.1 Choose Modeling Approach

**Algorithm Selection by Problem Type**
- **Classification**: Logistic Regression, Random Forest, SVM, Neural Networks
- **Regression**: Linear Regression, Random Forest, Gradient Boosting
- **Clustering**: K-Means, Hierarchical, DBSCAN
- **Time Series**: ARIMA, Prophet, LSTM
- **Recommendation**: Collaborative filtering, Matrix factorization
- **NLP**: Transformers, BERT, RNNs

**Interpretability Considerations**
- **High interpretability**: Linear models, Decision Trees
- **Medium interpretability**: Random Forest, Gradient Boosting
- **Low interpretability**: Neural Networks, Deep Learning
- **SHAP values**: Model-agnostic explanations
- **LIME**: Local interpretable explanations

**AI Services vs Custom Models**
- **Use AI Services when**: Standard use cases, quick deployment, no ML expertise
- **Amazon Translate**: Text translation
- **Amazon Transcribe**: Speech-to-text
- **Amazon Rekognition**: Image/video analysis
- **Amazon Bedrock**: Foundation models, generative AI
- **Use Custom Models when**: Specialized requirements, proprietary data, performance optimization

**SageMaker Built-in Algorithms**
- **XGBoost**: Gradient boosting, tabular data
- **Linear Learner**: Classification/regression, large datasets
- **Factorization Machines**: Sparse data, recommendations
- **DeepAR**: Time series forecasting
- **BlazingText**: Text classification, word embeddings
- **Image Classification**: CNN-based image analysis

**Cost Considerations**
- **Training costs**: Instance types, training time
- **Inference costs**: Real-time vs batch, throughput requirements
- **Storage costs**: Model artifacts, training data
- **Spot instances**: Up to 90% savings for fault-tolerant workloads

### 2.2 Train & Refine Models

**Training Process Elements**
- **Epochs**: Full passes through training data
- **Batch size**: Samples processed before gradient update
- **Learning rate**: Step size for gradient descent
- **Early stopping**: Prevent overfitting, save time

**Training Optimization**
- **Distributed training**: Data parallelism, model parallelism, pipeline parallelism
- **Mixed precision**: FP16/BF16/FP32 for faster training
- **Gradient checkpointing**: Memory optimization
- **Pipe mode**: Stream data from S3 during training

**Model Performance Improvement**
- **Regularization**: L1 (Lasso), L2 (Ridge), Elastic Net
- **Dropout**: Randomly disable neurons during training
- **Batch normalization**: Stabilize training, faster convergence
- **Weight decay**: L2 penalty on weights

**Hyperparameter Tuning**
- **Grid search**: Exhaustive search, computationally expensive
- **Random search**: Sample random combinations
- **Bayesian optimization**: Use prior results to guide search
- **SageMaker Automatic Model Tuning**: Managed HPO service

**Key Hyperparameters by Algorithm**
- **Neural Networks**: Learning rate, hidden layers, dropout rate
- **Random Forest**: Number of trees, max depth, min samples split
- **SVM**: C parameter, kernel type, gamma
- **Gradient Boosting**: Learning rate, max depth, number of estimators

**Model Integration & Frameworks**
- **Bring Your Own Container (BYOC)**: Custom environments
- **SageMaker Script Mode**: TensorFlow, PyTorch, scikit-learn
- **Pre-trained models**: Transfer learning, fine-tuning
- **SageMaker JumpStart**: Pre-built solutions, foundation models

**Advanced Techniques**
- **Ensemble methods**: Bagging, boosting, stacking
- **Model compression**: Pruning, quantization, distillation
- **Catastrophic forgetting**: Elastic Weight Consolidation, rehearsal
- **Model versioning**: SageMaker Model Registry

### 2.3 Analyze Model Performance

**Evaluation Metrics**
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Regression**: MSE, RMSE, MAE, R²
- **Multi-class**: Macro/Micro averaging, confusion matrix
- **Ranking**: NDCG, MAP, MRR

**Performance Analysis**
- **Baseline models**: Simple heuristics, previous models
- **Learning curves**: Training vs validation performance
- **Overfitting**: High training accuracy, low validation accuracy
- **Underfitting**: Low training and validation accuracy
- **Convergence**: Loss stabilization, gradient norms

**AWS Analysis Tools**
- **SageMaker Clarify**: Bias detection, feature importance, SHAP
- **SageMaker Model Debugger**: Training metrics, tensor analysis
- **SageMaker Experiments**: Track and compare experiments
- **A/B testing**: Shadow variants, production comparisons

**Model Debugging**
- **Vanishing gradients**: Activation functions, initialization
- **Exploding gradients**: Gradient clipping, learning rate adjustment
- **Data leakage**: Future information in training data
- **Distribution shift**: Training vs production data differences

---

## Domain 3: Deployment & Orchestration (22%)

### 3.1 Select Deployment Infrastructure

**Deployment Patterns**
- **Real-time inference**: Low latency, synchronous
- **Batch inference**: High throughput, asynchronous
- **Edge deployment**: Local processing, reduced latency
- **Hybrid**: Combination based on use case

**SageMaker Endpoint Types**
- **Real-time endpoints**: Always-on, <100ms latency
- **Serverless endpoints**: Pay-per-use, automatic scaling
- **Asynchronous endpoints**: Long-running inference, queued requests
- **Batch Transform**: Batch processing, cost-effective

**Compute Selection**
- **CPU instances**: General purpose, cost-effective
- **GPU instances**: Deep learning, parallel processing
- **Inf1/Inf2 instances**: AWS Inferentia chips, cost-optimized inference
- **Trainium instances**: AWS Trainium chips for cost-effective training
- **Multi-model endpoints**: Host multiple models on single endpoint

**Container Strategies**
- **Pre-built containers**: AWS-provided, framework-specific
- **Custom containers**: Bring Your Own Container (BYOC)
- **SageMaker Neo**: Model optimization for edge devices
- **Multi-framework support**: TensorFlow, PyTorch, scikit-learn

**Orchestration Tools**
- **SageMaker Pipelines**: Native ML workflows
- **Apache Airflow (MWAA)**: Complex DAGs, external integrations
- **AWS Step Functions**: Serverless workflows, error handling
- **Lambda**: Event-driven, lightweight processing

**Deployment Targets**
- **SageMaker endpoints**: Managed inference
- **Amazon EKS**: Kubernetes-based, container orchestration
- **Amazon ECS**: Docker containers, AWS-managed
- **AWS Lambda**: Serverless, event-driven
- **Edge devices**: IoT, mobile applications

### 3.2 Create & Script Infrastructure

**Infrastructure as Code (IaC)**
- **AWS CloudFormation**: Native AWS IaC, JSON/YAML templates
- **AWS CDK**: Programming language-based IaC
- **Terraform**: Multi-cloud IaC tool
- **SageMaker Projects**: Pre-configured MLOps templates

**Scaling & Resource Management**
- **Auto Scaling**: Target tracking, step scaling, scheduled scaling
- **Spot instances**: Cost savings, fault-tolerant workloads
- **Reserved instances**: Predictable workloads, cost optimization
- **SageMaker Savings Plans**: Flexible pricing for SageMaker usage

**Containerization**
- **Amazon ECR**: Container registry, vulnerability scanning
- **Docker best practices**: Multi-stage builds, minimal base images
- **Container optimization**: Layer caching, image size reduction
- **Security**: Scan for vulnerabilities, least privilege access

**Network & Security Configuration**
- **VPC deployment**: Private subnets, security groups
- **NAT Gateway**: Outbound internet access from private subnets
- **VPC endpoints**: Private connectivity to AWS services
- **Network ACLs**: Subnet-level security

**Endpoint Configuration**
- **Instance types**: Performance vs cost optimization
- **Auto-scaling metrics**: Invocations per instance, CPU utilization
- **Data capture**: Model monitoring, debugging
- **Multi-AZ deployment**: High availability, fault tolerance

### 3.3 CI/CD for ML

**AWS Developer Tools**
- **CodeCommit**: Git repositories, source control
- **CodeBuild**: Build and test automation
- **CodeDeploy**: Application deployment automation
- **CodePipeline**: End-to-end CI/CD orchestration

**Git Workflows**
- **GitFlow**: Feature branches, release branches
- **GitHub Flow**: Simple, continuous deployment
- **Trunk-based**: Short-lived branches, frequent integration
- **Branch protection**: Required reviews, status checks

**Deployment Strategies**
- **Blue/Green**: Zero downtime, instant rollback
- **Canary**: Gradual traffic shift, risk mitigation
- **Rolling**: Sequential instance replacement
- **Linear**: Time-based traffic shifting

**Automated Testing**
- **Unit tests**: Individual component testing
- **Integration tests**: Component interaction testing
- **End-to-end tests**: Full system validation
- **Model validation**: Performance regression testing

**ML-Specific CI/CD**
- **Data validation**: Schema checks, statistical tests
- **Model validation**: Performance thresholds, bias checks
- **Automated retraining**: Triggered by performance degradation
- **A/B testing**: Automated traffic splitting and monitoring

---

## Domain 4: ML Solution Monitoring, Maintenance & Security (24%)

### 4.1 Monitor Model Inference

**Drift Detection**
- **Data drift**: Input feature distribution changes
- **Model drift**: Model performance degradation
- **Concept drift**: Target variable relationship changes
- **Covariate shift**: Feature distribution changes

**Monitoring Tools**
- **SageMaker Model Monitor**: Built-in drift detection
- **SageMaker Clarify**: Bias monitoring in production
- **CloudWatch**: Custom metrics, alarms
- **Custom solutions**: Statistical tests, KL divergence

**Quality Monitoring**
- **Data quality**: Missing values, outliers, schema violations
- **Prediction quality**: Accuracy, precision, recall over time
- **Ground truth collection**: Feedback loops, human labeling
- **Statistical process control**: Control charts, threshold monitoring

**A/B Testing**
- **Traffic splitting**: Percentage-based routing
- **Statistical significance**: Hypothesis testing, confidence intervals
- **Winner selection**: Automated promotion based on metrics
- **Multi-armed bandits**: Adaptive traffic allocation

### 4.2 Monitor & Optimize Infrastructure

**Infrastructure KPIs**
- **Utilization**: CPU, memory, GPU usage
- **Throughput**: Requests per second, batch processing rates
- **Availability**: Uptime, error rates
- **Latency**: Response times, queue delays

**Observability Tools**
- **CloudWatch**: Metrics, logs, dashboards, alarms
- **X-Ray**: Distributed tracing, performance analysis
- **CloudTrail**: API call logging, audit trails
- **VPC Flow Logs**: Network traffic analysis

**Cost Optimization**
- **Cost Explorer**: Usage analysis, cost breakdown
- **Trusted Advisor**: Cost optimization recommendations
- **SageMaker Inference Recommender**: Instance type recommendations
- **Compute Optimizer**: Right-sizing recommendations

**Performance Optimization**
- **Auto Scaling**: Dynamic resource adjustment
- **Load balancing**: Traffic distribution
- **Caching**: Model caching, response caching
- **Batch optimization**: Batch size tuning

**Resource Management**
- **Tagging strategy**: Cost allocation, resource organization
- **Budgets and alerts**: Cost control, spending notifications
- **Reserved capacity**: Long-term cost savings
- **Spot instances**: Fault-tolerant workload optimization

### 4.3 Secure AWS Resources

**Identity & Access Management (IAM)**
- **Least privilege**: Minimal required permissions
- **Role-based access**: Service roles, user roles
- **Policy types**: AWS managed, customer managed, inline
- **SageMaker Role Manager**: Pre-configured ML roles

**Network Security**
- **VPC isolation**: Private subnets, security groups
- **NACLs**: Subnet-level access control
- **VPC endpoints**: Private service connectivity
- **WAF**: Web application firewall protection

**Data Protection**
- **Encryption at rest**: KMS keys, S3 default encryption
- **Encryption in transit**: TLS/SSL, VPN connections
- **Key management**: KMS, CloudHSM, key rotation
- **Secrets management**: Secrets Manager, Parameter Store

**Compliance & Governance**
- **Data classification**: Sensitivity levels, handling requirements
- **Audit logging**: CloudTrail, access logging
- **Compliance frameworks**: SOC, PCI DSS, HIPAA, GDPR
- **Data residency**: Regional data requirements

**ML-Specific Security**
- **Model artifact protection**: S3 bucket policies, access logging
- **Training data security**: Access controls, data masking
- **Inference security**: Authentication, rate limiting
- **Model intellectual property**: Access controls, encryption

**Monitoring & Incident Response**
- **Security monitoring**: GuardDuty, Security Hub, Macie
- **Anomaly detection**: Unusual access patterns, data exfiltration
- **Incident response**: Automated responses, notification systems
- **Vulnerability management**: Patch management, security scanning

---

## Key AWS Services Summary

### Core ML Services
- **SageMaker**: End-to-end ML platform
- **Bedrock**: Foundation models and generative AI
- **Comprehend**: Natural language processing
- **Rekognition**: Computer vision
- **Textract**: Document text extraction
- **Translate**: Language translation
- **Transcribe**: Speech-to-text

### Data & Analytics
- **S3**: Object storage and data lakes
- **Glue**: ETL and data catalog
- **Athena**: Serverless query service
- **Kinesis**: Real-time data streaming
- **EMR**: Big data processing
- **QuickSight**: Business intelligence

### Compute & Containers
- **EC2**: Virtual servers
- **Lambda**: Serverless functions
- **ECS/EKS**: Container orchestration
- **Batch**: Batch computing

### Security & Monitoring
- **IAM**: Identity and access management
- **KMS**: Key management
- **CloudWatch**: Monitoring and logging
- **CloudTrail**: API logging
- **X-Ray**: Application tracing

### Developer Tools
- **CodePipeline/CodeBuild/CodeDeploy**: CI/CD
- **CloudFormation**: Infrastructure as code
- **Systems Manager**: Configuration management