
# AWS Certified Machine Learning Engineer - Comprehensive Study Guide

## Table of Contents
- [Domain 1: Data Preparation for ML (28%)](#domain-1-data-preparation-for-ml-28)
- [Domain 2: ML Model Development (26%)](#domain-2-ml-model-development-26)
- [Domain 3: Deployment & Orchestration (22%)](#domain-3-deployment--orchestration-22)
- [Domain 4: ML Solution Monitoring, Maintenance & Security (24%)](#domain-4-ml-solution-monitoring-maintenance--security-24)

---

## Domain 1: Data Preparation for ML (28%)

### Data Ingestion & Storage

**AWS Data Sources**
| Source Type | AWS Service | Key Concepts |
|-------------|------------|--------------|
| Object Storage | S3 | Data lakes, lifecycle policies |
| File Systems | EFS, FSx for NetApp ONTAP | Network file system, concurrent access |
| Streaming | Kinesis Data Streams, Kinesis Firehose, MSK (Managed Kafka) | Real-time ingestion and transformation |
| Databases | RDS, Aurora, DynamoDB, Redshift | Use JDBC or connectors with SageMaker Data Wrangler |

**Data & Analytics**
- **S3**: Object storage and data lakes
- **Glue**: ETL and data catalog
- **Athena**: Serverless query service
- **Kinesis**: Real-time data streaming
- **EMR**: Big data processing
- **QuickSight**: Business intelligence

**Storage Trade-offs**
- **Cost**: S3 Standard > S3 IA > S3 Glacier > S3 Glacier Deep Archive
- **Performance**: EBS PIOPS > EBS GP3 > EFS > S3

**Data Extraction Features**
- **S3 Transfer Acceleration**: CloudFront edge locations for faster uploads
- **EBS PIOPS**: Provisioned IOPS for consistent performance
- **Multi-part uploads**: Large file handling
- **S3 Select**: Query data without downloading entire objects
- **AWS DataSync**: Automated data transfer service for moving large amounts of data between on-premises storage systems and AWS storage services (S3, EFS, FSx). Features include bandwidth throttling, data validation, encryption, and scheduling.

**Best Practices**
- Use Parquet/ORC for analytics (columnar format → faster queries)
- Validate schema with Glue crawlers
- Store data in S3 buckets with partitioning (e.g., s3://data/year=2025/month=11/)
- Automate ETL workflows with Step Functions and Lambda

**Integration Tools**
- **AWS Glue**: Serverless ETL, data catalog
- **Spark on EMR**: Big data processing
- **AWS DataSync**: Automated data transfer between on-premises and AWS storage
- **SageMaker Data Wrangler**: Visual data prep
- **SageMaker Feature Store**: Centralized feature management

### Data Cleaning & Transformation

**Common Transformations**
- **Handle missing values**: fillna, interpolation, deletion
- **Scale numeric data**: standardization or normalization
- **Encode categorical data**: label encoding, one-hot
- **Detect and remove outliers**: Z-score, IQR
- **Transform skewed data**: log or Box-Cox transforms

**Data Cleaning**
- **Outlier handling**: Statistical methods, domain knowledge
- **Missing value imputation**: Mean/median/mode, forward fill, interpolation
- **Deduplication**: Exact matches, fuzzy matching
- **Data combination**: Joins, unions, concatenation

**Tools**
- **AWS Glue DataBrew**: GUI-based cleaning
- **SageMaker Data Wrangler**: End-to-end preprocessing pipelines
- **SageMaker Canvas**: No-code visual interface for data preparation, transformation, and model building without writing code
- **Athena / SQL / PySpark**: Custom transformations

### Feature Engineering

**Feature Engineering Techniques**
- **Scaling/Standardization**: MinMaxScaler, StandardScaler, RobustScaler
- **Normalization**:
  - **L1 Normalization (Manhattan/Taxicab)**: Divides each value by the sum of absolute values; creates sparse outputs with many zeros; robust to outliers; sum of absolute values = 1
  - **L2 Normalization (Euclidean)**: Divides each value by the square root of the sum of squared values; preserves direction while scaling magnitude; sum of squares = 1; sensitive to outliers
- **Binning**: Equal width, equal frequency, custom bins
- **Log transforms**: Reduce skewness, handle exponential relationships

**Encoding Methods**
- **Categorical features**:
  - **One-Hot Encoding**: Nominal categorical variables (no order)
    - **Idea**: One binary column per category
    - **Example**: Color = Red → [1, 0, 0]
    - **Issue**: High dimensionality
  - **Label Encoding**: Ordinal categorical variables
    - **Idea**: Categories mapped to integers
    - **Example**: Size = Small, Medium, Large → 1, 2, 3
    - **Risk**: Incorrect for nominal data (implies numeric distance)
  - **Ordinal Encoding**: Ordered categories (explicit order)
    - **Idea**: User-defined ranking
    - **Example**: Education → HS < Bachelor < Master < PhD
  - **Binary Encoding**: High-cardinality categorical variables
    - **Idea**: Integer → binary representation → multiple columns
    - **Example**: Category 5 → 101
    - **Benefit**: Fewer columns than one-hot
  - **Frequency / Count Encoding**: High-cardinality features
    - **Idea**: Replace categories with occurrence count
    - **Example**: City = Sydney → 5000
    - **Risk**: Loses category meaning
  - **Target Encoding (Mean Encoding)**: Supervised learning
    - **Idea**: Category replaced by mean target value
    - **Example**: City → average house price
    - **Risk**: Target leakage (needs CV/smoothing)
**Text features (NLP)**:
  - **Tokenization**: Text preprocessing (NLP)
    - **Idea**: Split text into tokens
    - **Example**: "I love ML" → [I, love, ML]
      Note: Not numerical yet
  - **Bag of Words (BoW)**: Basic text features
    - **Idea**: Word frequency counts
    - **Example**: "ML is fun" → {ML:1, is:1, fun:1}
    - **Limitation**: Ignores word order
  - **TF-IDF**: Text importance weighting
    - **Idea**: Rare words weighted higher than common words
    - **Example**: "the" → low score, "neural" → high
    - **Better than**: BoW
  - **Word Embeddings**: Semantic text representation
    - **Idea**: Dense vectors capturing meaning
    - **Example**: king − man + woman ≈ queen
    - **Models**: Word2Vec, GloVe
**Scaling & Normalization**:
- **MinMaxScaler**: Scaling numeric features to a fixed range (usually 0–1)
  - **Idea**:** Rescales each feature using (x - min) / (max - min)
  - **Example**: Age = [18, 25, 50] → [0, 0.35, 1]
  - **Benefit**: Preserves original distribution, simple for bounded features
  - **Risk**: Sensitive to outliers (min/max values can skew scaling)
- **StandardScaler**: Standardization of numeric features
  - **Idea**: Rescales features to have mean = 0 and std = 1 using (x - mean)/std
  - **Example**: Age = [18, 25, 50] → [-1.12, -0.47, 1.59]
  - **Benefit**: Works well for algorithms assuming normal distribution (e.g., Linear Regression, Logistic Regression, Neural Networks)
- **Risk**: Outliers affect mean/std; does not bound values


**Feature Selection & Creation**
- **Feature selection**: Reduce dimensionality using correlation, mutual info, or PCA
- **Feature creation**: Combine or decompose features (ratios, polynomial terms, time lags)
- **Embedding generation**: Text (Word2Vec/BERT), Categorical (entity embeddings)

**Tools**
- **SageMaker Feature Store**: Store & reuse features
- **SageMaker Data Wrangler**: 300+ transforms, visual interface

**Data Annotation Services**
- **SageMaker Ground Truth**: Human labeling workflows
- **Mechanical Turk**: Crowdsourced labeling
- **Ground Truth Plus**: Managed labeling teams

### Data Quality, Bias & Governance

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
  - **PII**: Personally Identifiable Information (e.g., name, SSN, email)
  - **PHI**: Protected Health Information (health data covered by HIPAA)
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
  - **EFS**: Use for Linux-based workloads with moderate performance needs, automatic scaling, and built-in multi-AZ redundancy.
  - **FSx**: Use for high-performance workloads (Lustre), Windows compatibility, or specialized file system features.

---

## Domain 2: ML Model Development (26%)

### Algorithm Selection

**Algorithm Selection by Problem Type**

| Problem Type | Algorithms | AWS Tools |
|--------------|-----------|----------|
| Classification | Logistic Regression, Random Forest, SVM, XGBoost, Neural Networks | SageMaker built-ins, Linear Learner, XGBoost |
| Regression | Linear Regression, Random Forest, Gradient Boosting | SageMaker built-ins, Linear Learner, XGBoost |
| Clustering | K-Means, Hierarchical, DBSCAN | SageMaker K-Means |
| Time Series | ARIMA, Prophet, LSTM, DeepAR | SageMaker DeepAR |
| Recommendation | Collaborative filtering, Matrix factorization | SageMaker Factorization Machines |
| NLP | Transformers, BERT, RNNs, BlazingText | SageMaker built-ins, Comprehend (entity recognition, sentiment analysis, key phrase extraction) |
| Computer Vision | CNNs, Vision Transformers | SageMaker Image Classification, Rekognition |

### SageMaker Built-in Algorithms

**Supervised Learning**
- **Linear Learner**: Fast linear regression/classification optimized for large-scale sparse data
- **XGBoost**: Gradient-boosted decision trees for high-accuracy tabular prediction tasks
- **Random Cut Forest (RCF)**: Detects anomalies in unlabeled datasets using unsupervised forest models
- **K-NN**: Finds nearest neighbors for classification or regression using L2 distance
- **Factorization Machines**: Performs sparse prediction tasks such as recommendations with interactions
- **Seq2Seq**: Neural translation and sequence-to-sequence tasks using encoder–decoder architectures
- **BlazingText**: Fast and scalable word embedding + text classification
- **Image Classification**: Trains deep CNNs for multi-class image classification
- **Object Detection**: Detects and localizes objects in images using SSD-based architecture
- **Semantic Segmentation**: Produces pixel-wise image classifications for segmentation tasks
- **Neural Topic Model (NTM)**: Discovers topics in large document corpora using variational inference
- **DeepAR Forecasting**: Probabilistic time-series forecasting using autoregressive RNNs
- **LDA (Latent Dirichlet Allocation)**: Unsupervised topic modeling for text documents

**Unsupervised Learning**
- **K-Means**: Clusters unlabeled data by minimizing within-cluster distance
- **PCA**: Performs dimensionality reduction via principal component transformation
- **IP Insights**: Detects suspicious entities (e.g., fraudulent IP addresses) using embedding models

**Computer Vision (Specialized)**
- **Object2Vec**: Learns embeddings for paired entities (e.g., queries/products)
- **Reinforcement Learning (RLEstimator)**: Provides frameworks to train RL agents using SageMaker containers

**Time Series / Forecasting**
- **DeepAR**: Probabilistic multivariate forecasting using autoregressive RNNs
- **Forecasting with XGBoost containers**: Tabular models adapted for time-series using tree ensembles

**NLP (Specialized)**
- **BlazingText**: Word2Vec embeddings and text classification at large scale
- **Seq2Seq**: Neural translation and sequence transformation
- **NTM**: Topic modeling for unlabeled text

**Anomaly Detection**
- **Random Cut Forest**: Detects unusual patterns in numerical or streaming data
- **IP Insights**: Identifies anomalous entity behaviors like fraud or misuse

**Key Decision Factors for Choosing ML Approaches**
1. **Data characteristics**: Volume, velocity, variety, veracity
2. **Problem complexity**: Simple classification vs. complex multi-stage decision making
3. **Performance requirements**: Accuracy, latency, throughput, explainability
4. **Regulatory constraints**: Privacy, fairness, transparency
5. **Resource limitations**: Compute, memory, cost constraints
6. **Domain knowledge availability**: Ability to incorporate expert knowledge
7. **Deployment environment**: Edge, cloud, hybrid, multi-region

**Tips**
- Structured tabular data → XGBoost
- Text → BlazingText or BERT
- Images → CNN or Rekognition
- Sequential/time → RNN/LSTM/DeepAR

**Interpretability Considerations**
- **High interpretability**: Linear models, Decision Trees
- **Medium interpretability**: Random Forest, Gradient Boosting
- **Low interpretability**: Neural Networks, Deep Learning
- **SHAP values**: Model-agnostic explanations
- **LIME**: Local interpretable explanations

**AI Services vs Custom Models**
- **Use AI Services when**: Standard use cases, quick deployment, no ML expertise
- **Amazon Textract**: Document text extraction
- **Amazon Translate**: Text translation
- **Amazon Transcribe**: Speech-to-text
- **Amazon Rekognition**: Image/video analysis
- **Amazon Bedrock**: Foundation models, generative AI
- **Amazon Comprehend**: Natural language processing service that uses machine learning to find insights and relationships in text
  - **Key features**: Entity recognition, key phrase extraction, sentiment analysis, language detection, topic modeling, PII identification
  - **Custom classification**: Train custom models for domain-specific text classification
  - **Custom entity recognition**: Train models to recognize custom entities specific to your domain
  - **Real-time or batch processing**: Process documents individually or in batches
  - **Medical NLP**: Specialized version (Comprehend Medical) for healthcare text analysis
  - **Document analysis**: Extract text, analyze structure, and identify relationships in documents
- **SageMaker Canvas**: No-code ML model building and deployment with visual interface; supports tabular data, time series forecasting, image classification, and text prediction; includes automated data preparation and model explanation features
- **Use Custom Models when**: Specialized requirements, proprietary data, performance optimization

**Cost Considerations**
- **Training costs**: Instance types, training time
- **Inference costs**: Real-time vs batch, throughput requirements
- **Storage costs**: Model artifacts, training data
- **Spot instances**: Up to 90% savings for fault-tolerant workloads

### Key Formulas & Metrics

**Classification Metrics**
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
  - What it means: The proportion of all predictions that were correct
  - When to use: Balanced datasets where all classes are equally important
  - Limitations: Can be misleading with imbalanced datasets (e.g., 99% negative cases)

- **Precision** = TP / (TP + FP)
  - What it means: Of all instances predicted as positive, how many were actually positive
  - When to use: When false positives are costly (e.g., spam detection, medical diagnosis)
  - Also called: Positive Predictive Value

- **Recall** = TP / (TP + FN)
  - What it means: Of all actual positive instances, how many were correctly identified
  - When to use: When false negatives are costly (e.g., disease detection, fraud detection)
  - Also called: Sensitivity, True Positive Rate

- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
  - What it means: Harmonic mean of precision and recall
  - When to use: When you need to balance precision and recall, especially with imbalanced datasets
  - Range: 0 (worst) to 1 (best)

- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
  - What it means: Measures the model's ability to distinguish between classes across all threshold values
  - When to use: When you need a threshold-independent performance measure
  - Range: 0.5 (random) to 1 (perfect)

 **Regression Metrics**
- **MSE** = Σ(y - ŷ)² / n
  - What it means: Average of squared differences between predicted and actual values
  - When to use: When larger errors should be penalized more heavily
  - Limitations: Not in the same units as the target variable

- **RMSE** = √(MSE)
  - What it means: Square root of MSE, giving a measure in the same units as the target variable
  - When to use: When you need an error metric in the original units of the target variable
  - Interpretation: Average magnitude of error

- **MAE** = Σ|y - ŷ| / n
  - What it means: Average of absolute differences between predicted and actual values
  - When to use: When all errors should be treated equally regardless of direction
  - Advantage: Less sensitive to outliers than MSE/RMSE

- **R²** = 1 - (SSres / SStot)
  - What it means: Proportion of variance in the dependent variable explained by the model
  - When to use: To assess how well the model explains the variation in the data
  - Range: ≤ 1 (1 is perfect, 0 means no better than predicting the mean, negative values indicate worse than predicting the mean)

### Confusion Matrix Terms
- **TP (True Positive)**: Correctly predicted positive
- **FP (False Positive)**: Incorrectly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FN (False Negative)**: Incorrectly predicted negative

### Distribution Comparison Metrics
- **Total Variation Distance (TVD)**
  - What it is: A measure of how different two probability distributions are
  - In plain terms: "How much would I need to change one distribution to make it look like the other?"
  - Range: 0 = exactly the same, 1 = completely different
  - Example: If two datasets label classes very differently, TVD tells you how far apart those label distributions are

- **Kullback–Leibler Divergence (KL)**
  - What it is: A measure of how much information is lost when one probability distribution is used to approximate another
  - In plain terms: "How surprised would I be if I assumed one distribution, but the data actually came from another?"
  - Key point:
    - Not symmetric (KL(A‖B) ≠ KL(B‖A))
    - 0 means identical distributions
    - Larger = more mismatch
  - Example: Used often to compare predicted probabilities vs true distributions

- **Difference in Proportions of Labels (DPL)**
  - What it is: The difference in class proportions between two datasets or groups
  - In plain terms: "How much more (or less) common is a label in one group compared to another?"
  - Example: If Group A has 70% positive labels and Group B has 50%, DPL = 20%

- **Class Imbalance (CI)**
  - What it is: A measure of how unevenly classes are represented in a single dataset
  - In plain terms: "Are some labels much more common than others?"
  - Example:
    - 95% "No", 5% "Yes" → highly imbalanced
    - 50/50 → balanced
  - Why it matters: Class imbalance can make models look accurate while performing poorly on rare classes

- **Partial Dependence Plots (PDPs)**
  - What it is: A model interpretability technique that shows the average effect of one or more features on a model’s predictions, marginalizing over all other features
  - In plain terms: "If I change this feature, how does the model’s prediction change on average?"
  - Example:
    - Visualizing how predicted house prices change as square footage increases
    - Understanding how loan default risk varies with interest rate
      
- **Shapley Values**
  - What it is: Shapley values to determine the contribution that each feature made to model predictions.
  - In plain terms: “How much did each feature help or hurt this specific prediction compared to an average baseline?”
  - Example:
    - Baseline prediction: 50% chance of default
    - Income increases prediction to 35% (−15%)
    
### Model Training & Optimization

**Training Process Elements**
- **Epochs**: Full passes through training data
- **Batch size**: Samples processed before gradient update
- **Learning rate**: Step size for gradient descent
- **Early stopping**: Prevent overfitting, save time

**AWS Training Tools**
- **SageMaker Training Jobs**: Single or distributed training
- **SageMaker Experiments**: Track metrics & versions
- **Spot Training**: Cost optimization
- **Managed Frameworks**: TensorFlow, PyTorch, Scikit-learn, MXNet

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

**Neural Network Tuning**
- Small batch sizes tend to provide better generalization and avoid local minima
- Large batch sizes can converge faster but may settle on suboptimal solutions
- Large learning rates can overshoot the correct solution or cause divergence
- Small learning rates increase training time but provide more stable convergence

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

### Model Evaluation & Analysis

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

**Bias & Interpretability**
- Use SageMaker Clarify for bias detection (pre/post-training) and explainability (SHAP values)
- Use SHAP / LIME locally for model transparency

---

## Domain 3: Deployment & Orchestration (22%)

### Deployment Infrastructure

**Deployment Patterns**
- **Real-time inference**: Low latency, synchronous
- **Batch inference**: High throughput, asynchronous
- **Edge deployment**: Local processing, reduced latency
- **Hybrid**: Combination based on use case

**SageMaker Endpoint Types**

| Type | Description | Use Case |
|------|-------------|----------|
| Real-time endpoints | Always-on, <100ms latency | Interactive applications |
| Serverless endpoints | Pay-per-use, automatic scaling | Variable traffic patterns |
| Asynchronous endpoints | Long-running inference, queued requests | Large inputs/outputs |
| Batch Transform | Batch processing, cost-effective | Offline predictions |
| Multi-model endpoints | Host multiple models on single endpoint | Cost optimization |

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

**Deployment Targets**
- **SageMaker endpoints**: Managed inference
- **Amazon EKS**: Kubernetes-based, container orchestration
- **Amazon ECS**: Docker containers, AWS-managed
- **AWS Lambda**: Serverless, event-driven
- **Edge devices**: IoT, mobile applications

### Infrastructure as Code & Orchestration

**Infrastructure as Code (IaC)**
- **AWS CloudFormation**: Native AWS IaC, JSON/YAML templates
- **AWS CDK**: Programming language-based IaC
- **Terraform**: Multi-cloud IaC tool
- **SageMaker Projects**: Pre-configured MLOps templates

**Orchestration Tools**
- **SageMaker Pipelines**: Native ML workflows
- **Apache Airflow (MWAA)**: Complex DAGs, external integrations
- **AWS Step Functions**: Serverless workflows, error handling
- **Lambda**: Event-driven, lightweight processing

**Pipeline Steps**
- Data processing → Training → Evaluation → Model registration → Deployment

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

### CI/CD for ML

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

### Model Monitoring

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

### Infrastructure Monitoring & Optimization

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

### Security & Compliance

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
