# AWS Certified Machine Learning Engineer - Practice Set 2

## 50 Additional Practice MCQ Questions

**Question 1:** Which feature of Amazon S3 would you use to automatically transition older training datasets to lower-cost storage classes?  
A) S3 Transfer Acceleration  
**B) S3 Lifecycle Management**  
C) S3 Cross-Region Replication  
D) S3 Versioning  

**Question 2:** In distributed training with SageMaker, which approach splits the model across multiple instances?  
A) Data parallelism  
**B) Model parallelism**  
C) Pipeline parallelism  
D) Gradient parallelism  

**Question 3:** What is the primary purpose of Amazon SageMaker Pipelines?  
A) Real-time model inference  
**B) Orchestrating ML workflows**  
C) Data visualization  
D) Model training only  

**Question 4:** Which metric would indicate that your binary classification model has high precision but low recall?  
A) Many false positives, few false negatives  
**B) Few false positives, many false negatives**  
C) Equal false positives and false negatives  
D) No false predictions  

**Question 5:** What does SMOTE (Synthetic Minority Oversampling Technique) help address?  
A) Overfitting in neural networks  
**B) Class imbalance in datasets**  
C) Feature correlation issues  
D) Missing value imputation  

**Question 6:** Which AWS service provides managed Apache Airflow for ML workflow orchestration?  
A) AWS Step Functions  
**B) Amazon MWAA (Managed Workflows for Apache Airflow)**  
C) AWS Batch  
D) Amazon EventBridge  

**Question 7:** In SageMaker Automatic Model Tuning, which search strategy is most efficient for finding optimal hyperparameters?  
A) Grid search  
B) Random search  
**C) Bayesian optimization**  
D) Exhaustive search  

**Question 8:** What is the main advantage of using Amazon EFS over Amazon S3 for ML training data?  
A) Lower cost per GB  
**B) Concurrent access from multiple instances**  
C) Better data durability  
D) Automatic data encryption  

**Question 9:** Which technique helps prevent catastrophic forgetting when fine-tuning pre-trained models?  
A) Increasing the learning rate  
**B) Elastic Weight Consolidation (EWC)**  
C) Removing all regularization  
D) Training from scratch  

**Question 10:** What is the primary benefit of using SageMaker Data Wrangler's built-in transformations?  
A) Custom algorithm development  
**B) Visual data preparation without coding**  
C) Model deployment automation  
D) Hyperparameter optimization  

**Question 11:** Which deployment strategy allows you to test a new model with a small percentage of traffic before full rollout?  
A) Blue/Green deployment  
B) Rolling deployment  
**C) Canary deployment**  
D) All-at-once deployment  

**Question 12:** In Amazon Kinesis Data Streams, what determines the throughput capacity?  
A) Number of consumers  
**B) Number of shards**  
C) Data retention period  
D) Partition key distribution  

**Question 13:** Which regularization technique is most effective for feature selection in linear models?  
**A) L1 (Lasso) regularization**  
B) L2 (Ridge) regularization  
C) Dropout  
D) Batch normalization  

**Question 14:** What is the purpose of Amazon SageMaker Feature Store's offline store?  
A) Real-time feature serving  
**B) Training and batch inference**  
C) Model deployment  
D) Data visualization  

**Question 15:** Which AWS service would you use to detect personally identifiable information (PII) in your ML datasets?  
A) Amazon Inspector  
**B) Amazon Macie**  
C) AWS Config  
D) Amazon GuardDuty  

**Question 16:** In time series forecasting, what does seasonality refer to?  
A) Random fluctuations in data  
B) Long-term trends in data  
**C) Repeating patterns at regular intervals**  
D) One-time events or anomalies  

**Question 17:** Which SageMaker built-in algorithm is specifically designed for recommendation systems?  
A) XGBoost  
**B) Factorization Machines**  
C) Linear Learner  
D) K-Means  

**Question 18:** What is the main purpose of gradient clipping in neural network training?  
A) Prevent overfitting  
**B) Prevent exploding gradients**  
C) Speed up convergence  
D) Reduce memory usage  

**Question 19:** Which AWS service provides serverless query capabilities for data stored in S3?  
A) Amazon RDS  
B) Amazon Redshift  
**C) Amazon Athena**  
D) Amazon DynamoDB  

**Question 20:** In model evaluation, what does the Area Under the ROC Curve (AUC) measure?  
A) Model training time  
B) Feature importance  
**C) Model's ability to discriminate between classes**  
D) Data quality metrics  

**Question 21:** Which technique would you use to handle missing values in categorical features?  
A) Mean imputation  
**B) Mode imputation or creating a 'missing' category**  
C) Linear interpolation  
D) Forward fill only  

**Question 22:** What is the primary advantage of using Amazon SageMaker Serverless Inference?  
A) Lower latency than real-time endpoints  
**B) Pay-per-request pricing with automatic scaling**  
C) Better model accuracy  
D) Unlimited concurrent requests  

**Question 23:** Which AWS service helps you track and version your ML experiments?  
A) SageMaker Experiments  
B) SageMaker Pipelines  
C) SageMaker Model Registry  
**D) All of the above**  

**Question 24:** In deep learning, what is transfer learning?  
A) Moving models between regions  
**B) Using pre-trained models as starting points**  
C) Transferring data between storage systems  
D) Converting model formats  

**Question 25:** Which metric is most appropriate for evaluating regression models?  
A) F1-score  
B) Accuracy  
**C) Root Mean Squared Error (RMSE)**  
D) ROC-AUC  

**Question 26:** What is the purpose of Amazon SageMaker Model Monitor's baseline job?  
A) Train the initial model  
**B) Establish normal data distribution patterns**  
C) Deploy the model to production  
D) Optimize hyperparameters  

**Question 27:** Which encoding technique preserves ordinal relationships in categorical data?  
A) One-hot encoding  
**B) Label encoding**  
C) Binary encoding  
D) Hash encoding  

**Question 28:** What is the main benefit of using AWS Spot Instances for ML training?  
A) Better performance  
B) More reliable infrastructure  
**C) Cost savings up to 90%**  
D) Automatic model optimization  

**Question 29:** In Amazon SageMaker, what is the difference between training jobs and processing jobs?  
A) No difference, they're the same  
**B) Training jobs create models, processing jobs transform data**  
C) Training jobs are cheaper than processing jobs  
D) Processing jobs only work with built-in algorithms  

**Question 30:** Which AWS service provides distributed tracing for debugging ML applications?  
A) CloudWatch Logs  
B) CloudTrail  
**C) AWS X-Ray**  
D) VPC Flow Logs  

**Question 31:** What is ensemble learning primarily used for?  
A) Data preprocessing  
**B) Combining multiple models to improve performance**  
C) Feature selection  
D) Hyperparameter tuning  

**Question 32:** Which storage class in Amazon S3 offers the lowest cost for long-term archival?  
A) S3 Standard  
B) S3 Intelligent-Tiering  
**C) S3 Glacier Deep Archive**  
D) S3 One Zone-IA  

**Question 33:** In Amazon SageMaker, what is the purpose of the Model Registry?  
A) Store training data  
**B) Version and manage model artifacts**  
C) Monitor model performance  
D) Deploy models automatically  

**Question 34:** Which technique helps reduce overfitting in decision trees?  
A) Increasing tree depth  
**B) Pruning and setting minimum samples per leaf**  
C) Adding more features  
D) Removing regularization  

**Question 35:** What is the primary purpose of Amazon EventBridge in ML workflows?  
A) Model training  
**B) Event-driven automation and integration**  
C) Data storage  
D) Model inference  

**Question 36:** Which AWS service would you use for speech-to-text conversion in your ML pipeline?  
A) Amazon Polly  
**B) Amazon Transcribe**  
C) Amazon Translate  
D) Amazon Comprehend  

**Question 37:** In feature engineering, what is the purpose of polynomial features?  
A) Reduce dimensionality  
B) Handle missing values  
**C) Capture non-linear relationships**  
D) Normalize data distributions  

**Question 38:** Which SageMaker endpoint configuration allows hosting multiple models on a single endpoint?  
A) Real-time endpoints  
**B) Multi-Model Endpoints**  
C) Batch Transform jobs  
D) Serverless endpoints  

**Question 39:** What is the main advantage of using Amazon Kinesis Data Firehose over Kinesis Data Streams?  
A) Lower cost  
**B) Fully managed delivery to destinations**  
C) Real-time processing capabilities  
D) Higher throughput  

**Question 40:** In neural networks, what is the vanishing gradient problem?  
A) Gradients become too large during backpropagation  
**B) Gradients become very small in early layers during backpropagation**  
C) Gradients oscillate during training  
D) Gradients become zero immediately  

**Question 41:** Which AWS service provides natural language processing capabilities without requiring ML expertise?  
A) Amazon SageMaker  
**B) Amazon Comprehend**  
C) AWS Glue  
D) Amazon Textract  

**Question 42:** What is the purpose of cross-validation in model evaluation?  
A) Speed up training  
**B) Assess model performance and reduce overfitting bias**  
C) Increase model accuracy  
D) Reduce training data requirements  

**Question 43:** Which AWS service would you use to automatically scale SageMaker training jobs based on queue length?  
A) Amazon EC2 Auto Scaling  
B) Application Auto Scaling  
C) AWS Auto Scaling  
**D) SageMaker doesn't support automatic scaling of training jobs**  

**Question 44:** In time series analysis, what is stationarity?  
**A) Constant mean and variance over time**  
B) Increasing trend over time  
C) Seasonal patterns in data  
D) Random fluctuations  

**Question 45:** Which technique is most effective for handling high-cardinality categorical features?  
A) One-hot encoding  
**B) Target encoding or embedding**  
C) Label encoding  
D) Binary encoding  

**Question 46:** What is the primary benefit of using Amazon SageMaker Processing jobs?  
A) Model training  
**B) Scalable data preprocessing and feature engineering**  
C) Real-time inference  
D) Model deployment  

**Question 47:** Which AWS service helps you optimize costs by analyzing your ML workload usage patterns?  
A) AWS Cost Explorer  
B) AWS Trusted Advisor  
C) SageMaker Inference Recommender  
**D) All of the above**  

**Question 48:** In deep learning, what is the purpose of batch normalization?  
A) Reduce batch size  
**B) Normalize input batches to stabilize training**  
C) Increase training speed only  
D) Prevent overfitting only  

**Question 49:** Which deployment pattern provides the fastest rollback capability in case of issues?  
A) Rolling deployment  
**B) Blue/Green deployment**  
C) Canary deployment  
D) Linear deployment  

**Question 50:** What is the main purpose of Amazon SageMaker Clarify in production environments?  
A) Model training  
**B) Bias monitoring and explainability**  
C) Data preprocessing  
D) Cost optimization  