# AWS Certified Machine Learning Engineer - Associate (MLA-C01) Practice Questions

## 50 Practice MCQ Questions

**Question 1:** What is the primary purpose of A/B testing in ML model deployment?  
A) To reduce costs  
B) To compare performance between different model versions  
C) To speed up training  
D) To improve data quality  

**Question 2:** Which SageMaker endpoint type is most cost-effective for infrequent, unpredictable inference requests?  
A) Real-time endpoints  
B) Batch transform  
C) Serverless endpoints  
D) Asynchronous endpoints  

**Question 3:** What is the purpose of log transformation in feature engineering?  
A) To encrypt sensitive data  
B) To reduce skewness in data distribution  
C) To increase the number of features  
D) To convert categorical to numerical data  

**Question 4:** Which hyperparameter tuning strategy typically provides the best results for complex search spaces?  
A) Grid search  
B) Random search  
C) Bayesian optimization  
D) Manual tuning  

**Question 5:** What does data drift refer to in ML model monitoring?  
A) Changes in model architecture  
B) Changes in the statistical properties of input data over time  
C) Changes in training algorithms  
D) Changes in deployment infrastructure  

**Question 6:** Which encoding technique is most appropriate for categorical features with no ordinal relationship?  
A) Label encoding  
B) Binary encoding  
C) One-hot encoding  
D) Ordinal encoding  

**Question 7:** What is the main advantage of using spot instances for ML training jobs?  
A) Better performance  
B) Significant cost savings (up to 90% off)  
C) More reliable infrastructure  
D) Automatic model optimization  

**Question 8:** Which technique is most effective for detecting selection bias in your training data?  
A) Cross-validation  
B) SageMaker Clarify bias analysis  
C) Hyperparameter tuning  
D) Feature scaling  

**Question 9:** What is the primary advantage of using SageMaker Multi-Model Endpoints?  
A) Better model accuracy  
B) Cost reduction by hosting multiple models on a single endpoint  
C) Faster training time  
D) Automatic model versioning  

**Question 10:** Which AWS service helps you discover and protect sensitive data in your ML datasets?  
A) AWS Config  
B) Amazon Macie  
C) AWS Shield  
D) AWS WAF  

**Question 11:** What does the ROC-AUC metric measure?  
A) The model's ability to distinguish between classes across all thresholds  
B) The average prediction error  
C) The correlation between features  
D) The training time efficiency  

**Question 12:** Which AWS service would you use to automatically trigger model retraining based on data drift?  
A) CloudWatch Events  
B) Amazon EventBridge  
C) AWS Lambda  
D) All of the above  

**Question 13:** What is the primary benefit of using Amazon FSx for NetApp ONTAP for ML workloads?  
A) Lowest cost storage option  
B) High-performance file system with data management features  
C) Automatic data labeling  
D) Built-in ML algorithms  

**Question 14:** Which technique would you use to reduce catastrophic forgetting when fine-tuning a pre-trained model?  
A) Increase learning rate  
B) Use elastic weight consolidation or gradual unfreezing  
C) Remove regularization  
D) Train for more epochs  

**Question 15:** What is the recommended way to handle secrets and credentials in ML applications?  
A) Hard-code them in the application  
B) Store them in environment variables  
C) Use AWS Secrets Manager or Systems Manager Parameter Store  
D) Store them in S3 buckets  

**Question 16:** You need to ingest real-time streaming data from multiple sources into your ML pipeline. Which AWS service combination would be most appropriate?  
A) S3 + Lambda  
B) Kinesis Data Streams + Kinesis Analytics  
C) RDS + DynamoDB  
D) EFS + EC2  

**Question 17:** What is the primary purpose of blue/green deployment?  
A) Cost optimization  
B) Zero-downtime deployment with quick rollback capability  
C) Faster model training  
D) Better model accuracy  

**Question 18:** Which SageMaker tool would you use to analyze model bias and explainability?  
A) SageMaker Debugger  
B) SageMaker Clarify  
C) SageMaker Experiments  
D) SageMaker Pipelines  

**Question 19:** What is the purpose of VPC endpoints in ML security?  
A) To increase network speed  
B) To keep traffic within AWS network and improve security  
C) To reduce costs  
D) To enable internet access  

**Question 20:** Which deployment strategy minimizes risk by gradually shifting traffic to a new model version?  
A) Blue/Green deployment  
B) Canary deployment  
C) Rolling deployment  
D) All-at-once deployment  

**Question 21:** What does class imbalance refer to in the context of pre-training bias?  
A) Unequal number of samples across different classes  
B) Different feature scales  
C) Missing values in the dataset  
D) Inconsistent data types  

**Question 22:** Which AWS service automatically scales your ML inference infrastructure based on demand?  
A) Amazon EC2 Auto Scaling  
B) SageMaker Automatic Scaling  
C) AWS Lambda  
D) All of the above  

**Question 23:** Which algorithm would be most appropriate for a binary classification problem with high interpretability requirements?  
A) Neural Networks  
B) Random Forest  
C) Logistic Regression  
D) Deep Learning  

**Question 24:** What is the recommended approach for implementing least privilege access for ML resources?  
A) Give all users admin access  
B) Create specific IAM roles with minimal required permissions  
C) Use root credentials for all operations  
D) Disable access controls for better performance  

**Question 25:** Which data format is most suitable for analytical workloads with columnar access patterns and provides the best compression for large datasets?  
A) CSV  
B) JSON  
C) Parquet  
D) Avro  

**Question 26:** What is the benefit of using container images for model deployment?  
A) Faster training  
B) Environment consistency and portability  
C) Lower storage costs  
D) Automatic scaling  

**Question 27:** Which storage option provides the lowest latency for frequently accessed ML training data?  
A) Amazon S3 Standard  
B) Amazon S3 Glacier  
C) Amazon EFS  
D) Amazon EBS with Provisioned IOPS  

**Question 28:** What does L1 regularization primarily help with?  
A) Increasing model complexity  
B) Feature selection by driving some weights to zero  
C) Faster convergence  
D) Better generalization only  

**Question 29:** Which SageMaker feature automatically detects model and data drift?  
A) SageMaker Debugger  
B) SageMaker Model Monitor  
C) SageMaker Clarify  
D) SageMaker Experiments  

**Question 30:** What is the primary purpose of early stopping in model training?  
A) To reduce training costs  
B) To prevent overfitting and save training time  
C) To increase model accuracy  
D) To handle missing data  

**Question 31:** Which tool helps you optimize costs by recommending the right instance types for your ML workloads?  
A) AWS Trusted Advisor  
B) SageMaker Inference Recommender  
C) AWS Cost Explorer  
D) All of the above  

**Question 32:** What is the primary advantage of using Amazon S3 Transfer Acceleration for data ingestion?  
A) Reduces storage costs  
B) Encrypts data in transit  
C) Speeds up uploads using CloudFront edge locations  
D) Automatically converts file formats  

**Question 33:** Which orchestration tool is best for complex ML workflows with conditional logic?  
A) AWS Lambda  
B) SageMaker Pipelines  
C) Amazon EventBridge  
D) AWS Step Functions  

**Question 34:** In SageMaker Feature Store, what is the difference between online and offline feature stores?  
A) Online is for batch inference, offline is for real-time  
B) Online is for real-time inference, offline is for training and batch inference  
C) No difference, they serve the same purpose  
D) Online stores more data than offline  

**Question 35:** What metric should you monitor to detect performance degradation in deployed models?  
A) Training accuracy only  
B) Inference latency and accuracy/error rates  
C) Storage utilization only  
D) Network bandwidth only  

**Question 36:** Which AWS service provides distributed tracing for ML applications?  
A) CloudWatch  
B) CloudTrail  
C) AWS X-Ray  
D) VPC Flow Logs  

**Question 37:** What is the primary benefit of using SageMaker JumpStart?  
A) Cheaper training costs  
B) Pre-built models and solution templates  
C) Unlimited compute resources  
D) Automatic hyperparameter tuning  

**Question 38:** In SageMaker Data Wrangler, which technique would you use to handle missing values in a numerical feature?  
A) One-hot encoding  
B) Mean imputation  
C) Label encoding  
D) Tokenization  

**Question 39:** For real-time inference with strict latency requirements, which compute type should you choose?  
A) CPU instances  
B) GPU instances  
C) FPGA instances  
D) Depends on the model complexity and requirements  

**Question 40:** Which technique helps prevent overfitting in neural networks?  
A) Increasing the learning rate  
B) Adding more layers  
C) Dropout regularization  
D) Reducing the training dataset size  

**Question 41:** What is the recommended approach for handling PII data in ML datasets?  
A) Store it in plain text  
B) Use data masking or anonymization  
C) Ignore compliance requirements  
D) Only encrypt during transit  

**Question 42:** Which scaling metric is most appropriate for auto-scaling SageMaker endpoints?  
A) CPU utilization only  
B) Memory utilization only  
C) Invocations per instance  
D) Network traffic  

**Question 43:** When should you consider using Amazon Bedrock instead of training a custom model?  
A) When you need maximum control over the algorithm  
B) When you have unlimited training data  
C) For general-purpose tasks like text generation or analysis  
D) When you need the lowest possible inference latency  

**Question 44:** What is ensemble learning primarily used for?  
A) Reducing training time  
B) Combining multiple models to improve performance  
C) Reducing memory usage  
D) Simplifying model architecture  

**Question 45:** Which AWS service is best suited for performing ETL operations on large datasets without managing infrastructure?  
A) AWS Lambda  
B) Amazon EMR  
C) AWS Glue  
D) Amazon EC2  

**Question 46:** In AWS CodePipeline, which service is responsible for building and testing code?  
A) CodeCommit  
B) CodeBuild  
C) CodeDeploy  
D) CodeStar  

**Question 47:** What is the purpose of SageMaker Neo?  
A) Model training optimization  
B) Model compilation for edge devices  
C) Data preprocessing  
D) Hyperparameter tuning  

**Question 48:** Which metric is most appropriate for evaluating a multi-class classification problem with imbalanced classes?  
A) Accuracy  
B) F1-score (macro-averaged)  
C) Mean Squared Error  
D) R-squared  

**Question 49:** Which AWS service would you use for human-in-the-loop data labeling tasks?  
A) Amazon Comprehend  
B) Amazon Textract  
C) Amazon SageMaker Ground Truth  
D) Amazon Rekognition  

**Question 50:** In SageMaker Automatic Model Tuning, what is the maximum number of concurrent training jobs you can run?  
A) 5  
B) 10  
C) 20  
D) 100  

---

## Answer Key

1. B) To compare performance between different model versions
2. C) Serverless endpoints - Pay-per-use pricing for infrequent requests
3. B) To reduce skewness in data distribution
4. C) Bayesian optimization - Most efficient for complex search spaces
5. B) Changes in the statistical properties of input data over time
6. C) One-hot encoding - No ordinal relationship assumed
7. B) Significant cost savings (up to 90% off)
8. B) SageMaker Clarify bias analysis - Purpose-built for bias detection
9. B) Cost reduction by hosting multiple models on a single endpoint
10. B) Amazon Macie - Data discovery and protection service
11. A) The model's ability to distinguish between classes across all thresholds
12. D) All of the above - All can trigger automated workflows
13. B) High-performance file system with data management features
14. B) Use elastic weight consolidation or gradual unfreezing
15. C) Use AWS Secrets Manager or Systems Manager Parameter Store - Secure credential management
16. B) Kinesis Data Streams + Kinesis Analytics - Purpose-built for real-time streaming data
17. B) Zero-downtime deployment with quick rollback capability
18. B) SageMaker Clarify - Purpose-built for bias analysis and explainability
19. B) To keep traffic within AWS network and improve security
20. B) Canary deployment - Gradual traffic shifting
21. A) Unequal number of samples across different classes
22. D) All of the above - Multiple services provide auto-scaling capabilities
23. C) Logistic Regression - High interpretability for binary classification
24. B) Create specific IAM roles with minimal required permissions
25. C) Parquet - Optimized for analytical workloads with excellent compression
26. B) Environment consistency and portability
27. D) Amazon EBS with Provisioned IOPS - Lowest latency storage option
28. B) Feature selection by driving some weights to zero
29. B) SageMaker Model Monitor - Purpose-built for drift detection
30. B) To prevent overfitting and save training time
31. D) All of the above - All provide cost optimization recommendations
32. C) Speeds up uploads using CloudFront edge locations
33. D) AWS Step Functions - Best for complex workflows with conditional logic
34. B) Online is for real-time inference, offline is for training and batch inference
35. B) Inference latency and accuracy/error rates - Key performance indicators
36. C) AWS X-Ray - Distributed tracing service
37. B) Pre-built models and solution templates
38. B) Mean imputation - Standard technique for handling missing numerical values
39. D) Depends on the model complexity and requirements
40. C) Dropout regularization - Prevents overfitting in neural networks
41. B) Use data masking or anonymization - Required for compliance
42. C) Invocations per instance - Most relevant for ML inference
43. C) For general-purpose tasks like text generation or analysis
44. B) Combining multiple models to improve performance
45. C) AWS Glue - Serverless ETL service
46. B) CodeBuild - Responsible for building and testing
47. B) Model compilation for edge devices
48. B) F1-score (macro-averaged) - Handles class imbalance well
49. C) Amazon SageMaker Ground Truth - Human-in-the-loop labeling service
50. B) 10 - Default limit for concurrent training jobs
