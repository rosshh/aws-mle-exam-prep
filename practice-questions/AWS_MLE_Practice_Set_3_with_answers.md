# AWS Certified Machine Learning Engineer - Practice Set 3

## 50 Practice MCQ Questions (Exam Style)

**Question 1:** A data scientist is preparing a dataset for training a fraud detection model. The dataset contains 95% legitimate transactions and 5% fraudulent transactions. The model consistently achieves 95% accuracy but fails to detect most fraud cases. What is the most effective approach to address this issue?  

A) Increase the training epochs to improve model performance  
**B) Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the classes**  
C) Use a larger neural network architecture  
D) Reduce the learning rate for better convergence  

**Question 2:** An ML team needs to process real-time streaming data from IoT sensors for anomaly detection. The solution must handle varying throughput from 1,000 to 100,000 records per second with minimal operational overhead. Which AWS architecture would be most suitable?  

**A) Amazon Kinesis Data Streams with Lambda functions for processing**  
B) Amazon SQS with EC2 instances running batch processing  
C) Amazon S3 with scheduled AWS Glue jobs  
D) Amazon RDS with stored procedures for real-time processing  

**Question 3:** A company has deployed a recommendation model using SageMaker real-time endpoints. After 6 months, they notice the model's click-through rate has dropped significantly. Investigation reveals that user preferences have shifted due to seasonal trends. What should be implemented to address this issue?  

A) Increase the endpoint instance size for better performance  
**B) Implement SageMaker Model Monitor to detect data drift and trigger retraining**  
C) Switch to batch inference to reduce costs  
D) Deploy multiple model variants simultaneously  

**Question 4:** During model training with SageMaker, you notice that the training loss decreases steadily but validation loss starts increasing after epoch 10. The training job is configured to run for 50 epochs. What is the best approach to optimize this training process?  

A) Increase the learning rate to speed up convergence  
**B) Configure early stopping with patience=5 and monitor validation loss**  
C) Reduce the batch size to improve gradient estimates  
D) Add more layers to the neural network  

**Question 5:** A financial services company needs to ensure their ML model for loan approval decisions doesn't exhibit bias against protected demographic groups. They want to continuously monitor this in production. Which AWS services and approaches should they use?  

A) SageMaker Debugger to analyze model weights  
**B) SageMaker Clarify for bias detection and ongoing monitoring**  
C) CloudWatch to monitor prediction accuracy  
D) SageMaker Experiments to track model versions  

**Question 6:** An e-commerce company wants to deploy a product recommendation model that can handle traffic spikes during flash sales (10x normal traffic) but also scale down to zero during low-traffic periods to minimize costs. Which deployment option should they choose?  

A) SageMaker real-time endpoints with auto scaling  
**B) SageMaker serverless inference**  
C) SageMaker batch transform jobs  
D) Amazon ECS with spot instances  

**Question 7:** A data engineering team is ingesting customer transaction data from multiple sources with different schemas. They need to standardize the data format and perform feature engineering before storing it in a feature store. The process should be cost-effective and handle schema evolution. What AWS service combination is most appropriate?  

A) AWS Lambda for transformation + S3 for storage  
**B) AWS Glue DataBrew for transformation + SageMaker Feature Store**  
C) Amazon EMR for processing + Amazon RDS for storage  
D) SageMaker Processing jobs + Amazon DynamoDB  

**Question 8:** A machine learning model trained on historical sales data is showing degraded performance in production. Analysis reveals that the input feature distributions have changed significantly compared to the training data. What type of drift is this, and how should it be addressed?  

A) Model drift - retrain with more recent data  
B) Data drift - implement feature scaling in the inference pipeline  
C) Concept drift - collect new ground truth labels  
**D) Covariate shift - adjust the model's decision threshold**  

**Question 9:** A healthcare company is building an ML system to analyze medical images. They have strict requirements for data privacy, audit trails, and the ability to explain model decisions to medical professionals. Which combination of AWS services best addresses these requirements?  

**A) SageMaker + CloudTrail + SageMaker Clarify + KMS encryption**  
B) EC2 instances + S3 + Lambda + CloudWatch  
C) Bedrock + Macie + GuardDuty + IAM  
D) Rekognition + Comprehend Medical + X-Ray + VPC  

**Question 10:** Your team is training a deep learning model for natural language processing. The model has 500 million parameters and the training dataset is 100GB. Training on a single GPU instance takes 2 weeks. What approach would most effectively reduce training time?  

A) Use distributed training with data parallelism across multiple GPU instances  
B) Implement model parallelism to split the model across instances  
C) Use mixed precision training with FP16  
**D) All of the above approaches combined**  

**Question 11:** A retail company wants to implement A/B testing for their recommendation algorithm. They need to split traffic 80/20 between the current model and a new model, with the ability to quickly switch traffic if the new model performs poorly. Which SageMaker feature should they use?  

A) SageMaker Multi-Model Endpoints  
**B) SageMaker endpoints with production variants**  
C) SageMaker Pipelines with conditional steps  
D) SageMaker Batch Transform with multiple jobs  

**Question 12:** An ML engineer is working with a time series forecasting model that needs to predict daily sales for the next 30 days. The historical data shows strong weekly seasonality and holiday effects. The dataset contains 3 years of daily sales data. Which approach would be most effective?  

A) Use a simple linear regression model with trend and seasonality features  
B) Apply ARIMA model with seasonal decomposition  
**C) Use SageMaker's DeepAR algorithm with related time series features**  
D) Implement a neural network with only the sales values as input  

**Question 13:** A company has implemented CI/CD for their ML models using AWS CodePipeline. They want to automatically validate model performance before deployment and rollback if the new model performs worse than the current production model. What should be included in their pipeline?  

A) Unit tests for data quality and integration tests for model accuracy  
B) Automated A/B testing with statistical significance checks  
C) Model validation against holdout test set and performance threshold checks  
**D) All of the above components**  

**Question 14:** Your organization processes customer support tickets using a text classification model. The model was trained 6 months ago and initially achieved 92% accuracy. Recent evaluation shows accuracy has dropped to 78%. Investigation reveals that customers are using new terminology and slang. What type of issue is this and how should it be addressed?  

A) Data drift - implement text normalization preprocessing  
**B) Concept drift - retrain with recent labeled data including new terminology**  
C) Model decay - increase the model complexity  
D) Infrastructure drift - scale up the inference endpoints  

**Question 15:** A startup is building their first ML system and wants to minimize development time while ensuring scalability. They need to implement image classification for a mobile app. The solution should handle varying traffic loads and require minimal ML expertise. What approach should they take?  

A) Train a custom CNN model using SageMaker and deploy on EKS  
**B) Use Amazon Rekognition Custom Labels for the classification task**  
C) Implement a pre-trained model using SageMaker JumpStart  
D) Build a custom solution using EC2 and open-source frameworks  

**Question 16:** An ML team is implementing feature engineering for a customer churn prediction model. They have categorical features with very high cardinality (>10,000 unique values). One-hot encoding would create an extremely sparse feature matrix. What encoding strategy would be most effective?  

A) Use label encoding to convert categories to integers  
**B) Apply target encoding with cross-validation to prevent overfitting**  
C) Use binary encoding to reduce dimensionality  
D) Group low-frequency categories into an "Other" category  

**Question 17:** A company is migrating their on-premises ML training workload to AWS. They use TensorFlow and have custom training scripts. The training requires GPU instances and takes 12 hours to complete. They want to optimize for cost while maintaining the existing workflow. What approach should they take?  

**A) Use SageMaker training jobs with spot instances**  
B) Deploy on EC2 spot instances with custom AMIs  
C) Use AWS Batch with spot instances  
D) Migrate to SageMaker and rewrite the training code  

**Question 18:** Your team has built a model that predicts customer lifetime value. The model performs well on the test set (R² = 0.85) but shows poor performance in production (R² = 0.45). The training data is from 2 years ago, and the production data is current. What is the most likely cause and solution?  

A) Overfitting - add regularization to the model  
B) Data leakage - remove future information from features  
**C) Temporal distribution shift - retrain with more recent data**  
D) Incorrect evaluation metrics - use different performance measures  

**Question 19:** A financial institution wants to deploy a credit scoring model that must provide explanations for each decision to comply with regulatory requirements. The model needs to process 10,000 requests per hour with sub-second latency. Which approach would best meet these requirements?  

A) Use SageMaker Clarify for real-time explanations with each prediction  
**B) Deploy an interpretable model (linear/tree-based) with built-in explanations**  
C) Pre-compute SHAP explanations and store them in a database  
D) Use a complex model with post-hoc explanation generation  

**Question 20:** An e-commerce company wants to implement real-time personalization for their website. They need to update user preferences based on click-stream data and serve recommendations with <100ms latency. Which architecture would be most suitable?  

A) Kinesis Data Streams → Lambda → DynamoDB → API Gateway  
**B) Kinesis Data Streams → SageMaker Feature Store → SageMaker endpoints**  
C) S3 → Glue → Redshift → SageMaker batch inference  
D) Direct API calls to SageMaker endpoints with cached features  

**Question 21:** A data scientist is training a convolutional neural network for medical image analysis. The training dataset has 10,000 images, but the model with 50 million parameters is overfitting severely. What combination of techniques would be most effective to address this issue?  

**A) Data augmentation, dropout, and transfer learning from a pre-trained model**  
B) Increase the learning rate and add more convolutional layers  
C) Use a larger batch size and remove all regularization  
D) Implement early stopping and reduce the dataset size  

**Question 22:** Your company processes streaming IoT sensor data using Kinesis Data Streams. The downstream ML model expects batches of 1000 records for optimal performance, but the stream produces individual records. How should you implement batching efficiently?  

A) Use Kinesis Data Analytics to aggregate records into batches  
**B) Configure Lambda with a batch size of 1000 and appropriate timeout**  
C) Use Kinesis Data Firehose with buffering configuration  
D) Implement a custom application on EC2 to batch records  

**Question 23:** A retail company has trained multiple models for different product categories (electronics, clothing, books). They want to serve all models from a single endpoint to simplify their architecture. The models have different memory requirements and inference patterns. What SageMaker feature should they use?  

**A) SageMaker Multi-Model Endpoints with dynamic loading**  
B) SageMaker Pipeline Mode for efficient data loading  
C) SageMaker Batch Transform with multiple models  
D) Multiple single-model endpoints behind a load balancer  

**Question 24:** An ML team is implementing continuous integration for their model training pipeline. They want to automatically trigger retraining when new data arrives in S3 and ensure that only models meeting quality thresholds are deployed. Which AWS services should they use?  

A) S3 event notifications → Lambda → SageMaker training → CodeDeploy  
**B) S3 → EventBridge → SageMaker Pipelines → Model Registry → CodePipeline**  
C) CloudWatch Events → Step Functions → SageMaker → CloudFormation  
D) S3 → SNS → Lambda → EC2 training instances → ECS deployment  

**Question 25:** A company is building a fraud detection system that needs to process credit card transactions in real-time. The system must handle 50,000 transactions per second with <10ms latency for predictions. Which architecture would best meet these requirements?  

A) API Gateway → Lambda → SageMaker serverless inference  
B) Application Load Balancer → ECS → SageMaker real-time endpoints  
**C) Kinesis → Lambda → DynamoDB lookup with pre-computed scores**  
D) Direct integration with SageMaker multi-model endpoints  

**Question 26:** Your team is working with a time series dataset that has missing values at irregular intervals. The missing data is not random but occurs during system maintenance windows. What imputation strategy would be most appropriate?  

A) Forward fill (carry last observation forward)  
B) Linear interpolation between known values  
C) Mean imputation using historical averages  
**D) Seasonal decomposition with trend-aware interpolation**  

**Question 27:** A healthcare organization is training models on patient data that must comply with HIPAA regulations. They need to ensure that the training data never leaves their private network and that all model artifacts are encrypted. What AWS architecture should they implement?  

**A) VPC with private subnets → SageMaker training in VPC → encrypted S3 buckets**  
B) Public SageMaker training with IAM roles and S3 encryption  
C) On-premises training with results uploaded to encrypted S3  
D) SageMaker training with VPC endpoints and AWS PrivateLink  

**Question 28:** An ML engineer is optimizing a deep learning model for deployment on edge devices with limited memory and compute. The original model is 500MB and provides 95% accuracy. Which technique would be most effective for model compression while maintaining reasonable accuracy?  

A) Knowledge distillation to train a smaller student model  
B) Quantization to reduce model precision from FP32 to INT8  
C) Pruning to remove less important weights and connections  
**D) All of the above techniques should be applied together**  

**Question 29:** A company wants to implement automated hyperparameter tuning for their XGBoost model. They have a limited budget and want to find good hyperparameters within 20 training jobs. Which tuning strategy and configuration should they use?  

A) Grid search with predefined hyperparameter ranges  
B) Random search with wide hyperparameter ranges  
**C) Bayesian optimization with early stopping enabled**  
D) Hyperband algorithm with successive halving  

**Question 30:** Your ML pipeline processes customer reviews for sentiment analysis. The pipeline currently uses a rule-based approach but you want to upgrade to a machine learning model. The solution should require minimal ML expertise and provide good performance out-of-the-box. What AWS service should you use?  

A) Train a custom model using SageMaker with BERT  
**B) Use Amazon Comprehend's built-in sentiment analysis**  
C) Implement a custom neural network using SageMaker JumpStart  
D) Use Amazon Textract to extract text and apply custom rules  

**Question 31:** A financial services company is implementing model risk management for their ML systems. They need to track model lineage, monitor performance degradation, and maintain an audit trail of all model changes. Which AWS services combination addresses these requirements?  

**A) SageMaker Model Registry + SageMaker Model Monitor + CloudTrail**  
B) S3 versioning + CloudWatch + AWS Config  
C) CodeCommit + CodePipeline + CloudFormation  
D) SageMaker Experiments + Lambda + DynamoDB  

**Question 32:** An e-commerce platform wants to implement real-time product recommendations. They have 10 million products and 100 million users. The recommendation system must handle traffic spikes during sales events and provide personalized recommendations based on recent user behavior. What architecture would be most scalable?  

A) Pre-compute all recommendations and store in DynamoDB  
B) Real-time collaborative filtering using SageMaker endpoints  
C) Hybrid approach with pre-computed candidates + real-time ranking  
**D) Use Amazon Personalize for fully managed recommendations**  

**Question 33:** Your team is training a model that uses both structured data (customer demographics) and unstructured data (customer reviews). The structured data is in a relational database, while the reviews are stored as text files in S3. What's the best approach for feature engineering?  

**A) Use SageMaker Processing to join data sources and create unified features**  
B) Process each data type separately and combine predictions at inference time  
C) Convert all data to the same format and use a single algorithm  
D) Use separate models for each data type and ensemble the results  

**Question 34:** A manufacturing company wants to predict equipment failures using sensor data. The data shows that failures are rare events (0.1% of observations) but critical to detect. The business cost of missing a failure is 100x higher than a false alarm. How should you optimize the model?  

A) Optimize for accuracy to get the best overall performance  
**B) Optimize for recall to minimize missed failures, accept higher false positives**  
C) Use precision as the primary metric to reduce false alarms  
D) Use a balanced approach with equal weight to precision and recall  

**Question 35:** Your organization is deploying ML models across multiple AWS regions for a global application. You need to ensure model consistency, manage deployments efficiently, and handle regional compliance requirements. What approach should you take?  

A) Deploy identical models in all regions using CloudFormation templates  
B) Use SageMaker multi-region endpoints with automatic failover  
**C) Implement region-specific models with centralized training and distributed deployment**  
D) Use a single region and serve global traffic with CloudFront  

**Question 36:** A data science team is working with a dataset containing highly correlated features. They've identified that several features have correlation coefficients above 0.9. What's the best approach to handle this multicollinearity?  

A) Remove all correlated features to eliminate multicollinearity  
B) Use principal component analysis (PCA) to create orthogonal features  
C) Apply regularization techniques like Ridge regression  
**D) Analyze feature importance and remove less predictive correlated features**  

**Question 37:** An ML system processes streaming social media data for brand sentiment monitoring. The system needs to handle varying data volumes, language detection, and provide real-time alerts for negative sentiment spikes. Which AWS architecture would be most appropriate?  

**A) Kinesis Data Streams → Lambda → Comprehend → SNS alerts**  
B) Kinesis Data Firehose → S3 → Glue → Athena → QuickSight  
C) API Gateway → Lambda → RDS → CloudWatch alarms  
D) SQS → EC2 processing → ElastiCache → CloudWatch dashboards  

**Question 38:** Your team has deployed a recommendation model that uses collaborative filtering. Users are complaining about the "cold start" problem where new users receive poor recommendations. What strategy would most effectively address this issue?  

A) Increase the model training frequency to include new users faster  
**B) Implement a hybrid approach combining collaborative and content-based filtering**  
C) Use demographic-based recommendations for new users until sufficient data is collected  
D) Apply matrix factorization techniques with user similarity measures  

**Question 39:** A company is implementing MLOps practices for their model development lifecycle. They want to automate the transition from experimentation to production while ensuring reproducibility and governance. Which combination of AWS services would best support this workflow?  

**A) SageMaker Studio + SageMaker Pipelines + Model Registry + CodePipeline**  
B) EC2 + S3 + Lambda + CloudFormation  
C) Glue + Athena + QuickSight + Step Functions  
D) EKS + ECR + CodeBuild + CloudWatch  

**Question 40:** An insurance company is building a model to assess claim fraud risk. The model must process claims in real-time during submission and provide risk scores within 200ms. The model uses 150 features including external data sources. How should they optimize for latency?  

A) Use SageMaker serverless inference with optimized instance types  
**B) Pre-compute high-latency features and cache them in ElastiCache**  
C) Deploy the model on GPU instances for faster inference  
D) Use SageMaker multi-model endpoints to reduce cold start times  

**Question 41:** Your ML pipeline includes a feature selection step that identifies the most predictive features for your model. You want to automate this process and track feature importance across different model versions. Which approach would be most effective?  

**A) Use SageMaker Clarify to generate feature importance and store in Model Registry**  
B) Implement custom feature selection algorithms using SageMaker Processing  
C) Use AWS Glue DataBrew for automated feature engineering and selection  
D) Apply statistical tests in Lambda functions and log results to CloudWatch  

**Question 42:** A video streaming company wants to implement content-based recommendation using video metadata and user viewing history. They have 1 million videos and need to update recommendations as new content is added. What architecture would be most efficient?  

A) Batch process all recommendations daily using EMR  
B) Use real-time feature engineering with Kinesis and SageMaker endpoints  
C) Implement incremental learning with online model updates  
**D) Pre-compute embeddings and use approximate nearest neighbor search**  

**Question 43:** Your team is preparing to deploy a computer vision model for autonomous vehicle testing. The model must work reliably in various lighting conditions, weather, and geographical locations. What validation approach would be most comprehensive?  

A) Use k-fold cross-validation on the entire dataset  
B) Split data by geographical regions and validate across different locations  
**C) Implement stratified sampling based on environmental conditions**  
D) Use temporal validation with the most recent data as the test set  

**Question 44:** A financial institution wants to implement explainable AI for their loan approval model. Regulators require explanations to be provided in human-readable format within 24 hours of any loan decision. Which approach would best meet these requirements?  

A) Generate SHAP explanations for each prediction in real-time  
B) Use an interpretable model architecture with built-in explanations  
**C) Batch generate explanations overnight and store in a searchable database**  
D) Implement LIME explanations with cached results for common scenarios  

**Question 45:** Your organization is scaling ML workloads and wants to optimize costs across training, inference, and storage. You've identified that 40% of your ML costs come from idle resources. What cost optimization strategies should you implement?  

A) Use spot instances for all workloads and implement fault tolerance  
B) Implement auto-scaling for training jobs and serverless inference  
C) Schedule non-critical workloads during off-peak hours with reduced pricing  
**D) Use a combination of spot instances, auto-scaling, and rightsizing based on utilization**  

**Question 46:** A healthcare company is developing a diagnostic model using medical images. They need to ensure the model works well across different hospitals with varying equipment and imaging protocols. What approach would best address domain adaptation?  

A) Train separate models for each hospital and deploy accordingly  
**B) Use domain adversarial training to create domain-invariant features**  
C) Apply transfer learning from models trained on each domain  
D) Implement ensemble methods combining hospital-specific models  

**Question 47:** Your ML system processes customer support tickets and routes them to appropriate teams. The system needs to handle multiple languages and maintain consistent performance as ticket volumes fluctuate throughout the day. Which architecture would be most robust?  

**A) API Gateway → Lambda → Comprehend → SQS → processing teams**  
B) Kinesis → SageMaker endpoints → Step Functions → notification system  
C) Load Balancer → ECS tasks → multilingual NLP models → database routing  
D) CloudFront → Lambda@Edge → language detection → regional processing  

**Question 48:** A retail company wants to implement dynamic pricing using ML. The model must consider competitor prices, inventory levels, demand forecasts, and seasonal trends. The pricing updates need to be applied across 100,000 products multiple times daily. What architecture would handle this scale efficiently?  

A) Scheduled Lambda functions with DynamoDB for price storage  
B) SageMaker batch transform jobs with S3 for data processing  
C) Kinesis Analytics for real-time pricing with RDS for storage  
**D) EMR cluster with Spark for distributed processing and ElastiCache for fast access**  

**Question 49:** Your team is implementing federated learning across multiple data sources that cannot share raw data due to privacy constraints. Each data source has different data distributions and sizes. What approach would be most effective?  

A) Train separate models and ensemble the results  
B) Use differential privacy techniques with centralized training  
C) Implement secure multi-party computation for joint training  
**D) Use transfer learning with domain adaptation techniques**  

**Question 50:** A logistics company wants to optimize delivery routes using reinforcement learning. The system must adapt to real-time traffic conditions, delivery priorities, and driver availability. The solution should learn and improve over time while handling operational constraints. Which AWS approach would be most suitable?  

**A) Use SageMaker RL with custom environments and reward functions**  
B) Implement deep Q-learning using SageMaker training jobs  
C) Use AWS DeepRacer infrastructure adapted for logistics optimization  
D) Build a custom RL solution using EC2 with distributed training  