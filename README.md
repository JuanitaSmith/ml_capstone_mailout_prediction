## Udacity Machine Learning Engineer Nanodegree

---
## Capstone Project
#### Created by: Juanita Smith
#### Last date: 13 Dec 2021

---



## Customer Segmentation Report for Arvato Financial Services



![img_1.png](images/img2.png)

## Project Overview
The project were sponsored by Udacity in partnership with Arvato Financial Solutions.
The project are within the marketing domain, with main objective to help one of Arvato’s clients, a mail-order company in Germany, acquire more clients easier and smarter.
The mail-out company are running mail-out campaigns, with their objective to increase efficiency in their customer 
acquisition process by targeting the right people. The main goal of the project, is to establish who are the right 
people to target.
The data was provided by Arvato and is protected under Terms and Conditions
**This is a real data-science project with real data.**

   
## Data

Data were supplied by Udacity and protected under license agreement. It can thus not be shared. 

Brief description of the data:


There are four data files associated with this project:
1) Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 
features (columns).
2) Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 
   features (columns).
3) Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 
   982 persons (rows) x 367 (columns).
4) Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 
   persons (rows) x 366 (columns).

## Algorithms and Techniques Used
To solve the problem the following steps will be followed:
1. As the datasets have over 360 features PCA (Principal Component Analysis) will be used to reduce it's
   dimensionally. A scree plot will be used to analyse how each PCA component contribute to the total variance
   explained.
2. Train a model to cluster the general population of Germany using sklearn's KMeans class to perform clustering on
   the PCA-transformed data. Elbow method will be used to select the most optimal number of clusters.
3. Use the trained KMeans model to cluster the customers of the mail order company. By comparing the clusters we can
   describe the relationship between the demographics of the company's existing customers and the general population of Germany. By the end of this part, it should be possible to describe parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.
4. In the last step, supervised learning is used, where customer clustering results is used to predict which
   individuals will respond to mail-out campaigns. Performance of various decision trees algorithms are compared.
   XGBoost, which are the best performing algorithm, will be further refined using Sagemaker's hyper parameter tuner.


## Requirements
- Python 3 (Python 3.7 interpreter was used)
- Libraries needed:
    - pandas==1.3.4
    - yes numpy==1.21.2
    - scikit-learn==1.0.1
    - imbalanced-learn==0.8.0
    - xgboost==1.3.3
    - seaborn==0.11.2
    - matplotlib==3.5.0
    - missingno==0.4.2
    - openpyxl==3.0.9
    - boto3==1.18.21
    - sagemaker-python-sdk
    - jupyter=1.0.0
    - sagemaker==2.68.0
    - python-graphviz
    
- Access to AWS account services sagameker, IAM, Cloudwatch and S3


## Installation

The data are protected under license agreement. It will this not be possible to clone and run the notebooks unless 
you are an enrolled Udacity student. 

To clone the repository: https://github.com/JuanitaSmith/ml_capstone_customer_segmentation.git

Refer to 'requirements.txt' for all related dependencies and versions used

script 'on_create.sh', 'on_start.sh' and environment.yml can be used as a base to recreate the environment in a 
sagemaker notebook instance 

**AWS setup**
- Create IAM role with at least the following services attached: AmazonS3FullAccess, AmazonSageMakerFullAccess
- Create S3 bucket for the project and update default bucket name in `src/config.py` variable `prefix`
- 4 main csv datasets supplied by Udacity needs to be uploaded locally within directory data/raw
- Enhanced feature and level .xlsx-files are already available in data/raw

**Setup local environment** (optional)

-  In AWS create a user with programmatic access and assign sagemaker role created above or alternatively assign
   directly roles `AmazonS3FullAccess` and `AmazonSageMakerFullAccess`
    - Download credentials file

-  Update sagemaker execution role in `scr/config.py` variable `role`

   example how config.py parameters should look like:

    ```
    sagemaker_role = 'arn:aws:iam::<XXX>:role/service-role/AmazonSageMaker-ExecutionRole-20210428T170819'
    prefix = 'arvato_customer_segmentation'
   ```


-  Update aws credentials in file located in '.aws' directory `.aws/config` example

    ```[default]
    region=eu-west-1
    aws_access_key_id = key id from downloaded creditials file
    aws_secret_access_key = secret key from download creditials file
    AWS_DEFAULT_REGION=eu-west-1
    ```



## Notebook Usage
The project has 3 jupyter notebooks located in directory `/notebooks` which needs to be run in sequence. They are 
clearly marked.

- __1_data_exploration__ - Mainly data exploration and decision-making
  - Generate configuration files which are used in cleaning function
  - Trained MinMax scaler and IterativeImputers are generated
- __2_customer_segmentation__ - Unsupervised learning to cluster data
- __3_campaign_prediction__ - Supervised learning to predict mail-out campaign candidates


   
## Resources used:

References

Imputing missing values
- https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- https://stackoverflow.com/questions/61945250/im-getting-negative-values-as-output-of-iterativeimputer-from-sklearn


Imbalanced datasets:
- https://towardsdatascience.com/how-to-deal-with-imbalanced-data-34ab7db9b100
- https://towardsdatascience.com/classification-framework-for-imbalanced-data-9a7961354033
- https://imbalanced-learn.org/stable/references/over_sampling.html#smote-algorithms
- https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
- https://medium.com/analytics-vidhya/how-to-handle-imbalanced-dataset-b3dc05b85bf9
- https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_shrinkage_effect.html


XGBoost:
- https://towardsdatascience.com/xgboost-is-not-black-magic-56ca013144b4
- https://xgboost.readthedocs.io/en/latest/python/python_intro.html#training
- https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/
- https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
- https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2

Feature Selection:
- https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f

PCA
- https://online.stat.psu.edu/stat505/lesson/11/11.4
- https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

AWS and Sagemaker
- https://aws-ml-blog.s3.amazonaws.com/artifacts/prevent-customer-churn/part_2_preventing_customer_churn_XGBoost.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-training-xgboost-report.html
- https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/

Metrics:
- https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781838555078/6/ch06lvl1sec34/confusion
  -matrix

Visualisation
- https://stackoverflow.com/questions/49564844/3d-pca-in-matplotlib-how-to-add-legend
-  https://stackoverflow.com/questions/28227340/kmeans-scatter-plot-plot-different-colors-per-cluster

Environment setup
- https://stackoverflow.com/questions/35802939/install-only-available-packages-using-conda-install-yes-file-requirements-t
- https://aws.amazon.com/premiumsupport/knowledge-center/sagemaker-lifecycle-script-timeout/
- https://subscription.packtpub.com/book/data/9781800208919/2/ch02lvl1sec06/setting-up-amazon-sagemaker-on-your-local-machine
