import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sagemaker
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import time

from src.config import *


def build_feature_summary(data, information_level, feature_description, missing_values, generate_excel=False):
    """
    Build new dataset describing each column more precisely to support decision making and analytics
    
    All columns from the main dataset is merged datasets `information levels` and `feature descriptions`
    Some columns exists in some datasets and not in others and this will attempt to bring it all together.
    Access which information is generally missing
    Also calculate for each column present in the data, the number of unique entries, and list those unique entries
    This will help decide if any more transformations are needed
    
    Args:
       data (string): main demographics dataset
       information_level (string): mapping of each column to a information level with a brief description
       feature_description (string): additional description for each column
       missing_values: the % of missing values in each column
       generate_excel: Option to save excel file locally
       
    Returns:
       dataframe listing all columns with enriched with descriptions and unique values
    """

    # Count the unique number of values for each column in the main dataset
    feature_summary = pd.DataFrame(data.nunique(axis=0, dropna=True),
                                   columns=['distinct_values']).reset_index().sort_values(by='distinct_values',
                                                                                          axis=0,
                                                                                          ascending=False)
    feature_summary.rename(columns={'index': 'Attribute'}, inplace=True)

    # Add a column containing a list of all the unique values for each column
    feature_summary['unique_values'] = [list(data[col].unique()) for col in feature_summary['Attribute']]

    # To which level does each column belongs to ?
    feature_summary = feature_summary.join(information_level, on='Attribute', how='left')

    #  Add an additional description for each feature coming from features dataset
    feature_summary["feature_description"] = feature_summary["Attribute"].apply(lambda x: feature_description.get(x))

    # add a column to show the % of missing values in each column
    df_missing_columns_proportions = pd.DataFrame(missing_values, columns=['%missing_values'])
    df_missing_columns_proportions.index.rename('Attributes', inplace=True)
    feature_summary = feature_summary.join(df_missing_columns_proportions, on='Attribute', how='left')

    if generate_excel:
        path = '{}/{}'.format(config.path_output, config.filename_enhanced_features)
        feature_summary.to_excel(path)

    return feature_summary


def find_components(df, from_component=1, to_component=140, interval=10):
    """Find number of components providing around 90% of variance explained"""

    accs = []
    comps = []

    for comp in range(from_component, to_component, interval):
        comps.append(comp)
        pca = PCA(comp)
        pca.fit_transform(df)
        acc = sum(pca.explained_variance_ratio_)
        accs.append(acc)
        print('Number of components {}, Accuracy {}'.format(comp, acc))

    print('Final Number of components {}, Accuracy {}'.format(comp, acc))
    plt.subplots(figsize=(16, 10))
    plt.plot(comps, accs, 'bo')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.title('Components Accuracy')
    plt.show()

    return accs, comps


def summarize_pca_results(df, pca):
    """Enhance pca data with index, column headings and explained variance"""

    # Dimension indexing
    dimensions = ['PCA {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=df.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def get_kmeans_score(df_pca, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=3)
    model = kmeans.fit(df_pca)
    predict = model.predict(df_pca)
    score = np.abs(model.score(df_pca))
    return score


def make_train_csv(x, y, prefix, local_path, filename, sagemaker_session, bucket):
    """
    Merges features and labels and converts them into one csv file with labels in the first column.
    
    File is saved locally and then uploaded to s3. AWS requires no column headings or indexes to be present

    Args:
        x: data features
        y: data labels
        prefix: default S3 sub folder for this project
        local_path: directory where training and validation files will be saved in s3
        filename: name of csv file, ex. 'train.csv'
        sagemaker_session: sagemaker session
        bucket: default bucket assigned to sagemaker session

    Returns: S3 file path where data is stored

    """

    # make data dir, if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())    
    full_local_filename = os.path.join(local_path, filename)

    print('Local path: {} with shape {}'.format(full_local_filename, x.shape))

    # save file locally
    df = pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1)
    df.to_csv(full_local_filename, header=False, index=False)

    # will save also the column features an index still ?
    # df.to_parquet(full_local_filename)

    # copy local file to S3    
    s3_path = os.path.join(prefix, local_path)
    s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)

    print('File created: {}'.format(s3_full_path))

    return s3_full_path, df


def make_test_csv(x, prefix, local_path, filename, sagemaker_session, bucket):
    """
    Saves features to local csv file and upload to S3. 
   
    AWS required that abel column are not present in this file, no column headings or indexes should be present

    Args:
        x: data features
        prefix: default S3 sub folder for this project
        local_path: directory where training and validation files will be saved in s3
        filename: name of csv file, ex. 'train.csv'
        sagemaker_session: sagemaker session
        bucket: default bucket assigned to sagemaker session

    Returns: S3 file path where data is stored

    """

    # make data dir, if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    full_local_filename = os.path.join(local_path, filename)
    print('Local path: {} with shape {}'.format(full_local_filename, x.shape))    

    # save file locally
    pd.DataFrame(x).to_csv(full_local_filename, header=False, index=False)
    #     pd.DataFrame(x).to_parquet(full_local_filename)

    # copy local file to S3  
    s3_path = os.path.join(prefix, local_path)
    s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)

    print('File created: {}'.format(s3_full_path))

    return s3_full_path


def create_feature_map(features):
    # https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2
    outfile = open(filename_model_featuremap, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def train_predict(learner, X_train, y_train, X_test, y_test): 
    """
    train variance learners to compare performance on accuracy and auc
    
    args:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    """
    
    print('Start of training: {}'.format(learner.__class__.__name__))  
        
    results = {}
    
#   Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time.time() # Get start time
    learner = learner.fit(X_train, y_train)
    end = time.time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions 
    start = time.time() # Get start time
    predictions_test_proba = learner.predict_proba(X_test)[:, -1]
    predictions_train_proba = learner.predict_proba(X_train)[:,-1]   
    end = time.time() # Get end time
    
    predictions_test = [round(num) for num in predictions_test_proba.squeeze()]
    predictions_train = [round(num) for num in predictions_train_proba.squeeze()]    
   
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the training samples 
    results['acc_train'] = accuracy_score(predictions_train, y_train)

    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(predictions_test, y_test)         
    
    # Compute AUC on training set
    results['auc_train'] = roc_auc_score(y_train, predictions_train_proba, )
          
    # Compute AUC on testing set
    results['auc_test'] = roc_auc_score(y_test, predictions_test_proba) 
          
    # Success
    print("{} training completed".format(learner.__class__.__name__))
        
    # Return the results
    return results    
    

def evaluate(path_output, y):
       
    # get predictions
    y_pred = pd.read_csv(os.path.join(path_output, 'test.csv.out'), header=None) 
    
    # round probability predictions to calculate the accuracy
    predictions = [round(num) for num in y_pred.squeeze().values]
    print('Accuracy: {}'.format(accuracy_score(y_validate.reset_index(drop=True), predictions)))
    
    # print confusion metrix
    print('\nConfusion matrix : \n{}'.format(confusion_matrix(y_validate, predictions)))
    
    # ROC needs that we pass the probability
    # https://stackoverflow.com/questions/62192616/strange-behavior-of-roc-auc-score-roc-auc-auc
    print('\nAUC: {}'.format(roc_auc_score(y, y_pred)))
    