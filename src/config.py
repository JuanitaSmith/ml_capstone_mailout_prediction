# pointers to raw data files

filename_levels = 'DIAS Information Levels - Attributes 2017.xlsx'
filename_levels_sheet = 'Komplett'

filename_attributes = 'DIAS Attributes - Values 2017 Enhanced.xlsx'
filename_attributes_sheet = 'Tabelle1'

filename_demographics = 'Udacity_AZDIAS_052018.csv'
filename_demographics_delimiter = ';'

filename_customer = 'Udacity_CUSTOMERS_052018.csv'
filename_customer_delimiter = ';'

filename_enhanced_features = 'demographic_attributes.xlsx'

filename_mailout_test = 'Udacity_MAILOUT_052018_TEST.csv'
filename_mailout_train = 'Udacity_MAILOUT_052018_TRAIN.csv'
filename_mailout_delimiter = ';'

# models
filename_imputer = '../data/models/imputer.pkl'
filename_imputer_mean = '../data/models/imputer_mean.pkl'
filename_imputer_most = '../data/models/imputer_most.pkl'
filename_kmeans = '../data/models/kmeans.pkl'
filename_pca = '../data/models/pca.pkl'
filename_scaler_supervised = '../data/models/scaler_supervised.pkl'
filename_scaler_unsupervised = '../data/models/scaler_unsupervised.pkl'
filename_model_featuremap = '../data/models/xgb.fmap'

# configuration files
filename_ohn_supervised = '../data/config/ohn_supervised.csv'
filename_ohn_unsupervised = '../data/config/ohn_unsupervised.csv'
filename_drop_missing = '../data/config/drop.csv'
filename_drop_corr_supervised = '../data/config/drop_corr_supervised.csv'
filename_drop_corr_unsupervised = '../data/config/drop_corr_unsupervised.csv'
filename_gridcolumns = '../data/config/grid.csv'
filename_customer_clusters = '../data/clean/customer_clusters.csv'

# points to prepared training and testing files
filename_train_csv = 'train.csv'
filename_validation_csv = 'validation.csv'
filename_test_csv = 'test.csv'
filename_train_parquet = 'train.parquet'
filename_validation_parquet = 'validation.parquet'
filename_test_parquet = 'test.parquet'

# path to directories where raw, clean, config data and model outputs are stored
data_dir = '../data'
path_raw = '../data/raw'
path_clean = '../data/clean'
path_train = '../data/train/train'
path_validation = '../data/train/validation'
path_validation_test = '../data/train/test'
path_test = '../data/test'
path_output = '../data/output'
path_output_validation = '../data/output/validation'
path_output_test = '../data/output/test'
path_output_kaggle = '../data/output/kaggle'
path_model = '../data/models'
path_config = '../data/config'

# SAGEMAKER SET-UP

# AWS role when working locally
# Udacity
# sagemaker_role = 'arn:aws:iam::575658184623:role/service-role/AmazonSageMaker-ExecutionRole-20210428T162000'

# Own account
sagemaker_role = 'arn:aws:iam::554017193854:role/service-role/AmazonSageMaker-ExecutionRole-20210428T170819'

# default bucket name
prefix = 'arvato_customer_segmentation'
