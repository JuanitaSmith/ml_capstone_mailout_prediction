# data cleansing functions for Arvato Financial Services datasets

import time
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from src.config import filename_drop_missing, filename_ohn_supervised, filename_ohn_unsupervised, \
    filename_drop_corr_supervised, filename_drop_corr_unsupervised, filename_gridcolumns, filename_imputer_mean, \
    filename_imputer_most
from src.visuals import plot_features


def ohn(df, columns):
    """
    A function to apply one-hot encoding on selected categorical variables

    """

    # Print the number of features before one-hot encoding
    encoded = list(df.columns)
    print("{} total features BEFORE one-hot encoding".format(len(encoded)))

    azdias_transformed = pd.get_dummies(df, columns=columns)

    # Print the number of features after one-hot encoding
    encoded = list(azdias_transformed.columns)
    print("{} total features AFTER one-hot encoding".format(len(encoded)))

    return azdias_transformed


def binarizer(x):
    """
    A function to transform a categorical variable into a binary variable
    """

    binarize = {'W': 1, 'O': 0}
    if pd.isna(x):
        return x
    else:
        return binarize.get(x, np.nan)


def decade(x):
    """
    Extract only the century (first 2 digits) from a data string

    Original values:
    
    1 	40ies - war years (Mainstream, O+W)
    2 	40ies - reconstruction years (Avantgarde, O+W)
    3 	50ies - economic miracle (Mainstream, O+W)
    4 	50ies - milk bar / Individualisation (Avantgarde, O+W)
    5 	60ies - economic miracle (Mainstream, O+W)
    6 	60ies - generation 68 / student protestors (Avantgarde, W)
    7 	60ies - opponents to the building of the Wall (Avantgarde, O)
    8 	70ies - family orientation (Mainstream, O+W)
    9 	70ies - peace movement (Avantgarde, O+W)
    10	80ies - Generation Golf (Mainstream, W)
    11	80ies - ecological awareness (Avantgarde, W)
    12	80ies - FDJ / communist party youth organisation (Mainstream, O)
    13	80ies - Swords into ploughshares (Avantgarde, O)
    14	90ies - digital media kids (Mainstream, O+W)
    15	90ies - ecological awareness (Avantgarde, O+W)
    
    
    New values
    1 = 40
    2 = 50
    3 = 60
    4 = 70
    5 = 80
    6 = 90
    
    """

    decade_lookup = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5,
                     12: 5, 13: 5, 14: 6, 15: 6}
    if pd.isna(x):
        return x
    else:
        return decade_lookup[x]


def movement(x):
    """
    Extract the dominating movement of person's youth by converting numeric numbers to binary indicator

    New values:
    1 = Mainstream
    0 = Avantgarde

    Args:
        x (object): Numeric number with in range 1 -15
    """

    movement_lookup = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0}
    if pd.isna(x):
        return x
    else:
        return movement_lookup.get(x, np.nan)


def neighborhood_quality(x):
    """
    Strip field 'WOHNLAGE' to be an interval representing only the neighborhood quality
    
    If it's a rural area, fill the value with NaN
    Re-order values from poor to good
    
    Original values of 'WOHNLAGE':
    0: no score calculated (should be converted to NaN already)
    1: very good neighborhood
    2: good neighborhood
    3: average neighborhood
    4: poor neighborhood
    5: very poor neighborhood
    7: rural neighborhood
    8: new building in rural neighborhood   
    """

    neighborhood = {0: np.nan, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: np.nan, 8: np.nan}
    return neighborhood.get(x, np.nan)


def rural_flag(x):
    """
    Strip field 'WOHNLAGE' to be a binary flag representing if an area is a rural area or not
    
    If it's rural area, fill with 1, otherwise 0
    
    Original values of 'WOHNLAGE':

    0: no score calculated
    1: very good neighborhood
    2: good neighborhood
    3: average neighborhood
    4: poor neighborhood
    5: very poor neighborhood
    7: rural neighborhood
    8: new building in rural neighborhood   
    """

    if pd.isna(x):
        return x
    elif x >= 7:
        return 1
    else:
        return 0


def wealth(x):
    """
    Function that returns the first digit from a string as the new value
    
    Meaning of new values:
    1: Wealthy Households 
    2: Prosperous Households
    3: Comfortable Households 
    4: Less Affluent Households
    5: Poorer Households
    """

    if pd.isna(x):
        return x
    else:
        return int(str(x)[0])


def lifestyle(x):
    """
    Function that returns the 2nd digit from a string as the new value
    
    Meaning of new values:
    1: Pre-Family Couples & Singles
    2: Young Couples With Children
    3: Families With School Age Children
    4: Older Families &  Mature Couples
    5: Elders In Retirement
    """

    if pd.isna(x):
        return x
    else:
        return int(str(x)[1])


def building_type_residential(x):
    """
    Strip fields PLZ8_BAUMAX/KBA05_BAUMAX to be an interval representing only number of family houses
    
    If it's a business area, fill the value with 0
    
    Original values of '*BAUMAX':

    1: mainly 1-2 family houses
    2: mainly 3-5 family houses
    3: mainly 6-10 family houses
    4: mainly > 10 family houses
    5: mainly business buildings 
    """

    building = {0: np.nan, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0}

    return building.get(x, np.nan)


def building_type_business(x):
    """
    Strip fields PLZ8_BAUMAX/KBA05_BAUMAX to be an binary flag indicate if this is a business or not
       
    Original values of '*BAUMAX':

    1: mainly 1-2 family houses
    2: mainly 3-5 family houses
    3: mainly 6-10 family houses
    4: mainly > 10 family houses
    5: mainly business buildings 
    """

    business = {0: np.nan, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}

    return business.get(x, np.nan)


def change_ranking_transactions(x):
    """
    Re-arrange ordinal values from high-to-low, to low-to-high

    1	highest activity within the last 12 months
    2	very high activity within the last 12 months
    3	high activity within the last 12 months
    4	increased activity within the last 12 months
    5	slightly increased activity within the last 12 months
    6	activity elder than 1 year
    7	activity elder than 1,5 years
    8	activity elder than 2 years
    9	activity elder than 3 years
    10	no transactions known
    """

    rerank = {1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0}
    return rerank.get(x, np.nan)


def change_ranking_grid(x):
    """
    Re-arrange ordinal values from high-to-low, to low-to-high

    0	no transaction known
    1	Multibuyer 0-12 months
    2	Doublebuyer 0-12 months
    3	Singlebuyer 0-12 months
    4	Multi-/Doublebuyer 13-24 months
    5	Singlebuyer 13-24 months
    6	Buyer > 24 months
    7	Prospects > 24 months
    """

    rerank = {0: 0, 7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7}
    return rerank.get(x, np.nan)


def enhance_konsumtyp_max(x):
    """
    Re-classify values to have a better scaling affect
    
    Unique values in the dataset are 1, 2, 3, 4, 8, 9
    This field is not present in feature description so meaning is unclear, but seem to be an important feature
    When comparing it with D19_KONSUMTYP, it can be inferred that 9 means NaN - this will be classified
    to -1
    We can also gather that 8 means an inactive customer
    
    When imputing missing values, it comes up with value predictions = 5 sometimes
    As 5 is not valid possible value, this cause problems when we one hot encode
    Therefore reduce the range from 1-8 to 1-5 to mitigate this
    
    """
    konsumtyp_max = {1: 1, 2: 2, 3: 3, 4: 4, 8: 5, 9: -1}
    return konsumtyp_max.get(x, np.nan)


def enhance_konsumtyp(x):
    """
    Re-classify values to have a better scaling affect, don't leave gaps in numbers

    Existing values:
    1	Universal
    2	Versatile
    3	Gourmet
    4	Family
    5	Informed 
    6	Modern
    9	Inactive -> change to 7 to have a better imputing result
    
    """
    konsumtyp = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 7}
    return konsumtyp.get(x, np.nan)


def grid_binarizer(x):
    """
    Strip field 'PLZ8-BAUMAX' to be an interval representing only number of family houses
    
    If it's a business area, fill the value with 0
    Re-order values from poor to good
    
    Original values:

    0	no transaction known
    1	Multibuyer 0-12 months
    2	Doublebuyer 0-12 months
    3	Singlebuyer 0-12 months
    4	Multi-/Doublebuyer 13-24 months
    5	Singlebuyer 13-24 months
    6	Buyer > 24 months
    7	Prospects > 24 months
    """

    if pd.isna(x):
        return x
    elif x == 0:
        return 0
    else:
        return 1


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """

    print('\nTriggering memory optimization.......\n')

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def get_columns(filename):
    """
    A function to read a csv file and build a list of columns
    """

    try:
        data = pd.read_csv(filename, header=None)
        return data[0].to_list()
    except:
        print('file {} does not exist'.format(filename))


def outlier_correcter(cols, data, direction='lower', lower_quantile=0.01, upper_quantity=0.99):
    """
    Remove lower or upper outliers in lower or upper percentiles

    Args:
        cols: column names from which outliers should be removed
        data: pandas dataframe
        direction: contain values 'lower' or 'higher'
        lower_quantile: set lower percentile threshold to trim
        upper_quantity: set upper percentile threshold to trim

    Returns:
        dataframe: updated columns with outliers removed

    """
    for col in cols:
        #         data[col] = data[col].apply(lambda x: np.log(x + 0.1))
        #         data[col].plot(kind='hist', title=col, alpha=0.6)
        if direction == 'lower':
            data[col] = data[col].clip(lower=data[col].quantile(lower_quantile))
        elif direction == 'upper':
            data[col] = data[col].clip(upper=data[col].quantile(upper_quantity))
        #         data[col].plot(kind='hist', title=col, alpha=0.6)
        plt.show()
    return data


def feature_engineering(data, visualize=False):
    """
    Routine to split up or merge features as required

    Args:
        data: dataframe containing all data
        visualize: To show values in columns before and after transformation

    Returns: enhanced dataframe with new, enhanced for removed features

    """

    # Split WOHNLAGE into 2 features capturing the residential neighborhood quality and residential vs rural binary flag
    print('Adding few features WOHNLAGE_QUALITY and WOHNLAGE_RURAL ..........')
    data['WOHNLAGE_QUALITY'] = data.loc[:, 'WOHNLAGE'].apply(neighborhood_quality)
    data['WOHNLAGE_RURAL'] = data.loc[:, 'WOHNLAGE'].apply(rural_flag)
    if visualize:
        plot_features(['WOHNLAGE_QUALITY', 'WOHNLAGE_RURAL', 'WOHNLAGE'],
                      data[['WOHNLAGE', 'WOHNLAGE_QUALITY', 'WOHNLAGE_RURAL']],
                      title='Transformed WOHNLAGE',
                      rect=[0.1, 0.1, 0.7, 0.95])

    # cleanup "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
    print('Adding few features PRAEGENDE_JUGENDJAHRE_DECADE and PRAEGENDE_JUGENDJAHRE_MOVEMENT ..........')
    data['PRAEGENDE_JUGENDJAHRE_DECADE'] = data.loc[:, 'PRAEGENDE_JUGENDJAHRE'].apply(decade)
    data['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = data.loc[:, 'PRAEGENDE_JUGENDJAHRE'].apply(movement)
    if visualize:
        plot_features(['PRAEGENDE_JUGENDJAHRE_DECADE', 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'],
                      data[['PRAEGENDE_JUGENDJAHRE_DECADE', 'PRAEGENDE_JUGENDJAHRE_MOVEMENT']],
                      title='Transformed PRAEGENDE_JUGENDJAHRE',
                      rect=[0.1, 0.1, 0.7, 0.95])

    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    print('Adding few features CAMEO_INTL_2015_WEALTH and CAMEO_INTL_2015_LIFESTAGE ..........')
    data['CAMEO_INTL_2015_WEALTH'] = data.loc[:, 'CAMEO_INTL_2015'].apply(wealth)
    data['CAMEO_INTL_2015_LIFESTAGE'] = data.loc[:, 'CAMEO_INTL_2015'].apply(lifestyle)
    if visualize:
        plot_features(['CAMEO_INTL_2015_WEALTH', 'CAMEO_INTL_2015_LIFESTAGE'],
                      data[['CAMEO_INTL_2015_WEALTH', 'CAMEO_INTL_2015_LIFESTAGE']],
                      title='Transformed CAMEO_INTL_2015',
                      rect=[0.1, 0.1, 0.7, 0.95])

    # clean-up building type to only rank family houses and build a new business flag
    print('Cleaning PLZ8_BAUMAX, KBA05_BAUMAX .......... ')
    data['PLZ8_BAUMAX_BUSINESS'] = data.loc[:, 'PLZ8_BAUMAX'].apply(building_type_business)
    data['KBA05_BAUMAX_BUSINESS'] = data.loc[:, 'KBA05_BAUMAX'].apply(building_type_business)
    data['PLZ8_BAUMAX'] = data.loc[:, 'PLZ8_BAUMAX'].apply(building_type_residential)
    data['KBA05_BAUMAX'] = data.loc[:, 'KBA05_BAUMAX'].apply(building_type_residential)

    if visualize:
        plot_features(['PLZ8_BAUMAX'],
                      data[['PLZ8_BAUMAX']],
                      title='Transformed PLZ8_BAUMAX',
                      rect=[0.1, 0.1, 0.7, 0.95])

    # Binarize column
    print('Cleaning OST_WEST_KZ .......... ')
    data['OST_WEST_KZ'] = data.loc[:, 'OST_WEST_KZ'].apply(binarizer)

    # reduce range of D19_KONSUMTYP_MAX  
    print('Reducing range for D19_KONSUMTYP_MAX from 1-8 to 1-5')
    data['D19_KONSUMTYP_MAX'] = data.loc[:, 'D19_KONSUMTYP_MAX'].apply(enhance_konsumtyp_max)

    # reduce range of D19_KONSUMTYP  
    print('Reducing range for D19_KONSUMTYP from 1-9 to 1-7')
    data['D19_KONSUMTYP'] = data.loc[:, 'D19_KONSUMTYP'].apply(enhance_konsumtyp)

    # drop columns we don't want to keep - due manual decisions made and above transformations
    columns_to_drop = ['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015']
    data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    #   change the values of these columns for low to high, so that transactions = 0 and not 10
    print('Rerank ranges for banking transactions')
    reclassify = ['D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM', 'D19_BANKEN_ONLINE_DATUM',
                  'D19_GESAMT_DATUM', 'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
                  'D19_TELKO_DATUM', 'D19_TELKO_OFFLINE_DATUM', 'D19_TELKO_ONLINE_DATUM',
                  'D19_VERSAND_DATUM', 'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_ONLINE_DATUM',
                  'D19_VERSI_ONLINE_DATUM', 'D19_VERSI_OFFLINE_DATUM', 'D19_VERSI_DATUM']
    for col in reclassify:
        if col in list(data.columns):
            data[col] = data.loc[:, col].apply(change_ranking_transactions)

    data = ohn_categorial(data)

    # Drop any decimals in column headings as a result of ohn on float columns that contain NaN
    # This will avoid example column names of KK_KUNDENTYP_1 vs KK_KUNDENTYP_1.0    
    data.columns = data.columns.str.replace(".0", "", regex=False)

    return data


def ohn_categorial(df):
    """
    One-hot encoding for categorical columns only
    """

    columns_to_ohn = list(df.loc[:, df.dtypes == 'object'].columns)
    print('Categorial columns to one hot encode: {}'.format(columns_to_ohn))
    df = ohn(df, columns_to_ohn)
    return df


def ohn_additional(df, kind='supervised'):
    """
    Get columns from a csv config file that should one-hot encoded

    Args:
        df: dataset containing all data
        kind: 'supervised' or 'unsupervised' - different columns can be configured for supervised vs unsupervised

    Returns: dataset with configured columns one-hot encoded

    """

    if kind == 'supervised':
        columns_to_ohn = get_columns(filename_ohn_supervised)
    elif kind == 'unsupervised':
        columns_to_ohn = get_columns(filename_ohn_unsupervised)

    remaining_columns_to_ohn = [col for col in columns_to_ohn if col in df.columns]
    print('Additional columns to one hot encode: {}'.format(remaining_columns_to_ohn))

    df = ohn(df, remaining_columns_to_ohn)

    return df


def fit_imputer(df, tolerance=0.2, verbose=2, max_iter=20, nearest_features=20, imputation_order='ascending',
                initial_strategy='most_frequent'):
    """
    A function to train an IterativeImputer using machine learning

    Args:
        df: dataset to impute
        tolerance: Tolerance of stopping function
        verbose: Verbosy flag, controls the debug messages that are issued as functions are evaluated
        max_iter: Maximum number of imputation rounds
        nearest_features: Number of other features to use to estimate the missing values
        imputation_order: ascending or descending - the order in which the features will be imputed
        initial_strategy: e.g. 'most_frequent' or 'mean'

    Returns: dataset with no missing values

    """

    start = time.time()

    # restrict the values to be predicted to a min / max range
    minimum_before = list(df.iloc[:, :].min(axis=0))
    maximum_before = list(df.iloc[:, :].max(axis=0))

    imputer = IterativeImputer(random_state=0,
                               imputation_order=imputation_order,
                               n_nearest_features=nearest_features,
                               initial_strategy=initial_strategy,
                               max_iter=max_iter,
                               min_value=minimum_before,
                               max_value=maximum_before,
                               skip_complete=True,
                               tol=tolerance,
                               verbose=verbose)

    imputer.fit(df)

    end = time.time()
    print('Execution time for IterativeImputer: {} sec'.format(end - start))

    return imputer


def custom_imputer_transform(df):
    """
    Impute missing values using machine learning where other fields are used to predict the missing values
    
    2 Imputers was pre-trained
    For categorical features using `most_frequent` as a base
    For numeric features using `mean` as a base
    """

    numerical_columns = ['ANZ_HAUSHALTE_AKTIV',
                         'ANZ_HH_TITEL',
                         'ANZ_KINDER',
                         'ANZ_PERSONEN',
                         'EINGEZOGENAM_HH_JAHR',
                         'GEBURTSJAHR',
                         'KBA13_ANZAHL_PKW',
                         'MIN_GEBAEUDEJAHR',
                         'VERDICHTUNGSRAUM']

    # load back the fitted imputers
    imputer_mean = load(open(filename_imputer_mean, 'rb'))
    imputer_most = load(open(filename_imputer_most, 'rb'))

    # Impute numeric data
    azdias_imputed_iterative_mean = df.copy(deep=True)
    azdias_imputed_iterative_mean.iloc[:, :] = imputer_mean.transform(azdias_imputed_iterative_mean)

    # Impute categorial and ordinal data
    azdias_imputed_iterative_most = df.copy(deep=True)
    azdias_imputed_iterative_most.iloc[:, :] = imputer_most.transform(azdias_imputed_iterative_most)

    # Construct the dataset
    azdias_imputed_iterative = azdias_imputed_iterative_most.copy(deep=True)
    azdias_imputed_iterative[numerical_columns] = azdias_imputed_iterative_mean[numerical_columns]

    return azdias_imputed_iterative


def simple_imputer(df, columns, strategy='mean'):
    """
    Impute missing values using SimpleImputer to set a baseline for InterativeImputer

    """

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(df[columns])
    df.loc[:, columns] = imputer.transform(df[columns])
    return df


def summarize_grid_columns(df_grid):
    """
    Summarize all grid columns to indicate the type of buyer and when the last transaction was
    
    Original values:
    0	no transaction known
    1	Multibuyer 0-12 months
    2	Doublebuyer 0-12 months
    3	Singlebuyer 0-12 months
    4	Multi-/Doublebuyer 13-24 months
    5	Singlebuyer 13-24 months
    6	Buyer > 24 months
    7	Prospects > 24 months
    
    New 'BUYER TYPE' mapping:
    0 -> 0 or Unknown
    1 -> Prospect
    2 -> Single Buyer > 2 years
    3 -> Single Buyer < 2 years
    4 -> Multi Buyer
    
    New 'YEAR_LAST_ACTIVE' mapping:
    0: 0 or unknown
    1: 1-3 - transactions in last year
    2: 4-5 - tranasaction between 1-2 years
    3: 6-7 - transactions > 2 years
    
    """

    buyer_type = {0: np.nan, 1: 'MULTIBUYER', 2: 'MULTIBUYER', 3: 'SINGLEBUYER', 4: 'MULTIBUYER',
                  5: 'SINGLEBUYER', 6: 'BUYER', 7: 'PROSPECT'}

    buyer_type_label_encoder = {'MULTIBUYER': 4, 'SINGLEBUYER': 3, 'BUYER': 2, 'PROSPECT': 1, np.nan: 0}

    buyer_horizon = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 0: 0, np.nan: 0}

    df_grid['GRID_BUYER_TYPE'] = df_grid.min(axis=1).map(buyer_type).map(buyer_type_label_encoder)
    df_grid['GRID_YEAR_LAST_ACTIVE'] = df_grid.min(axis=1).map(buyer_horizon)

    return df_grid['GRID_BUYER_TYPE'], df_grid['GRID_YEAR_LAST_ACTIVE']


def binarize_grid_columns(df):
    """ 
    Simplify grid columns by setting a flag if they bought in this category or not. Prospect will be excluded
    
    As there are generally not a lot of data available for grid columns, null values is causing bias and overfit.
    Neutralize the power of these fields
    
    Original values:
    0	no transaction known
    1	Multibuyer 0-12 months
    2	Doublebuyer 0-12 months
    3	Singlebuyer 0-12 months
    4	Multi-/Doublebuyer 13-24 months
    5	Singlebuyer 13-24 months
    6	Buyer > 24 months
    7	Prospects > 24 months 
    """

    back_to_binary = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, np.nan: 0}

    for col in df:
        df[col] = df[col].map(back_to_binary)

    return df


def clean(df, df_extended_na, rows_threshold=0.65, kind='unsupervised', impute=False):
    """
    Main cleaning function entry point

    Args:
        df: dataset to clean
        df_extended_na: dataset where all types of missing values are converted to np.nan
        rows_threshold: rows to delete when % missing values exceed this threshold
        kind: supervised or unsupervised - different approaches are configured what to delete or encode
        impute: If set to False imputation will be bypassed

    Returns:
        dataframe: cleaned dataframe

    """

    # visualize missing values in columns before we will clean it up
    missing_columns_proportion = round(df.isna().mean() * 100, 0).sort_values(ascending=False)
    missing_columns_proportion[missing_columns_proportion > 30].plot(kind="bar",
                                                                     figsize=(14, 6),
                                                                     title='% Missing Values in Columns BEFORE',
                                                                     alpha=0.7,
                                                                     width=1,
                                                                     color='b')
    plt.show()

    # As grid columns by category has low data volume, summarize the columns and build one new column instead
    # We have to summarize the grid columns here before we drop a lot of the columns in the next step
    print('Summarize GRID columns.....')
    grid_columns = get_columns(filename_gridcolumns)
    df['GRID_BUYER_TYPE'], df['GRID_YEAR_LAST_ACTIVE'] = summarize_grid_columns(df_extended_na[grid_columns].copy())

    # Binarize grid columns
    print('Binarize GRID columns.....')
    df[grid_columns] = binarize_grid_columns(df_extended_na[grid_columns].copy())

    # Remove columns not needed due to too many missing values
    columns_to_drop = get_columns(filename_drop_missing)
    df.drop(columns_to_drop, inplace=True, axis=1, errors='ignore')
    print('\nColumns dropped due to missing values {}'.format(columns_to_drop))

    missing_columns_proportion = round(df.isna().mean() * 100, 0).sort_values(ascending=False)
    missing_columns_proportion[missing_columns_proportion > 30].plot(kind="bar",
                                                                     figsize=(14, 6),
                                                                     title='% Missing Values in Columns AFTER',
                                                                     ylim=(0, 100),
                                                                     alpha=0.7,
                                                                     width=1,
                                                                     color='r')
    plt.show()

    # Remove rows with missing values exceeding threshold
    print('Shape BEFORE row deletion: {}'.format(df.shape))
    (df.isna().mean(axis=1) * 100).plot(kind='hist', title='% missing data in rows', figsize=(8, 6), alpha=0.7)
    df = df.loc[df.isna().mean(axis=1) < rows_threshold, :]
    (df.isna().mean(axis=1) * 100).plot(kind='hist', title='% missing data in rows', figsize=(8, 6), alpha=0.7)
    plt.legend(['Before', 'After'])
    plt.ylabel('% missing')
    plt.show()
    print('Shape AFTER row deletion: {} with threshold of: {}'.format(df.shape, rows_threshold))

    # Feature engineering
    df_transformed = feature_engineering(df.copy())

    # Impute missing values using a pretrained ML Interative Imputer, we do this before removing correlated columns
    # 2 imputers was trained one using most_frequent and one using mean for numeric fields
    if impute:
        df_imputed = custom_imputer_transform(df_transformed.copy())
    else:
        df_imputed = df_transformed.copy()

    # After imputing, it's now possible to convert all columns from float to integer as no more null values exist
    # but we need to round first. Imputing to integer would give us better columns names during one-hot encoding
    try:
        df_imputed = df_imputed.round(0).astype(np.int16, copy=False, errors='raise')
    except:
        print('/nIMPORTANT: Not possible to convert datatypes to integer as they contain NaN')

    # Remove highly correlated columns
    columns_to_drop = []
    if kind == 'unsupervised':
        columns_to_drop = get_columns(filename_drop_corr_unsupervised)
    elif kind == 'supervised':
        columns_to_drop = get_columns(filename_drop_corr_supervised)

    if len(columns_to_drop) > 0:
        print('\nDrop additional correlated columns {}\n'.format(columns_to_drop))
        df_imputed.drop(columns_to_drop, inplace=True, errors='ignore', axis=1)

        # one-hot encode some last fields
    df_imputed = ohn_additional(df_imputed, kind)

    # Drop any decimals in column headings as a result of ohn on float columns that contain NaN
    # This will avoid example column names of KK_KUNDENTYP_1 vs KK_KUNDENTYP_1.0    
    df_imputed.columns = df_imputed.columns.str.replace(".0", "", regex=False)

    # correct the distribution of year columns as it seems to have low values 1900 - 1940 which adds no value
    print('Correct distributions of numerical fields....')
    df_imputed = outlier_correcter(['EINGEZOGENAM_HH_JAHR'], df_imputed)

    return df_imputed
