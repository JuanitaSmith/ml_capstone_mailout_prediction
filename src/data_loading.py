import pandas as pd

from src.config import path_raw, filename_levels, filename_attributes, filename_levels_sheet, filename_attributes_sheet, \
    filename_customer_delimiter


def load_levels(filename, sheet):
    """
    Load attribute information levels from excel into a dataframe
    
    Args:
       filename (string): name of the attributes level file
       sheet (string): sheet from the excel to read
       
    Returns:
       dataframe: features mapped to levels
    """

    levels = pd.read_excel(filename,
                           sheet_name=sheet,
                           engine='openpyxl',
                           skiprows=1)

    # copy level value to cells below, it's normally only filled in for the first line in a category
    levels.fillna(method='ffill', axis=0, inplace=True)

    # drop empty columns
    levels.dropna(axis=1, how='all', inplace=True)

    # some levels contains 2 column names in 1 line, split and explode it so that one row contains only one attribute
    levels['Attribute'] = levels['Attribute'].astype(str).str.split('   ', n=1)
    levels = levels.explode('Attribute')

    # remove leading zero's after the split
    levels['Attribute'] = levels['Attribute'].str.strip()

    # set column 'Attribute' as the index
    levels = levels.set_index('Attribute')

    # build a dictionary we can use to map an attribute to a level later on
    levels_dict = levels['Information level'].to_dict()

    return levels, levels_dict


def load_attribute_descriptions(filename, sheet):
    """
    Load feature descriptions

    Args:
       filename (string): name of the attributes level file
       sheet (string): sheet from the excel to read

    Returns:
        attributes: dataset containing feature descriptions
        missing_dict: dictionary contain true missing values
        missing_dict2: dictionary containing a different kind of missing values where values are 0 (but not missing)
        missing_df:  dataset containing 0 values for transactional values only

    """

    attributes = pd.read_excel(filename,
                               sheet_name=sheet,
                               engine='openpyxl',
                               skiprows=1,
                               na_values=['…'])

    # forward fill column values
    attributes.fillna(method='ffill', axis=0, inplace=True)

    # drop empty columns
    attributes.dropna(axis=1, how='all', inplace=True)

    # Build a missing values dictionary containing only the missing values for each column
    missing_values = attributes.loc[attributes['Meaning'].str.contains('unknown'), ['Attribute', 'Value']].set_index(
        ['Attribute'])
    missing_values['Value'] = missing_values['Value'].astype(str).str.split(', ')
    missing_dict = missing_values['Value'].to_dict()

    # build a second missing values dictionary to treat additional values as unknown rather that 0
    missing_list = ['unknown', 'no transactions known', 'no transaction known', 'no Online-transactions']
    missing_df = attributes.loc[
        attributes['Meaning'].str.contains('|'.join(missing_list)), ['Attribute', 'Value']].set_index(['Attribute'])
    missing_df['Value'] = missing_df['Value'].astype(str).str.split(', ')
    missing_dict2 = missing_df['Value'].to_dict()

    missing_list_ekstra = ['no transactions known', 'no transaction known', 'no Online-transactions']
    missing_df = attributes.loc[
        attributes['Meaning'].str.contains('|'.join(missing_list_ekstra)), ['Attribute', 'Value']].set_index(
        ['Attribute'])
    missing_df

    return attributes, missing_dict, missing_dict2, missing_df


def load_dataset(filename, delimiter, na_values, reset_na=None, visualize=False):
    """
    Load data with enhanced missing values

    Args:
        filename: dataset contain demographics values
        delimiter: delimiter
        na_values: dictionary containing missing values that needs to be
        reset_na: dictionary containing a different kind of missing values where values are 0 (but not missing)
        visualize: print unique values of certain columns after imputing

    Returns:

    """

    # EINGEFUEGT_AM is a date/time stamp. Using google translate it's assumed it's the date the customer was added
    # to the database. Trimmed this field down to year
    custom_date_parser = lambda x: pd.to_datetime(x, errors='ignore').strftime('%Y')

    # enhance missing values definition
    data = pd.read_csv(filename,
                       sep=delimiter,
                       na_values=na_values,
                       parse_dates=['EINGEFUEGT_AM'],
                       date_parser=custom_date_parser)

    # convert date into year
    data['EINGEFUEGT_AM'] = data['EINGEFUEGT_AM'].dt.year

    if len(reset_na) > 0:
        for i, row in reset_na.iterrows():
            if i in list(data.columns):
                data[i].fillna(row['Value'], inplace=True)

    if visualize:
        print('\nUnique values:\n')
        print('\nAGER_TYPE: {}'.format(list(data.AGER_TYP.unique())))
        print('\nCAMEO_INTL_2015: {}'.format(list(data.CAMEO_INTL_2015.unique())))
        print('\nCAMEO_DEUG_2015: {}'.format(list(data.CAMEO_DEUG_2015.unique())))
        print('\nCAMEO_DEU_2015: {}'.format(list(data.CAMEO_DEU_2015.unique())))
        print('\nEINGEFEUGT_AM: {}'.format(list(data.EINGEFUEGT_AM.unique())))
        print('\nD19_GESAMT_DATUM: {}'.format(list(data.D19_GESAMT_DATUM.unique())))

    return data


def get_data(data_path):
    """
    General entry point to import all datasets

    Args:
        data_path: path where the dataset can be read from

    Returns:
        dataframe containing the data

    """

    # Get data levels
    path = "{}/{}".format(path_raw, filename_levels)
    levels, levels_dict = load_levels(path, filename_levels_sheet)

    # get attribute descriptions and missing data values
    path = "{}/{}".format(path_raw, filename_attributes)
    attributes, missing_dict, missing_dict2, missing_df = load_attribute_descriptions(path, filename_attributes_sheet)

    # Reading main dataset replacing missing values
    df = load_dataset(filename=data_path,
                      delimiter=filename_customer_delimiter,
                      na_values=missing_dict,
                      reset_na=missing_df,
                      visualize=False)

    # Reading main dataset again, treating transaction fields = 0, temporarily as missing values
    df_extended_na = load_dataset(data_path,
                                  delimiter=filename_customer_delimiter,
                                  na_values=missing_dict2,
                                  reset_na=[],
                                  visualize=False)

    df.set_index('LNR', inplace=True, verify_integrity=True)
    df_extended_na.set_index('LNR', inplace=True, verify_integrity=True)

    return df, df_extended_na
