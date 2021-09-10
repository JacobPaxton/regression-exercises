import os
from env import host, password, username

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_db_url(db_name, username=username, hostname=host, password=password):
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

def get_zillow():
    """
        Checks for local copy of Codeup's Zillow information, and if it doesn't exist,
            Grabs 'Single Family Residential' home data from Codeup's 'zillow' database,
        Then returns compiled dataframe.
        
    """
    # Local copy check and grab
    if not os.path.isfile('zillow.csv'):
        url = get_db_url(db_name='zillow')
        query = """
                SELECT bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet, 
                    taxvaluedollarcnt,
                    yearbuilt, 
                    taxamount, 
                    fips,
                    propertylandusetypeid, 
                    propertylandusedesc
                FROM properties_2017
                JOIN propertylandusetype using(propertylandusetypeid)
                WHERE propertylandusedesc = 'Single Family Residential';
            """
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv')
    
    return pd.read_csv('zillow.csv', index_col=0)

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def wrangle_zillow():
    """
        Imports zillow data, then:
            Drops nulls and duplicates,
            Removes outliers using Inter-Quartile Rule,
            Fixes column dtypes and cleans values,
            Drops unnecessary columns,
            Renames columns for easier reading,
            Changes 'yearbuilt column' to 'age', then
         returns df
    """

    # Import zillow data
    zillow = get_zillow()

    # Drop 12,628 rows with null values from 2,152,863 total observations
    zillow = zillow.dropna()

    # Drop 10,021 duplicate rows from 2,140,235 total observations
    zillow = zillow.drop_duplicates()

    # Remove outliers, dataframe now 1,790,213 total observations
    col_list = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
                'taxvaluedollarcnt', 'yearbuilt', 'taxamount']
    zillow = remove_outliers(df=zillow, k=1.5, col_list=col_list)

    # Convert 'fips' from float to county code
    # EX: float64 6037.0 -> object '06037'
    zillow['fips'] = zillow.fips.astype('int').astype('str').apply(lambda x: '0' + x)

    # Convert 'yearbuilt' from float to ordinal
    # EX: float64 1954.0 -> object '1954'
    zillow['yearbuilt'] = zillow.yearbuilt.astype('int').astype('str')

    # Convert multiple columns to int (all values are whole numbers)
    zillow['bedroomcnt'] = zillow.bedroomcnt.astype('int')
    zillow['bathroomcnt'] = zillow.bathroomcnt.astype('int')
    zillow['calculatedfinishedsquarefeet'] = zillow.calculatedfinishedsquarefeet.astype('int')
    zillow['taxvaluedollarcnt'] = zillow.taxvaluedollarcnt.astype('int')

    # Drop unnecessary columns
    zillow.drop(columns=['propertylandusetypeid','propertylandusedesc'], inplace=True)

    # Rename remaining columns
    zillow.rename(columns={'bedroomcnt':'beds',
                            'bathroomcnt':'baths',
                            'calculatedfinishedsquarefeet':'area',
                            'taxvaluedollarcnt':'worth',
                            'yearbuilt':'built',
                            'taxamount':'tax',
                            'fips':'locality'}, inplace=True)

    # Add new 'age' column (most recent after outliers removed is 2006)
    zillow['age'] = 2006 - zillow['built'].astype('int')
    
    # Drop 'built' column
    zillow.drop(columns='built', inplace=True)

    # Pass back dataframe
    return zillow

def dummy_zillow(df):
    """ Turns locality column of zillow data into dummy columns """
    df = pd.get_dummies(df, columns=['locality'], drop_first=True)
    return df

def train_val_test(df):
    """
        Splits a dataframe into train, validate, and test, then
        Returns the three dataframes.
    """

    train_validate, test = train_test_split(df, test_size=0.2, 
                                                random_state=123)
    
    train, validate = train_test_split(train_validate, test_size=0.25, 
                                                random_state=123)
    
    return train, validate, test

def zillow_standard_scaler(train, validate, test):
    """
        Accepts Zillow train, validate, and test data, then returns the scaled versions of each.
    """

    # Set Columns
    cols = ['beds', 'baths', 'area', 'worth', 'tax']
    cols_scaled = ['beds_scaled', 'baths_scaled', 'area_scaled', 'worth_scaled', 'tax_scaled']

    # Create new dataframes
    train_standard = train[cols].copy()
    validate_standard = validate[cols].copy()
    test_standard = test[cols].copy()

    # Build, Fit Standard Scaler
    standardscaler = StandardScaler()
    standardscaler.fit(train[cols])

    # Transform Scaler
    scaled_train = standardscaler.transform(train[cols])
    scaled_validate = standardscaler.transform(validate[cols])
    scaled_test = standardscaler.transform(test[cols])

    # Assign Scaled Values to DataFrames
    train_standard[cols_scaled] = scaled_train
    validate_standard[cols_scaled] = scaled_validate
    test_standard[cols_scaled] = scaled_test

    # Return Scaled Splits
    return train_standard, validate_standard, test_standard

def plot_scaled_unscaled(df):
    """
        Pulls in unscaled-scaled dataframe and prints visualizations for each comparison.
        WARNING: uses df.columns.sort_values() and reads first two, then next two, and on.
        Dataframe must have unscaled 'colname' and scaled 'colname_scaled' for each comparison,
        or else function will break.
    """

    new_col_order = list(df.columns.sort_values())
    df = df[new_col_order]
    i = 0
    
    for col in new_col_order:
        plt.figure(figsize=(13, 6))

        # subplot 1
        plt.subplot(121)
        df[new_col_order[i]].plot.hist(title=new_col_order[i])

        #subplot 2
        plt.subplot(122)
        df[new_col_order[i+1]].plot.hist(title=new_col_order[i+1])
        
        i += 2
        if i > (len(new_col_order) - 1):
            break
    
def full_zillow_wrangle():
    """
        Run the entire gamut; acquire, prepare, encode, split, and scale Zillow.
    """
    # Acquire and prepare Zillow data from Codeup database
    df = wrangle_zillow()
    # Create dummy variables for 'locality' column
    df = dummy_zillow(df)
    # Split data
    train, validate, test = train_val_test(df)
    # Isolate target variable 'worth'
    X_train_exp = train.drop(columns='worth')
    X_train, y_train = train.drop(columns='worth'), train.worth
    X_validate, y_validate = validate.drop(columns='worth'), validate.worth
    X_test, y_test = test.drop(columns='worth'), test.worth
    # Scale data
    X_train, X_validate, X_test = zillow_standard_scaler(train, validate, test)

    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test