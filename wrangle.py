import pandas as pd
import os
from env import host, password, username


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
        Imports zillow data, drops nulls and duplicates, returns df
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

    # Pass back dataframe
    return zillow