import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Telco_Churn Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#####################################################################


def zillow_data():
    '''
    This function reads the zillow data filtered by single family residential from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = '''
    select transactiondate, p17.propertylandusetypeid,bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,unitcnt,
    taxvaluedollarcnt,yearbuilt,taxamount,fips
    from properties_2017 as p17
    join predictions_2017 as pr17 ON p17.parcelid = pr17.parcelid
    join propertylandusetype as plt ON p17.propertylandusetypeid = plt.propertylandusetypeid
    where (pr17.transactiondate BETWEEN '2017-05-01' AND '2017-08-31') and p17.propertylandusetypeid IN (31, 46, 47, 260,
    261,262,263,264,265,268,273,274,275,276, 279);
    '''
    
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df


def get_zillow(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco_churn df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_data.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_data.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    return df