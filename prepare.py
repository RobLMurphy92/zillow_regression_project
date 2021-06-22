import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

#train validate, split
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# metrics and confusion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#model classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ignore warnings

# ignore warnings
import warnings
warnings.filterwarnings("ignore")




    


###################### Prep Telco Data ######################

def prep_zillow(df):
    '''
    This function take in the telco_churn data acquired by get_telco_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    df.set_index('parcelid', drop = True, inplace = True)

    df = df[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet','taxvaluedollarcnt']]
    df.rename(columns = {'calculatedfinishedsquarefeet':'total_squareft','taxvaluedollarcnt': 'assessment_value'},  inplace = True)

    #dropping duplicate values
    df.drop_duplicates(inplace = True)
    # so after reviewing the two columns taxvalue and sqrft, i feel comfortable dropping the NaN values
    df.dropna(inplace = True)

    #dropping outliers
    df= df[(df['assessment_value'] < 1276040)& (df['assessment_value']>0)]
    df= df[(df['total_squareft'] < 3882) & (df['total_squareft']>0)]
    df = df[(df['bedroomcnt'] < 5.5) &  (df['bedroomcnt']>1.5)]
    df= df[(df['bathroomcnt'] < 4.5) & (df['bathroomcnt']>0.5)]


    #created additional columns to represent less than and greater than with 0 and 1 value
    df['three_or_less_bedrooms']  =  pd.cut(x=df['bedroomcnt'], bins=[2, 3], right = True, include_lowest = True, labels = [1]).astype('object').fillna(0).astype('int')
    df['four_or_more_bedrooms']  =  pd.cut(x=df['bedroomcnt'], bins=[4, 5], right = True, include_lowest = True, labels = [1]).astype('object').fillna(0).astype('int')
    df['three_or_more_bathrooms']  =  pd.cut(x=df['bedroomcnt'], bins=[3, 4], right = True, include_lowest = True, labels = [1]).astype('object').fillna(0).astype('int')
    df['two_half_or_less_bathrooms']  =  pd.cut(x=df['bedroomcnt'], bins=[1, 2.5], right = True, include_lowest = True, labels = [1]).astype('object').fillna(0).astype('int')
    df.drop(columns = ['bathroomcnt', 'bedroomcnt'], inplace = True)
    

    
    return df


#function to look create df which shows records containing nulls
def view_null_records(df, variable):
    """
    function allows you to records which contain null, nan values.
    REMEMBER, will only work for individual column and if that columns has nulls, 
    otherwise will return empty dataframe
    """
    df[df[variable].isna()]
    
    return df[df[variable].isna()]


    
  
    
##############################################################################
# extract objects or numerical based columns 


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols

# Generic splitting function for continuous target.
def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test





##############################################################################
def model_metrics(X_train, y_train, X_validate, y_validate):
    '''
    this function will score models and provide confusion matrix.
    returns classification report as well as evaluation metrics.
    '''
    lr_model = LogisticRegression(random_state =1349)
    dt_model = DecisionTreeClassifier(max_depth = 2, random_state=1349)
    rf_model = RandomForestClassifier(max_depth=4, min_samples_leaf=3, random_state=1349)
    kn_model = KNeighborsClassifier()
    models = [lr_model, dt_model, rf_model]
    for model in models:
        #fitting our model
        model.fit(X_train, y_train)
        #specifying target and features
        train_target = y_train
        #creating prediction for train and validate
        train_prediction = model.predict(X_train)
        val_target = y_validate
        val_prediction = model.predict(X_validate)
        # evaluation metrics
        TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_train, train_prediction).ravel()
        TN_v, FP_v, FN_v, TP_v = confusion_matrix(y_validate, val_prediction).ravel()
        #calculating true positive rate, false positive rate, true negative rate, false negative rate.
        tpr_t = TP_t/(TP_t+FN_t)
        fpr_t = FP_t/(FP_t+TN_t)
        tnr_t = TN_t/(TN_t+FP_t)
        fnr_t = FN_t/(FN_t+TP_t)
        tpr_v = TP_v/(TP_v+FN_v)
        fpr_v = FP_v/(FP_v+TN_v)
        tnr_v = TN_v/(TN_v+FP_v)
        fnr_v = FN_v/(FN_v+TP_v)
        
        
        
        print('--------------------------')
        print('')
        print(model)
        print('train set')
        print('')
        print(f'train accuracy: {model.score(X_train, y_train):.2%}')
        print('classification report:')
        print(classification_report(train_target, train_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_t:.2%},  
        False Positive Rate :{fpr_t:.2%},
        True Negative Rate: {tnr_t:.2%},  
        False Negative Rate: {fnr_t:.2%}''')
        print('------------------------')
        
        print('validate set')
        print('')
        print(f'validate accuracy: {model.score(X_validate, y_validate):.2%}')
        print('classification report:')
        print(classification_report(y_validate, val_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_v:.2%},  
        False Positive Rate :{fpr_v:.2%},
        True Negative Rate: {tnr_v:.2%},  
        False Negative Rate: {fnr_v:.2%}''')
        print('')
        print('------------------------')
        
        
        ####################################################################
# Train, validate, split which doesnt exclude target from train. 
# target is categorical 


#genreal split when categorical
def general_split(df, stratify_var):
    '''
    This function take in the telco_churn_data acquired by get_telco_churn,
    performs a split and stratifies total_charges column. Can specify stratify as None which will make this useful for continous.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, stratify = stratify_var)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, stratify = stratify_var)
    """
    returns train, validate, test 
    """
    return train, validate, test






##################################################
# train, validate, split #
#  generates features and target.
################################################

def train_validate_test(df, target, stratify):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify = stratify)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123, stratify = stratify)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    '''    
    Returns X_train, y_train, X_validate, y_validate, X_test, y_test
    '''

    return X_train, y_train, X_validate, y_validate, X_test, y_test

################################################################




########################################
#                scaling

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )
    
    """
    returns X_train_scaled, X_validate_scaled, X_test_scaled 
    """

    return X_train_scaled, X_validate_scaled, X_test_scaled
        
    
 


    
    
    