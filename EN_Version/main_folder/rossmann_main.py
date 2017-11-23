# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:37:09 2017

@author: Mengfei Li
"""

# =============================================================================
# import packages
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# read data
# =============================================================================
df_train = pd.read_csv("../input/train.csv", low_memory=False)
df_test = pd.read_csv("../input/test.csv", low_memory=False)
df_store = pd.read_csv("../input/store.csv", low_memory=False)



# =============================================================================
# remove outlier
# =============================================================================
def remove_outliers(data):
    df_0 = data.loc[data.Sales ==0]   
    q1 = np.percentile(data.Sales, 25, axis=0)
    q3 = np.percentile(data.Sales, 75, axis=0)
#    k = 3
#    k = 2.5
#    k = 2.8
#    k = 2
    k = 1.5
    iqr = q3 - q1
    df_temp = data.loc[data.Sales > q1 - k*iqr]
    df_temp = data.loc[data.Sales < q3 + k*iqr]
    frames = [df_0, df_temp]
    result = pd.concat(frames)
    return result


# =============================================================================
# handle time data
# =============================================================================
time_format = '%Y-%m-%d'
def seperate_date(data):     
    # split date feature
    data_time = pd.to_datetime(data.Date, format=time_format)
    data['Year']= data_time.dt.year
    data['Month'] = data_time.dt.month
    data['DayOfYear'] = data_time.dt.dayofyear
    data['DayOfMonth'] = data_time.dt.day
    data['WeekOfYear'] = data_time.dt.week   
    return data

# =============================================================================
# add mean sales value
# =============================================================================
mean_store_sales = []

mean_store_sales_promo = []
mean_store_sales_not_promo = []

mean_store_sales_2013 = []
mean_store_sales_2014 = []
mean_store_sales_2015 = []

mean_store_sales_m1 = []
mean_store_sales_m2 = []
mean_store_sales_m3 = []
mean_store_sales_m4 = []
mean_store_sales_m5 = []
mean_store_sales_m6 = []
mean_store_sales_m7 = []
mean_store_sales_m8 = []
mean_store_sales_m9 = []
mean_store_sales_m10 = []
mean_store_sales_m11 = []
mean_store_sales_m12 = []

mean_store_sales_d1 = []
mean_store_sales_d2 = []
mean_store_sales_d3 = []
mean_store_sales_d4 = []
mean_store_sales_d5 = []
mean_store_sales_d6 = []
mean_store_sales_d7 = []

mean_store_sales_1month = []
mean_store_sales_2months = []
mean_store_sales_3months = []
mean_store_sales_6months = []

def add_mean_sales(data, data_store = df_store):
    # mean of sales
    stores = data.Store.unique()     
    
    for store in stores:
        serie = data[data.Store == store]
        
        # mean of sales by Promo or not
        mean_store_sales.append(np.mean(serie.Sales))
        mean_store_sales_promo.append(serie[serie['Promo'] == 1]['Sales'].mean())
        mean_store_sales_not_promo.append(serie[serie['Promo'] == 0]['Sales'].mean())
        
        # mean of salse by year
        mean_store_sales_2013.append(serie[serie['Year'] == 2013]['Sales'].mean())
        mean_store_sales_2014.append(serie[serie['Year'] == 2014]['Sales'].mean())
        mean_store_sales_2015.append(serie[serie['Year'] == 2015]['Sales'].mean())
                
        # mean of sales by last months
        mean_store_sales_1month.append(serie[(serie['Month'] == 7) & (serie['Year'] == 2015)]['Sales'].mean())                   
        mean_store_sales_2months.append(serie[(serie['Month'] <= 7) ^(serie['Month'] >= 6) & (serie['Year'] == 2015)]['Sales'].mean())
        mean_store_sales_3months.append(serie[(serie['Month'] <= 7) ^(serie['Month'] >= 5) & (serie['Year'] == 2015)]['Sales'].mean())        
        mean_store_sales_6months.append(serie[(serie['Month'] <= 7) ^(serie['Month'] >= 2) & (serie['Year'] == 2015)]['Sales'].mean())
             
    data_store['mean_sotre_sales_promo'] = mean_store_sales_promo
    data_store['mean_store_sales_not_promo'] = mean_store_sales_not_promo
    
    data_store['mean_store_sales_2013'] = mean_store_sales_2013
    data_store['mean_store_sales_2014'] = mean_store_sales_2014
    data_store['mean_store_sales_2015'] = mean_store_sales_2015
        
    data_store['mean_store_sales_1month'] = mean_store_sales_1month
    data_store['mean_store_sales_2months'] = mean_store_sales_2months
    data_store['mean_store_sales_3months'] = mean_store_sales_3months
    data_store['mean_store_sales_6months'] = mean_store_sales_6months
                          
    return data_store

# =============================================================================
# remove usless Store information
# =============================================================================
def drop_stores(data_test, data):

    stores = data_test.Store.unique()     
    
    for store in stores:
        serie = data[data.Store == store]
        data = serie
        
    return data



# =============================================================================
# more feature engineering
# =============================================================================	
def feature_eng_compl(data):
        
    # merge store dataset    
    data = data.join(df_store, on='Store', rsuffix='_')
    data = data.drop('Store_',axis=1)
       
    # handle the competition and promo2 feature, combination and drop  
    data['CompetitionLastMonths'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear'].apply(lambda x: x if x > 0 else np.nan) - 1 + data['CompetitionOpenSinceMonth'].apply(lambda x: x if x > 0 else np.nan))
    data['Promo2LastDays'] = 365 * (data['Year'] - data['Promo2SinceYear'].apply(lambda x: x if x > 0 else np.nan))/4.0 + (data['DayOfYear'] - 7*(data['Promo2SinceWeek'].apply(lambda x: x if x > 0 else np.nan)) - 1)
    data = data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1)

    # mapping
    data['Year'] = data['Year'].map({2013:1, 2014:2, 2015:3})
    data['StateHoliday'] = data['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3})
    data['StoreType'] = data['StoreType'].map({'0':0, 'a':1, 'b':2, 'c':3, 'd':4})
    data['Assortment'] = data['Assortment'].map({'0':0, 'a':1, 'b':2, 'c':3})  
    data['PromoInterval'] = data['PromoInterval'].map({'0':0,'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3})
    
    return data   


    
# =============================================================================
# begining feature engineering
# =============================================================================
    
# handle time data
print ('seperate date ...............')   
df_train = seperate_date(df_train)
df_test = seperate_date(df_test)

# add mean sales value
df_store = add_mean_sales(df_train, df_store)
print ('add mean sales ...............') 

# more feature engineering
print ('more feature engineering ...............')
df_train = feature_eng_compl(df_train).drop('Customers', axis=1)
df_test = feature_eng_compl(df_test)

# =============================================================================
# add 'DaysToHoliday' feature
# =============================================================================
holidaysofyear = df_train[(df_train['StateHoliday'] == 1)].DayOfYear.reset_index(name='DayOfHoliday').DayOfHoliday.unique()
holidaysofyear = sorted(holidaysofyear)
    
for holiday in holidaysofyear:
    df_train['DaysToHoliday' + str(holiday)] = holiday - df_train['DayOfYear']
for holiday in holidaysofyear:
    df_test['DaysToHoliday' + str(holiday)] = holiday - df_test['DayOfYear']
    
# drop useless store information
print ('drop useless store information ...............')  
df_store = drop_stores(df_test, df_store)
    
# remove outlier
print ('remove outliers ...............') 
df_train = remove_outliers(df_train)

# drop 'Date'
df_train = df_train.drop('Date', axis=1)
df_test = df_test.drop('Date', axis=1)


# =============================================================================
# Xgboost Kernel
# https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code
# =============================================================================

def create_feature_map(features):
    # feature importance plot
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def rmspe(y, yhat):
    # rmspe 
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    # rmspe in Xgboost
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


train = df_train
test = df_test

train = train[train["Open"] != 0]
train = train[train["Sales"] > 0]

print('training beging ................')

for iter in range(0,7):
    params = {'objective': 'reg:linear',
              'min_child_weight': 50, 
              'booster' : 'gbtree',
              'eta': 0.1,
              'alpha': 2,
              'gamma': 2,
              'max_depth': 12 - iter,
              'subsample': 0.9,
              'colsample_bytree': 0.9,
              'silent': 1,
              'seed': 1301,
              'tree_method': 'gpu_hist',
              'max_bin': 600
              }
    
    num_boost_round = 5000 # 5000
       
    
    print("Train a XGBoost model .....................")
    
    features = list(train.drop('Sales', axis=1)) 
    
    X_train, X_valid = train_test_split(train, test_size=0.01, random_state=1)
    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
      early_stopping_rounds=200, 
      feval=rmspe_xg, 
      verbose_eval=True)

    
    print("Validating")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))
    
    print("Make predictions on the test set")
    dtest = xgb.DMatrix(test[features])
    test_probs = gbm.predict(dtest)
    
    # Make Submission
    file_name = iter
    result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
    result.to_csv("XG_"+str(file_name)+'.csv', index=False)
    
    gbm.save_model(str(file_name)+'.model')
    
# =============================================================================
#  feature importance plot, based on：
#  https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code
# =============================================================================
create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)