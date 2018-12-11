---
layout: post
title: "prediction of customer revenue"
author: Levani Zandarashvili

---

### Predict Future Sales
This notebook is based on the data provided in the Kaggle competition "Google Analytics Customer Revenue Prediction".
https://www.kaggle.com/c/ga-customer-revenue-prediction.

The tast was done as part of Penn Data Science Group Kaggle team.
http://penndsg.com/.

In this notebook, I trained the model in two steps. In the first step, I used LightGBM classified to predict probabilities of making a purchase for each row. In the second step, I "filtered out" the rows with very low probability of purchase and trained the LightGBM regression model on the remaining data. Since majority of purchases in GStore are made by the small group of customers, such two-step process will allow us to filter out the non-purchasing customers and let us focus on the purchasing customers during the regression model training.

This notebook goes through the following steps to make a final prediction of a future revenue for each customer based on the provided data.
* Load data from csv files
* Convert json format features into individual columns
* Exclude unnecessary features
* Visualize data
* Create new features
* Prepare data for model training
* Perform classification using LGBMClassifier combined with grid search
* Filter out the low probability rows
* Perform regression using LGBMRegression combined with grid search
* Make prediction for the test data
* Save test data prediction into csv file
* Discussion: Future steps to improve this model

### Import some crucial libraries


```python
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import datetime
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import json
import os
from pandas.io.json import json_normalize
import datetime
```

### Loading the test and training data.
Some of the features are present in json format. We need to expand them in order to use in calculation. The following function does exactly that.


```python
def load_df(csv_path='train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

# expand the test and train data as pandas dataframes
# this step takes ~5 minutes on my laptop, so instead I created a separate file where I ran these steps once and 
# saved the outputs into csv files, so that I don't have to repeate this step every single time.
# uncomment the lines below when running for the first time

#%%time
#train_df = load_df()
#test_df = load_df("test.csv")

#train_df.to_csv('train_expanded.csv')
#test_df.to_csv('test_expanded.csv')
```


```python
# load data with expanded json features from files created using the code above
train_df = pd.read_csv('train_v2_expanded.csv', low_memory=False, dtype={'fullVisitorId': 'str'})
test_df = pd.read_csv('test_v2_expanded.csv', low_memory=False, dtype={'fullVisitorId': 'str'})
```

### Check columns with a single unique value. If there is at least one value and a nan value, then we'll keep such features. Absence of a value (nan value) is itself informative.


```python
constant_columns = [c for c in train_df.columns if train_df[c].nunique(dropna = False) == 1]
constant_columns
```




    ['socialEngagementType',
     'device.browserSize',
     'device.browserVersion',
     'device.flashVersion',
     'device.language',
     'device.mobileDeviceBranding',
     'device.mobileDeviceInfo',
     'device.mobileDeviceMarketingName',
     'device.mobileDeviceModel',
     'device.mobileInputSelector',
     'device.operatingSystemVersion',
     'device.screenColors',
     'device.screenResolution',
     'geoNetwork.cityId',
     'geoNetwork.latitude',
     'geoNetwork.longitude',
     'geoNetwork.networkLocation',
     'totals.visits',
     'trafficSource.adwordsClickInfo.criteriaParameters']



### Exclude some useless/extra features.


```python
# features with constant values should be excluded, since they don't carry any useful information.
columns_to_exclude = constant_columns

# in addition, I will also exclude 'sessionId' feature, since it carries no useful information and 'Unnamed: 0',
# since it is an artifact I created when I expanded json features.
# columns_to_exclude = columns_to_exclude + ['sessionId']
columns_to_exclude = columns_to_exclude

# I'll remove these features from both test and training data.
train_df = train_df.drop(columns_to_exclude, axis = 1)
test_df = test_df.drop(columns_to_exclude, axis = 1)
```

### At this point test and training data should have identical set of features, with a single exception of the training feature (y). let's see it is true.


```python
set(train_df.columns) - set(test_df.columns)
```




    {'trafficSource.campaignCode'}



### As we can see, the training data has two additional features. 'totals.transactionRevenue' is the training feature, so it should stay, but the 'trafficSource.campaignCode' is unique for only the training data, so it has to be removed.


```python
train_df = train_df.drop(['trafficSource.campaignCode'], axis = 1)
print('done')
```

    done
    

### At this point,let's look at the features in more details.


```python
test_df.shape
```




    (401589, 39)




```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelGrouping</th>
      <th>customDimensions</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device.browser</th>
      <th>device.deviceCategory</th>
      <th>device.isMobile</th>
      <th>device.operatingSystem</th>
      <th>geoNetwork.city</th>
      <th>geoNetwork.continent</th>
      <th>geoNetwork.country</th>
      <th>geoNetwork.metro</th>
      <th>geoNetwork.networkDomain</th>
      <th>geoNetwork.region</th>
      <th>geoNetwork.subContinent</th>
      <th>totals.bounces</th>
      <th>totals.hits</th>
      <th>totals.newVisits</th>
      <th>totals.pageviews</th>
      <th>totals.sessionQualityDim</th>
      <th>totals.timeOnSite</th>
      <th>totals.totalTransactionRevenue</th>
      <th>totals.transactionRevenue</th>
      <th>totals.transactions</th>
      <th>trafficSource.adContent</th>
      <th>trafficSource.adwordsClickInfo.adNetworkType</th>
      <th>trafficSource.adwordsClickInfo.gclId</th>
      <th>trafficSource.adwordsClickInfo.isVideoAd</th>
      <th>trafficSource.adwordsClickInfo.page</th>
      <th>trafficSource.adwordsClickInfo.slot</th>
      <th>trafficSource.campaign</th>
      <th>trafficSource.isTrueDirect</th>
      <th>trafficSource.keyword</th>
      <th>trafficSource.medium</th>
      <th>trafficSource.referralPath</th>
      <th>trafficSource.source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>3162355547410993243</td>
      <td>1508198450</td>
      <td>1</td>
      <td>1508198450</td>
      <td>Firefox</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Europe</td>
      <td>Germany</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>Western Europe</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>water bottle</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Referral</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>8934116514970143966</td>
      <td>1508176307</td>
      <td>6</td>
      <td>1508176307</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Chrome OS</td>
      <td>Cupertino</td>
      <td>Americas</td>
      <td>United States</td>
      <td>San Francisco-Oakland-San Jose CA</td>
      <td>(not set)</td>
      <td>California</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>referral</td>
      <td>/a/google.com/transportation/mtv-services/bike...</td>
      <td>sites.google.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>7992466427990357681</td>
      <td>1508201613</td>
      <td>1</td>
      <td>1508201613</td>
      <td>Chrome</td>
      <td>mobile</td>
      <td>True</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>windjammercable.net</td>
      <td>not available in demo dataset</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>True</td>
      <td>NaN</td>
      <td>(none)</td>
      <td>NaN</td>
      <td>(direct)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>9075655783635761930</td>
      <td>1508169851</td>
      <td>1</td>
      <td>1508169851</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>Western Asia</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'Central America'}]</td>
      <td>20171016</td>
      <td>6960673291025684308</td>
      <td>1508190552</td>
      <td>1</td>
      <td>1508190552</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>Mexico</td>
      <td>not available in demo dataset</td>
      <td>prod-infinitum.com.mx</td>
      <td>not available in demo dataset</td>
      <td>Central America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical features.
Let's look at some categorical features in more details. 

At first, let's go through the categorical features one by one and use the groupby function to see the total revenues, mean revenues and counts of non-zero transaction.


```python
# Define a function to group by a certain feature, aggregate total revenue sum, mean and count and show the head of
# resulting dataframe (sorted)
def describe_function(col, df = train_df):
    display(df.groupby(col)['totals.transactionRevenue'].agg(['sum', 'mean','count']).sort_values(by="count", ascending=False).head())
    
features = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent"]

for col in features:
    describe_function(col, df = train_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>channelGrouping</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Referral</th>
      <td>9.960210e+11</td>
      <td>1.160052e+08</td>
      <td>8586</td>
    </tr>
    <tr>
      <th>Organic Search</th>
      <td>5.394990e+11</td>
      <td>9.785941e+07</td>
      <td>5513</td>
    </tr>
    <tr>
      <th>Direct</th>
      <td>5.973243e+11</td>
      <td>1.799170e+08</td>
      <td>3320</td>
    </tr>
    <tr>
      <th>Paid Search</th>
      <td>6.696862e+10</td>
      <td>9.418934e+07</td>
      <td>711</td>
    </tr>
    <tr>
      <th>Display</th>
      <td>1.082893e+11</td>
      <td>5.282403e+08</td>
      <td>205</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>device.browser</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chrome</th>
      <td>2.068165e+12</td>
      <td>1.238645e+08</td>
      <td>16697</td>
    </tr>
    <tr>
      <th>Safari</th>
      <td>7.670099e+10</td>
      <td>6.338925e+07</td>
      <td>1210</td>
    </tr>
    <tr>
      <th>Firefox</th>
      <td>1.502462e+11</td>
      <td>4.432043e+08</td>
      <td>339</td>
    </tr>
    <tr>
      <th>Internet Explorer</th>
      <td>1.195796e+10</td>
      <td>8.079703e+07</td>
      <td>148</td>
    </tr>
    <tr>
      <th>Edge</th>
      <td>8.067300e+09</td>
      <td>1.090176e+08</td>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>device.deviceCategory</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>desktop</th>
      <td>2.229097e+12</td>
      <td>1.318602e+08</td>
      <td>16905</td>
    </tr>
    <tr>
      <th>mobile</th>
      <td>7.429866e+10</td>
      <td>5.447116e+07</td>
      <td>1364</td>
    </tr>
    <tr>
      <th>tablet</th>
      <td>1.294908e+10</td>
      <td>5.285339e+07</td>
      <td>245</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>device.operatingSystem</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Macintosh</th>
      <td>1.315231e+12</td>
      <td>1.270264e+08</td>
      <td>10354</td>
    </tr>
    <tr>
      <th>Windows</th>
      <td>5.750670e+11</td>
      <td>1.610381e+08</td>
      <td>3571</td>
    </tr>
    <tr>
      <th>Chrome OS</th>
      <td>2.696522e+11</td>
      <td>1.583395e+08</td>
      <td>1703</td>
    </tr>
    <tr>
      <th>Linux</th>
      <td>7.025809e+10</td>
      <td>5.404468e+07</td>
      <td>1300</td>
    </tr>
    <tr>
      <th>iOS</th>
      <td>3.588012e+10</td>
      <td>4.312514e+07</td>
      <td>832</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.city</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not available in demo dataset</th>
      <td>9.397274e+11</td>
      <td>1.320213e+08</td>
      <td>7118</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>3.258931e+11</td>
      <td>1.357322e+08</td>
      <td>2401</td>
    </tr>
    <tr>
      <th>Mountain View</th>
      <td>2.035964e+11</td>
      <td>9.854617e+07</td>
      <td>2066</td>
    </tr>
    <tr>
      <th>San Francisco</th>
      <td>1.429065e+11</td>
      <td>1.255769e+08</td>
      <td>1138</td>
    </tr>
    <tr>
      <th>Sunnyvale</th>
      <td>6.908219e+10</td>
      <td>8.146485e+07</td>
      <td>848</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Americas</th>
      <td>2.265098e+12</td>
      <td>1.247988e+08</td>
      <td>18150</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>2.339772e+10</td>
      <td>1.231459e+08</td>
      <td>190</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>1.124745e+10</td>
      <td>8.926548e+07</td>
      <td>126</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>6.781100e+09</td>
      <td>2.338310e+08</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Africa</th>
      <td>9.025990e+09</td>
      <td>7.521658e+08</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>2.197885e+12</td>
      <td>1.244697e+08</td>
      <td>17658</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>4.269923e+10</td>
      <td>1.355531e+08</td>
      <td>315</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>1.390126e+10</td>
      <td>2.044303e+08</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Taiwan</th>
      <td>2.934200e+09</td>
      <td>9.169375e+07</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>3.763080e+09</td>
      <td>1.393733e+08</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.metro</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not available in demo dataset</th>
      <td>9.397274e+11</td>
      <td>1.320213e+08</td>
      <td>7118</td>
    </tr>
    <tr>
      <th>San Francisco-Oakland-San Jose CA</th>
      <td>5.062230e+11</td>
      <td>1.015696e+08</td>
      <td>4984</td>
    </tr>
    <tr>
      <th>New York NY</th>
      <td>3.274385e+11</td>
      <td>1.351376e+08</td>
      <td>2423</td>
    </tr>
    <tr>
      <th>Chicago IL</th>
      <td>1.104102e+11</td>
      <td>1.665313e+08</td>
      <td>663</td>
    </tr>
    <tr>
      <th>Los Angeles CA</th>
      <td>9.190494e+10</td>
      <td>1.598347e+08</td>
      <td>575</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.networkDomain</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(not set)</th>
      <td>1.305843e+12</td>
      <td>1.201659e+08</td>
      <td>10867</td>
    </tr>
    <tr>
      <th>comcast.net</th>
      <td>1.773147e+11</td>
      <td>1.178171e+08</td>
      <td>1505</td>
    </tr>
    <tr>
      <th>verizon.net</th>
      <td>7.967340e+10</td>
      <td>1.055277e+08</td>
      <td>755</td>
    </tr>
    <tr>
      <th>unknown.unknown</th>
      <td>5.825398e+10</td>
      <td>8.759997e+07</td>
      <td>665</td>
    </tr>
    <tr>
      <th>rr.com</th>
      <td>5.322676e+10</td>
      <td>9.052170e+07</td>
      <td>588</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not available in demo dataset</th>
      <td>9.397274e+11</td>
      <td>1.320213e+08</td>
      <td>7118</td>
    </tr>
    <tr>
      <th>California</th>
      <td>6.091657e+11</td>
      <td>1.079124e+08</td>
      <td>5645</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>3.262089e+11</td>
      <td>1.357507e+08</td>
      <td>2403</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>1.104102e+11</td>
      <td>1.665313e+08</td>
      <td>663</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>4.921693e+10</td>
      <td>9.612682e+07</td>
      <td>512</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>geoNetwork.subContinent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Northern America</th>
      <td>2.240584e+12</td>
      <td>1.246639e+08</td>
      <td>17973</td>
    </tr>
    <tr>
      <th>South America</th>
      <td>1.855388e+10</td>
      <td>1.520810e+08</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Eastern Asia</th>
      <td>1.399618e+10</td>
      <td>1.686287e+08</td>
      <td>83</td>
    </tr>
    <tr>
      <th>Southeast Asia</th>
      <td>6.019380e+09</td>
      <td>1.020234e+08</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Western Europe</th>
      <td>4.259950e+09</td>
      <td>8.874896e+07</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>trafficSource.adContent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Google Merchandise Collection</th>
      <td>2.064188e+10</td>
      <td>1.358018e+08</td>
      <td>152</td>
    </tr>
    <tr>
      <th>Official Google Merchandise</th>
      <td>1.719470e+09</td>
      <td>6.877880e+07</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Display Ad created 3/11/14</th>
      <td>5.697200e+08</td>
      <td>6.330222e+07</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Google Online Store</th>
      <td>2.109800e+08</td>
      <td>3.516333e+07</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Placement Accessores 300 x 250</th>
      <td>5.771900e+08</td>
      <td>1.154380e+08</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


As we can see, using tables to visualize our data is very tricky, so next, I will draw the graphs instead.
For each of the following features, I calculated the sum and count of transactionRevenue and sorted in descending order (based on count) before plotting.


```python
#temp_df = train_df.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['sum', 'count']).sort_values(by="count", ascending=False).head(10)
plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(len(features), 2, figsize=(30,85))
# sns.barplot(y=temp_df.index, x=temp_df['sum'], ax = axes[0,0])
# sns.barplot(y=temp_df.index, x=temp_df['count'], ax = axes[0,1])

for n in range(len(features)):
    col = features[n]
    temp_df = train_df.groupby(col)['totals.transactionRevenue'].agg(['sum', 'count']).sort_values(by="count", ascending=False).head(10)
    # Some names were too long, so I shortened them to just 10 characters
    y_temp = list(map(lambda x: x[:10], list(temp_df.index)))
    sns.barplot(y=y_temp, x=temp_df['sum'], ax = axes[n,0])
    sns.barplot(y=y_temp, x=temp_df['count'], ax = axes[n,1])
    axes[n,0].set_title(col + ': sum')
    axes[n,1].set_title(col + ': count')
    axes[n,0].set_xlabel('')
    axes[n,1].set_xlabel('')
    #axes[n,0].tick_params(labelsize=20)
    #axes[n,1].tick_params(labelsize=20)
```


![png](images/blog/prediction_of_customer_revenue_21_0.png)


As you can see, while there is some strong imbalance for some features (e.g. continent, country, etc.), there is a quite good variation in some other features (e.g. region, metro, city, etc.). Such variation is very useful when training models.

### "visitStartTime" column contains the same information as "date", so it can be replaced. I'll also generate features for the day of the week, hour, month and day of the month. I will also generate features for the difference between next and previous visit sessions for both test and train datasets.


```python
train_df['date'] = pd.to_datetime(train_df['visitStartTime'], unit='s')
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['hour'] = train_df['date'].dt.hour
train_df['day_of_month'] = train_df['date'].dt.day
train_df['month'] = train_df['date'].dt.month

test_df['date'] = pd.to_datetime(test_df['visitStartTime'], unit='s')
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['hour'] = test_df['date'].dt.hour
test_df['day_of_month'] = test_df['date'].dt.day
test_df['month'] = test_df['date'].dt.month

train_df['next_session_1'] = (
        train_df['date'] - train_df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60

train_df['next_session_2'] = (
        train_df['date'] - train_df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60

test_df['next_session_1'] = (
        test_df['date'] - test_df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60

test_df['next_session_2'] = (
        test_df['date'] - test_df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60

# delete the "date" feature
train_df.drop('date', axis = 1, inplace = True)
test_df.drop('date', axis = 1, inplace = True)

print('done')
```

    done
    

### Impute 0 for missing target values. No need to do this for other features since LightGBM can deal with them on its own.



```python
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
# test_id will be used in the end when generating the submission file.
test_id = test_df["fullVisitorId"].values
```

### label encode the categorical variables and convert the numerical variables to float



```python
# define the categorical features
cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect',
            'day_of_week', 'hour', 'day_of_month', 'month']

# label encode the categorical features
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

# define categorical values as 'categorical'. Might be a bit redundant, but I want to make sure that they are defined as such.
for c in cat_cols:
    train_df[c] = train_df[c].astype('category')
    test_df[c] = test_df[c].astype('category')
        

# make sure that the numerical columns are float type
num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits', 'next_session_1', 'next_session_2']    
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

print('done')
```

    channelGrouping
    device.browser
    device.deviceCategory
    device.operatingSystem
    geoNetwork.city
    geoNetwork.continent
    geoNetwork.country
    geoNetwork.metro
    geoNetwork.networkDomain
    geoNetwork.region
    geoNetwork.subContinent
    trafficSource.adContent
    trafficSource.adwordsClickInfo.adNetworkType
    trafficSource.adwordsClickInfo.gclId
    trafficSource.adwordsClickInfo.page
    trafficSource.adwordsClickInfo.slot
    trafficSource.campaign
    trafficSource.keyword
    trafficSource.medium
    trafficSource.referralPath
    trafficSource.source
    trafficSource.adwordsClickInfo.isVideoAd
    trafficSource.isTrueDirect
    day_of_week
    hour
    day_of_month
    month
    done
    

### I am going to first perform a classification task where I will classify the rows based on the probability of a purchase, regardless of an exact value of the purchase.
Majority of visits don't result in a purchase and performing such classification first might help us remove the instances of visits with very low probability of visit. For this purpose, I will identify the visitors based on their 'fullVisitorId' and see if they have ever made even a single purchase. Then I will add a new identifier 'purchasing_customer' to each row. 0 if this customer has never made a purchase and 1 if they have. Then I will use this feature for the classification task. Next, I will have to use the results of the classification to predict the feature for the test set (since we don't know who makes a purchase in the test set).


```python
# calculate the total revenue per visitor. I'll use this later to see if a visitor has ever made a purchase
total_purchase_df = train_df[['fullVisitorId','totals.transactionRevenue']].groupby('fullVisitorId').sum()
total_purchase_df = total_purchase_df['totals.transactionRevenue'].to_dict()
```


```python
# create a feature purchasing_customer.
# 1 indicates that the visitor has made a purchase at least once. 0 indicates no purchase ever.
train_df['purchasing_customer'] = train_df["fullVisitorId"].apply(lambda x: 1 if total_purchase_df[x] > 0 else 0)
print("there are ", train_df['purchasing_customer'].sum(), "entres for customers who made a purchase at some point")
```

    there are  61319 entres for customers who made a purchase at some point
    

### Define the parameters for the grid search for classification task. 
Parameters for grid search and the fixed parameters need to be in separate dictionaries. For the parameters for which I already performed the grid search, I will put the tested options next to them as comments so that I don't need to test them again.


```python
# Set fixed params
params_classification = {#"objective" : "regression",
        "metric" : "rmse",
        'max_depth' : -1,
        'max_bin': 512,
        "min_child_samples" : 100,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1}

# Create parameters to search
gridParams_classification = {
    'learning_rate': [0.1], #0.1, 0.05, 0.2
    'n_estimators': [100], # 70, 50
    'num_leaves': [70], # 30, 50
    'colsample_bytree': [0.3], # 0.1, 1
    'subsample': [0.3], # 0.1, 1
    'reg_alpha': [1.4], # 1, 1.2
    'reg_lambda': [1], # 1.2, 1.4
    }
```

### create training dataframes. Of course, the 'totals.transactionRevenue' will be excluded, since it is not present in the test set.


```python
X_train_df = train_df[cat_cols + num_cols]
y_train_df = train_df['purchasing_customer']

# X_test_df will be used to predict the 'purchasing_customer' feature for the test set.
X_test_df = test_df[cat_cols + num_cols]
```

### Create the classifier and perform classification task.


```python
# Create classifier to use. Note that parameters have to be input manually, not as a dict!

mdl_classifier = lgb.LGBMClassifier(
          objective = 'binary',
          n_jobs = 3,
          silent = True,
          max_depth = params_classification['max_depth'],
          max_bin = params_classification['max_bin'],
          feature_fraction = params_classification['feature_fraction'],
          min_child_samples = params_classification['min_child_samples'],
          bagging_fraction = params_classification['bagging_fraction']
          )
```


```python
# Create the grid search with 4 fold cross validation
grid_classification = GridSearchCV(mdl_classifier, gridParams_classification,
                    verbose=0,
                    cv=5,
                    n_jobs=2)
```


```python
# Run the grid
grid_classification.fit(X_train_df, y_train_df)

# Print the best parameters found
print(grid_classification.best_params_)
print(grid_classification.best_score_)
```

    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 100, 'num_leaves': 70, 'reg_alpha': 1.4, 'reg_lambda': 1, 'subsample': 0.3}
    0.9701522591853949
    


```python
# check the quality of calculation
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_train_df, grid_classification.predict_proba(X_train_df)[:,1]))
print('rms = ', rms)

from sklearn.metrics import accuracy_score
accur = accuracy_score(y_train_df, grid_classification.predict(X_train_df))
print('accuracy = ', accur)

```

    rms =  0.14208548895989545
    accuracy =  0.9724708883551665
    

    C:\Users\Leo\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### Predict probability of 1 (of making a purchase). probability of 0 would be (1 - proba[1])


```python
train_df['proba_prediction_train'] = grid_classification.predict_proba(X_train_df)[:,1]
test_df['proba_prediction_train'] = grid_classification.predict_proba(X_test_df)[:,1]
```

### Now I will perform the regression task to predict the revenue first for each visit, and then sum up these predictions to predict the revenue per customer.
I will use the results of the classification task in two ways. First, I will use the new feature of 'proba_prediction_train' which predicts the probability of purchase for each row. Second, I will used this feature additionally to filter out the rows with very low probability of purchase. I do not know if such action will be useful, or what is the optimal cut off level for such filter, thus I will test several cut off levels and see if such treatment improves the score.

Similar to the classification task, I will employ the grid search for the regression task as well.


```python
# Set params
params_regression = {"objective" : "regression",
        "metric" : "rmse",
        'max_depth' : -1,
        'max_bin': 512,
        "min_child_samples" : 100,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1}

# Create parameters to search
gridParams_regression = {
    'learning_rate': [0.10], # 0.15, 0.10
    'n_estimators': [70],
    'num_leaves': [50],
    'colsample_bytree': [0.3],
    'subsample': [0.3],
    'reg_alpha': [1], #[1,1.2, 1.4],
    'reg_lambda': [1], #[1,1.2,1.4]
    }
```


```python
# Create classifier to use. Note that parameters have to be input manually
# not as a dict!

mdl_regression = lgb.LGBMRegressor(
          objective = 'regression',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          max_depth = params_regression['max_depth'],
          max_bin = params_regression['max_bin'],
          feature_fraction = params_regression['feature_fraction'],
          min_child_samples = params_regression['min_child_samples'],
          bagging_fraction = params_regression['bagging_fraction']
          )
```


```python
# cuot off levels for the probability of making a purchase
# 0.0 will indicate that no data will be cut off
probability_cutoff_list = [0.0, 0.01, 0.05, 0.1, 0.25]

grid_regression = []
for probability_cutoff in probability_cutoff_list:
    X_train_df = train_df[train_df['proba_prediction_train']>probability_cutoff][cat_cols + num_cols + ['proba_prediction_train']]
    y_train_df = np.log1p(train_df[train_df['proba_prediction_train']>probability_cutoff]["totals.transactionRevenue"].values)

    X_test_df = test_df[test_df['proba_prediction_train']>probability_cutoff][cat_cols + num_cols + ['proba_prediction_train']]
    
    # Create the grid search with 4 fold cross validation
    grid = GridSearchCV(mdl_regression, gridParams_regression,
                        verbose=0,
                        cv=5,
                        n_jobs=2)
    # Run the grid
    grid.fit(X_train_df, y_train_df)
    
    # add results of grid search to the list of grid_regression
    grid_regression = grid_regression + [grid]

    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)
```

    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 70, 'num_leaves': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.3}
    0.366094783813078
    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 70, 'num_leaves': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.3}
    0.3433724709301465
    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 70, 'num_leaves': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.3}
    0.335454834793921
    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 70, 'num_leaves': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.3}
    0.33146897589710855
    {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'n_estimators': 70, 'num_leaves': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.3}
    0.3310650060859367
    

### The best score is achieved when no data it cut off. In other words, removing the rows with low predicted probability of making a purchase does not improve out predictions. Still, it was an interesting option to test.


```python
# predict revenue for the test data used in the regression test. 
# Since the 0.0 cut off level gives the best results, the whole test set will be used.
X_test_df['prediction'] = grid_regression[0].predict(X_test_df)
```


```python
# following lines are technically not necessary since I ended up using the 0.0 cut off level, but I'll still keep them.
test_df['prediction'] = 0
test_df.loc[X_test_df.index.values,'prediction'] = X_test_df[['prediction']]

# replace all the negative predictions with 0, as there can be no negative value
test_df['prediction'] = test_df['prediction'].apply(lambda x: x if x > 0 else 0)
```

### Create a submission file


```python
# Load submission template file
submission = pd.read_csv('sample_submission_v2.csv')
submission.head()
```

    C:\Users\Leo\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>PredictedLogRevenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000018966949534117</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000039738481224681</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000073585230191399</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000087588448856385</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0000149787903119437</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sub_df = pd.DataFrame({"fullVisitorId":test_id})
sub_df["PredictedLogRevenue"] = np.expm1(test_df['prediction'].values)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d")
sub_df.to_csv("predict_revenues_" + now + ".csv", index=False)
print('Finished')
```

    Finished
    

### Discussion
In this notebook, I successfully used the LGB classification and regression models to predict the future revenue for the users. There are number of additional methods we could use to improve prediction by our model.
1. Blending. Often model can be improved by blending the predictions by different approaches (SVM, linear regression, RNN, etc).
2. In current approach, I did not take into consideration the chronology of the purchases. The model might perform better if we treat the data as a time series.
3. During the classification step I predicted the probability of a revenue (yes or no without an actual value) for each row. Since in the final step of the process we are predicting the revenue for each user, not each row, we might get a better result if we predict the probabilities for users instead of the rows during the classification step.
