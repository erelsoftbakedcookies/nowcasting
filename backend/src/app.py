#!/usr/bin/env python3.7
"""
Built for Hackathon to predict economic impact of COVID-19 to ADB country members

Usage:
    python3.7 -m notebook
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import pickle
import os
import glob
import datetime
import statsmodels.api as sm
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Clean up models folder
models_paths = list(glob.glob('/apps/models/*'))
for model in models_paths:
    os.remove(model)

# Load data
db_file = './COVID19_Challenge_Database_17Apr_2020.xlsx'
df = pd.read_excel(db_file, sheet_name=[0, 1, 2, 3], header=[0, 1])
detail = df[0]
a = df[1]
q = df[2]
m = df[3]

# Data cleaning a
a.drop(a.columns[[27]], axis=1, inplace=True)
headera = list(np.arange(2000, 2021))
a.columns = ["Member", "Indicator", "Unit",
             "Classification", "Subclassification", "Remarks"] + headera
nrow, ncol = a.shape
a_data_length = 21
a.loc[a['Unit'].isnull(), 'Unit'] = ''
a[~a[headera].applymap(np.isreal)] = np.nan
a[headera] = a[headera].astype(np.float)

# Data cleaning detail
detail.columns = ['countrynames', 'adbmember', 'cluster',
                  'nationalaccountsm', 'nationalaccountsq', 'nationalaccountsa',
                  'prodnretailsalesm', 'prodnretailsalesq', 'prodnretailsalesa',
                  'tradem', 'tradeq', 'tradea',
                  'remittancesm', 'remittancesq', 'remittancea',
                  'tourismarrivalsnreceiptsm', 'tourismarrivalsnreceiptsq',
                  'tourismarrivalsnreceiptsa',
                  'inflationm', 'inflationq', 'inflationa',
                  'fxm', 'fxq', 'fxa']
detail = detail[1:][:-6]


# Data cleaning q
q = q[1:]
headerq = []
for y in list(range(2000, 2021)):
    for quar in ["-Q1", "-Q2", "-Q3", "-Q4"]:
        headerq.append(str(y) + quar)
q.columns = ["Member", "Indicator", "Unit",
             "Classification", "Subclassification", "Remarks"] + headerq
q_data_length = 21 * 4
q.loc[q['Unit'].isnull(), 'Unit'] = ''
q[~q[headerq].applymap(np.isreal)] = np.nan
q[headerq] = q[headerq].astype(np.float)

# Data cleaning m
m = m[1:]
headerm = []
for y in list(range(2000, 2021)):
    for mon in ["-Jan", "-Feb", "-Mar", "-Apr", "-May", "-Jun",
                "-Jul", "-Aug", "-Sep", "-Oct", "-Nov", "-Dec"]:
        headerm.append(str(y) + mon)
m.columns = ["Member", "Indicator", "Unit",
             "Classification", "Subclassification", "Remarks"] + headerm
m_data_length = 21 * 12
m.loc[m['Unit'].isnull(), 'Unit'] = ''
m[~m[headerm].applymap(np.isreal)] = np.nan
m[headerm] = m[headerm].astype(np.float)

# Data Stiching

# China, its currency is in LCU/100US$
q.loc[(q['Member'].str.contains('Republic of China')) & (q['Classification'] == 'Exchange Rate'),
      '2000-Q1':'2020-Q4'] = q.loc[(q['Member'].str.contains('Republic of China')) & (q['Classification'] == 'Exchange Rate'),
                                   '2000-Q1':'2020-Q4'] / 100.0

# Viet Nam
tofix = q.loc[(q['Member'] == 'Viet Nam') & (
    q['Classification'] == 'National accounts'), '2000-Q1':'2020-Q4']
tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
          4 == 0] = tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                              4 == 0].values - tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                                                         4 == 3].values
tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
          4 == 3] = tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                              4 == 3].values - tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                                                         4 == 2].values
tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
          4 == 2] = tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                              4 == 2].values - tofix.loc[:, (np.arange(tofix.shape[1]) + 1) %
                                                         4 == 1].values
q.loc[(q['Member'] == 'Viet Nam') & (q['Classification'] ==
                                     'National accounts'), '2000-Q1':'2020-Q4'] = tofix.values

# Gather information
country_names = np.unique(np.array(detail['adbmember']))

# Generate FX: LCU/USD
fx = {}
for country_name in country_names:
    fx[country_name] = {}
    temp = a.loc[(a['Member'] == country_name) & (
        a['Subclassification'] == 'Average Exchange Rate')]
    if temp.shape[0] > 0:
        fx[country_name]['a'] = temp.iloc[0,
                                          6:(6 + a_data_length)].values.astype(np.float)
    else:
        fx[country_name]['a'] = np.zeros(shape=(1, a_data_length))
    temp = q.loc[(q['Member'] == country_name) & (
        q['Subclassification'] == 'Average Exchange Rate')]
    if temp.shape[0] > 0:
        fx[country_name]['q'] = temp.iloc[0,
                                          6:(6 + q_data_length)].values.astype(np.float)
    else:
        fx[country_name]['q'] = np.zeros(shape=(1, q_data_length))
    temp = m.loc[(m['Member'] == country_name) & (
        m['Subclassification'] == 'Average Exchange Rate')]
    if temp.shape[0] > 0:
        fx[country_name]['m'] = temp.iloc[0,
                                          6:(6 + m_data_length)].values.astype(np.float)
    else:
        fx[country_name]['m'] = np.zeros(shape=(1, m_data_length))

# Derive information such as magnitude and currency unit used
# We will later use it to standardize the value

# Annual data
a.loc[:, 'Mag'] = 0
a.loc[a['Unit'].str.contains('Thous|Thousand', regex=True), 'Mag'] = 3
a.loc[a['Unit'].str.contains('Million|Milllion|mn', regex=True), 'Mag'] = 6
a.loc[a['Unit'].str.contains('Billion|Billlion', regex=True), 'Mag'] = 9
# Custom
a.loc[83, 'Mag'] = 5
a.loc[205, 'Mag'] = 7
a.loc[254:258, 'Mag'] = 8
a.loc[472, 'Mag'] = 8

a.loc[:, 'UnitDol'] = 'LCU'
a.loc[a['Unit'].str.contains('US|USD', regex=True), 'UnitDol'] = 'USD'
a.loc[a['Unit'].str.contains(
    '%|LCU/US|Index|index', regex=True), 'UnitDol'] = 'Ratio'
a.loc[a['Unit'].str.contains(
    'Persons|Trips|persons', regex=True), 'UnitDol'] = 'Unit'

# Quarterly data
q.loc[:, 'Mag'] = 0
q.loc[q['Unit'].str.contains('Thous|Thousand', regex=True), 'Mag'] = 3
q.loc[q['Unit'].str.contains(
    'Million|Milllion|mn|Mil.|mil.', regex=True), 'Mag'] = 6
q.loc[q['Unit'].str.contains(
    'Billion|Billlion|Bil.|bil.', regex=True), 'Mag'] = 9
# Custom
q.loc[57, 'Mag'] = 5
q.loc[125, 'Mag'] = 7
q.loc[154:158, 'Mag'] = 8
q.loc[270, 'Mag'] = 8
q.loc[272:274, 'Mag'] = 9

q.loc[:, 'UnitDol'] = 'LCU'
q.loc[q['Unit'].str.contains('US|USD', regex=True), 'UnitDol'] = 'USD'
q.loc[q['Unit'].str.contains(
    '%|LCU/US|Index|index|=100', regex=True), 'UnitDol'] = 'Ratio'
q.loc[q['Unit'].str.contains(
    'Persons|Trips|persons', regex=True), 'UnitDol'] = 'Unit'

# Monthly data
m.loc[:, 'Mag'] = 0
m.loc[m['Unit'].str.contains('Thous|Thousand', regex=True), 'Mag'] = 3
m.loc[m['Unit'].str.contains(
    'Million|Milllion|mn|Mil.|mil.', regex=True), 'Mag'] = 6
m.loc[m['Unit'].str.contains(
    'Billion|Billlion|Bil.|bil.', regex=True), 'Mag'] = 9
# Custom
m.loc[19, 'Mag'] = 5
m.loc[67, 'Mag'] = 7
m.loc[80:84, 'Mag'] = 8
m.loc[151, 'Mag'] = 8

m.loc[:, 'UnitDol'] = 'LCU'
m.loc[m['Unit'].str.contains('US|USD', regex=True), 'UnitDol'] = 'USD'
m.loc[m['Unit'].str.contains(
    '%|LCU/US|Index|index|=100', regex=True), 'UnitDol'] = 'Ratio'
m.loc[m['Unit'].str.contains(
    'Persons|Trips|persons', regex=True), 'UnitDol'] = 'Unit'

# NEXT:
# 1. Modify unit to million USD
target_mag = 6
au = a.copy()
qu = q.copy()
mu = m.copy()
for country_name in country_names:
    au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'LCU'),
           headera] = (au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'LCU'),
                              headera].mul(10.0**(au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'LCU'),
                                                         'Mag'] - target_mag),
                                           axis=0)) / fx[country_name]['a']
    au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'USD'),
           headera] = (au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'USD'),
                              headera].mul(10.0**(au.loc[(au['Member'] == country_name) & (au['UnitDol'] == 'USD'),
                                                         'Mag'] - target_mag),
                                           axis=0))
    au.loc[(au['Member'] == country_name) & (
        au['UnitDol'] == 'LCU'), 'Mag'] = target_mag
    au.loc[(au['Member'] == country_name) & (
        au['UnitDol'] == 'USD'), 'Mag'] = target_mag

    qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'LCU'),
           headerq] = (qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'LCU'),
                              headerq].mul(10.0**(qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'LCU'),
                                                         'Mag'] - target_mag),
                                           axis=0)) / fx[country_name]['q']
    qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'USD'),
           headerq] = (qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'USD'),
                              headerq].mul(10.0**(qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'USD'),
                                                         'Mag'] - target_mag),
                                           axis=0))
    qu.loc[(qu['Member'] == country_name) & (qu['UnitDol'] == 'LCU'),
           ['Mag', 'UnitDol']] = [target_mag, 'USD']
    qu.loc[(qu['Member'] == country_name) & (
        qu['UnitDol'] == 'USD'), 'Mag'] = target_mag

    mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'LCU'),
           headerm] = (mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'LCU'),
                              headerm].mul(10.0**(mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'LCU'),
                                                         'Mag'] - target_mag),
                                           axis=0)) / fx[country_name]['m']
    mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'USD'),
           headerm] = (mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'USD'),
                              headerm].mul(10.0**(mu.loc[(mu['Member'] == country_name) & (mu['UnitDol'] == 'USD'),
                                                         'Mag'] - target_mag),
                                           axis=0))
    mu.loc[(mu['Member'] == country_name) & (
        mu['UnitDol'] == 'LCU'), 'Mag'] = target_mag
    mu.loc[(mu['Member'] == country_name) & (
        mu['UnitDol'] == 'USD'), 'Mag'] = target_mag

# Generate new column fname
au['fname'] = pd.Series(np.arange(au.shape[0])).astype(str).values + '-' + au['Member'].values + '-A-' + \
    au['Classification'].str.replace(
        ' ', '').values + au['Subclassification'].str.replace(' ', '').values
qu['fname'] = pd.Series(np.arange(qu.shape[0])).astype(str).values + '-' + qu['Member'].values + '-Q-' + \
    qu['Classification'].str.replace(
        ' ', '').values + qu['Subclassification'].str.replace(' ', '').values
mu['fname'] = pd.Series(np.arange(mu.shape[0])).astype(str).values + '-' + mu['Member'].values + '-M-' + \
    mu['Classification'].str.replace(
        ' ', '').values + mu['Subclassification'].str.replace(' ', '').values


# Normalize feature values and we store the mean and standard deviation
qu_stat = {}
for fname in qu['fname'].values:
    if 'RealGDPGrowth' not in fname:
        data = qu.loc[(qu['fname'] == fname), '2000-Q1':'2020-Q4'].values
        mean, std = np.nanmean(data), np.nanstd(data)
        qu.loc[(qu['fname'] == fname),
               '2000-Q1':'2020-Q4'] = (qu.loc[(qu['fname'] == fname),
                                              '2000-Q1':'2020-Q4'] - mean) / std
        qu_stat[fname] = mean, std


def deseasonalize(qu=None):
    """
    Deseasonalize quarterly data. We will decompose it first and get trend, seasonality, and random error.
    We then remove seasonality and save it to a dictionary to be used later

    Args:
        qu - quarterly data
    return:
        deseasonalized qu
    """
    seasonalities = {}
    na_data = qu.loc[(qu['fname'].str.contains(
        'Nationalaccounts')), '2000-Q1':'2019-Q4']
    column_ids = qu.loc[(qu['fname'].str.contains(
        'Nationalaccounts')), 'fname'].values
    na_data.index = qu.loc[(
        qu['fname'].str.contains('Nationalaccounts')), 'fname']
    na_data = na_data.T
    na_data.index = pd.to_datetime(na_data.index)
    na_data = na_data.resample('Q').fillna(
        method='ffill').fillna(
        method='bfill')

    for column_id in column_ids:
        seasonalities[column_id] = ''
        if not na_data[column_id].isnull().any():
            dec_na_data = sm.tsa.seasonal_decompose(na_data[column_id])
            qu.loc[(qu['fname'] == column_id),
                   '2000-Q1':'2019-Q4'] = na_data.loc[:,
                                                      column_id].values - dec_na_data.seasonal.values
            seasonalities[column_id] = dec_na_data.seasonal

    # dta = sm.datasets.co2.load_pandas().data.resample(
        # "M").fillna(method='ffill').fillna(method='bfill')
    # dec_dta = sm.tsa.seasonal_decompose(dta)

    return qu, seasonalities


# Deseasonalize all National Accounts data
qu_sa, seasonalities = deseasonalize(qu=qu.copy())

# Draw a sample
length = len(qu.loc[qu['fname'] == '127-Indonesia-Q-NationalaccountsRealGDP',
                    '2000-Q1':'2019-Q4'].values[0])
plt.figure(figsize=(20, 15))
plt.plot(np.arange(length),
         qu.loc[qu['fname'] == '127-Indonesia-Q-NationalaccountsRealGDP',
                '2000-Q1':'2019-Q4'].values[0],
         color='blue')
plt.plot(np.arange(length),
         qu_sa.loc[qu_sa['fname'] == '127-Indonesia-Q-NationalaccountsRealGDP',
                   '2000-Q1':'2019-Q4'].values[0],
         color='red')

# Generate data
# Target: Quarterly RGDP Growth
# Input: ALL non-National Accounts in the same period AND historical
# National Accounts


def gen_q_rgdp_nonNApluslagNA(country_name='Indonesia', qu=None, mu=None):
    """
    Generate a target feature (y) and independent features (x)

    Args:
        country_name - country we want to generate the target and features
        qu - quarterly dataset
        mu - monthly dataset
    return:
        qu - specific qu for the country
        target_fname - target name
    """
    # Change RGDP Growth of country_name to target
    target_fname = qu.loc[(qu['fname'].str.contains(country_name)) & (
        qu['fname'].str.contains('RealGDPGrowth')), 'fname'].values[0]
    qu.loc[(qu['fname'].str.contains(country_name)) & (qu['fname'].str.contains(
        'RealGDPGrowth')), ['Classification', 'fname']] = ['target', 'target']
    # Drop data from 2020-Q1 to 2020-Q4 because mostly it's NA
    # Dropped where there is not enough data
    qu.drop(qu.loc[:, '2020-Q1':'2020-Q4'], axis=1, inplace=True)
    # Slide National Accounts into last quarter
    qu.loc[(qu['fname'].str.contains(country_name)) & (qu['fname'].str.contains('Nationalaccounts')),
           '2000-Q2':'2019-Q4'] = qu.loc[(qu['fname'].str.contains(country_name)) & (qu['fname'].str.contains('Nationalaccounts')),
                                         '2000-Q1':'2019-Q3'].values
    qu.drop(columns=['2000-Q1'], inplace=True)
    # Drop unnecessary columns
    qu.drop(
        columns=[
            'Member',
            'Indicator',
            'Unit',
            'Classification',
            'Subclassification',
            'Remarks',
            'Mag',
            'UnitDol'],
        inplace=True)
    qu = qu.T
    qu.columns = qu.loc['fname', :]
    qu.drop('fname', axis=0, inplace=True)
    return qu, target_fname


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    """
    Generate moving window data

    Args:
        dataset - feature array
        target - target array
        start_index - index where window starts
        end_index - index where window ends
        history_size - window size
        target_size - how many target features we have
        step - step of the moving window
        single_step - if it's single_step prediction, we only add 1 target feature.
    return:
        data and labels
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


# Data : Indonesia
country_name = 'Indonesia'
qu_data, target_fname = gen_q_rgdp_nonNApluslagNA(
    country_name=country_name, qu=qu_sa.copy(), mu=mu.copy())
ts = np.array(pd.to_datetime(qu_data.index))

# Remove any column that contains NA.
# We want to preserve the most data points for target feature
for column in qu_data.columns:
    countnull = qu_data[column].isnull().sum()
    if countnull > 0 and column != 'target':
        qu_data.drop(column, axis=1, inplace=True)

y = qu_data['target'].values.astype('float32')
X = qu_data.loc[:, qu_data.columns != 'target'].values.astype('float32')
no_features = X.shape[1]
no_target = 1

PERCENTAGE_SPLIT = 0.9
SPLIT_IDX = int(PERCENTAGE_SPLIT * len(y))

# This project wants us to predict the future value.
# We will split so that the latest data will be used as testing data.
X_train = X[:SPLIT_IDX]
X_test = X[SPLIT_IDX:]
y_train = y[:SPLIT_IDX]
y_test = y[SPLIT_IDX:]


# Test model 1 - ID
# Hyperparameters
EPOCHS = 10  # 20000
lr = 0.0001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        512,
        activation=tf.keras.activations.linear,
        input_shape=(
            no_features,
        )),
    tf.keras.layers.Dense(256, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(no_features, activation=tf.nn.relu),
    tf.keras.layers.Dense(no_target, activation=tf.keras.activations.linear)
])
opt = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    amsgrad=False)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # loss: 0.1800
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=1, verbose=2)

# Calculate
predictions = model.predict(X_train)
plt.figure(figsize=(20, 15))
plt.plot(np.arange(len(y_train)), y_train, color='blue')
plt.plot(np.arange(len(y_train)), predictions.flatten(), color='red')

# Calculate RMSE for training and testing datasets
rmse = np.sqrt(np.mean((y_train - predictions.flatten())**2))
logging.info('RMSE Training:' + str(rmse))
rmse = np.sqrt(np.mean((y_test - model.predict(X_test).flatten())**2))
logging.info('RMSE Testing: ' + str(rmse))

# Draw with full data points
predictions = model.predict(X)
plt.figure(figsize=(20, 15))
plt.plot(np.arange(len(y)), y, color='blue')
plt.plot(np.arange(len(y)), predictions.flatten(), color='red')

# Will be used later to compare all testing models
predictions_fcnn = model.predict(X)
predictions_fcnn_test = model.predict(X_test)

# Draw error graph
plt.figure(figsize=(20, 15))
for idx, key in enumerate(list(history.history.keys())):
    plt.plot(history.history[key][500:EPOCHS], label=key)
plt.legend(loc='best')

# Save the model
model.save_weights('./models/' + str(int(time.time())) +
                   '-Q-nonNAplusNAlast-Dense1-' + country_name + '.h5')


# Test model 2 - ID
model_xgb = xgb.XGBRegressor()
history = model_xgb.fit(X_train, y_train)
model_xgb.score(X_train, y_train)
pickle.dump(model_xgb, open('./models/' + str(int(time.time())) +
                            '-Q-nonNAplusNAlast-XGB-' + country_name + '.h5', 'wb'))

predictions = model_xgb.predict(X_train)
plt.figure(figsize=(20, 15))
plt.plot(np.arange(len(y_train)), y_train, color='blue')
plt.plot(np.arange(len(y_train)), predictions, color='red')

rmse = np.sqrt(np.mean((y_train - predictions.flatten())**2))
logging.info('RMSE Training:' + str(rmse))
rmse = np.sqrt(np.mean((y_test - model_xgb.predict(X_test))**2))
logging.info('RMSE Testing: ' + str(rmse))

predictions_xgb = model_xgb.predict(X)
predictions_xgb_test = model_xgb.predict(X_test)


# Model 4: Recurrent Neural Networks - Data Processing
# Parameters for multivariate_data
past_history = 4  # Window size is 4
future_target = 0  # Predict current target
STEP = 1  # Moving step
# Hyperparameters
EPOCHS = 10  # 10000
lr = 0.00005
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07


features = qu_data[qu_data.columns[qu_data.columns != 'target']]
target = qu_data['target'].values.astype(np.float32)

TRAIN_SPLIT = int(features.shape[0] * PERCENTAGE_SPLIT)
dataset = features.values.astype(np.float32)

x_train_single, y_train_single = multivariate_data(dataset, target, 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, target,
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
x_raw_single, y_raw_single = multivariate_data(dataset, target,
                                               0, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

logging.info(
    'Single window of past history : {}'.format(
        x_train_single[0].shape))
logging.info(x_train_single.shape)


BATCH_SIZE = 1
BUFFER_SIZE = 1
no_features = x_train_single.shape[2]
train_data_single = tf.data.Dataset.from_tensor_slices(
    (np.asarray(x_train_single).astype(
        np.float32), np.asarray(y_train_single).astype(
            np.float32)))
train_data_single = train_data_single.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices(
    (x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# Model 4 : Input as Time series model
single_step_model = tf.keras.models.Sequential()
single_step_model.add(
    tf.keras.layers.Flatten(
        input_shape=x_train_single[0].shape))
single_step_model.add(
    tf.keras.layers.Dense(
        no_features * 2,
        activation=tf.keras.activations.linear))
single_step_model.add(
    tf.keras.layers.Dense(
        no_features,
        activation=tf.keras.activations.linear))
single_step_model.add(tf.keras.layers.Dense(
    128, activation=tf.keras.activations.linear))
single_step_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
single_step_model.add(
    tf.keras.layers.Dense(
        no_target,
        activation=tf.keras.activations.linear))
opt = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    amsgrad=False)
single_step_model.compile(optimizer=opt, loss='mse')

EVALUATION_INTERVAL = int(np.ceil(x_train_single.shape[0] / BATCH_SIZE))
validation_steps = x_val_single.shape[0] / BATCH_SIZE

logdir = os.path.join("logs", datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + '-RGDP-2DFCNN-' + country_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

single_step_history = single_step_model.fit(train_data_single,
                                            epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=validation_steps,
                                            callbacks=[tensorboard_callback])

predictions = single_step_model.predict(x_train_single)
plt.figure(figsize=(20, 15))
plt.plot(np.arange(len(y_train_single)), y_train_single, color='blue')
plt.plot(np.arange(len(y_train_single)), predictions.flatten(), color='red')

rmse = np.sqrt(np.mean((y_train_single - predictions.flatten())**2))
logging.info('RMSE Training:' + str(rmse))
rmse = np.sqrt(
    np.mean(
        (y_val_single - single_step_model.predict(x_val_single).flatten())**2))
logging.info('RMSE Testing:' + str(rmse))

predictions = single_step_model.predict(x_raw_single)
plt.figure(figsize=(20, 15))
plt.plot(np.arange(len(y_raw_single)), y_raw_single, color='blue')
plt.plot(np.arange(len(y_raw_single)), predictions.flatten(), color='red')


# Plotting
loss = single_step_history.history['loss']
val_loss = single_step_history.history['val_loss']
epochs = range(len(loss))  # Get number of epochs
# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.figure(figsize=(20, 15))
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.legend(loc='best')

single_step_model.save_weights('./models/' +
                               str(int(time.time())) +
                               '-Q-nonNAplusNAlast-2DFCNN-' +
                               country_name +
                               '.h5')

predictions_2dfcnn = single_step_model.predict(x_raw_single)
predictions_2dfcnn_test = single_step_model.predict(
    x_raw_single[-len(y_test):, :, :])


# Draw the comparison between Actual and Model Predictions
# Training
plt.figure(figsize=(20, 15))
plt.plot(ts, y, label='Actual', color='blue')
plt.plot(ts, predictions_fcnn.flatten(), label='FCNN', color='red')
plt.plot(ts, predictions_xgb.flatten(), label='XGB', color='green')
plt.plot(ts[4:len(y)], predictions_2dfcnn.flatten() *
         std + mean, label='2D FCNN', color='orange')
plt.legend(loc='best')

# Testing
plt.figure(figsize=(20, 15))
plt.plot(ts[-len(y_test):], y_test, label='Actual', color='blue')
plt.plot(ts[-len(y_test):],
         predictions_fcnn_test.flatten(),
         label='FCNN',
         color='red')
plt.plot(ts[-len(y_test):],
         predictions_xgb_test.flatten(),
         label='XGB',
         color='green')
plt.plot(ts[-len(predictions_2dfcnn_test.flatten()):],
         predictions_2dfcnn_test.flatten(), label='2D FCNN', color='orange')
plt.legend(loc='best')

# Sleep for 5 min to give frontend enough time to collect data
# logging.info('Waiting for 5 min...')
# time.sleep(5*60)
logging.info('Closing...')
