import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from functools import partial
from scipy.stats import skew, kurtosis, iqr
from tqdm import tqdm_notebook as tqdm
from sklearn.externals import joblib
import multiprocessing as mp
import re


warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Memory saved
def reduce_mem_usage(data, verbose = True):
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return data

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan


    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    # A new batch
    df['app missing'] = df.isnull().sum(axis = 1).values
    df['app EXT_SOURCE_1 * EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['app EXT_SOURCE_1 * EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_2 * EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_1 / DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_2 / DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_3 / DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_1 * DAYS_BIRTH'] = df['EXT_SOURCE_1'] * df['DAYS_BIRTH']
    df['app EXT_SOURCE_2 * DAYS_BIRTH'] = df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['app EXT_SOURCE_3 * DAYS_BIRTH'] = df['EXT_SOURCE_3'] * df['DAYS_BIRTH']
    df['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
    df['app most popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
                        .isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
    df['app popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
                        .isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})
    df['app DAYS_EMPLOYED - DAYS_BIRTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']

    # Traditional used
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['annuity_income_percentage'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['app EXT_SOURCE_1 * CA_RATIO'] = df['EXT_SOURCE_1'] * df['credit_to_annuity_ratio']
    df['app EXT_SOURCE_2 * CA_RATIO'] = df['EXT_SOURCE_2'] * df['credit_to_annuity_ratio']
    df['app EXT_SOURCE_3 * CA_RATIO'] = df['EXT_SOURCE_3'] * df['credit_to_annuity_ratio']
    df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    df['app EXT_SOURCE_1 / CD_RATIO'] = df['EXT_SOURCE_1'] / df['credit_to_goods_ratio']
    df['app EXT_SOURCE_2 / CD_RATIO'] = df['EXT_SOURCE_2'] / df['credit_to_goods_ratio']
    df['app EXT_SOURCE_3 / CD_RATIO'] = df['EXT_SOURCE_3'] / df['credit_to_goods_ratio']
    df['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_1 * EB_RATIO'] = df['EXT_SOURCE_1'] * df['days_employed_percentage']
    df['app EXT_SOURCE_2 * EB_RATIO'] = df['EXT_SOURCE_2'] * df['days_employed_percentage']
    df['app EXT_SOURCE_3 * EB_RATIO'] = df['EXT_SOURCE_3'] * df['days_employed_percentage']
    df['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['income_per_child'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['app EXT_SOURCE_1 / PM_RATIO'] = df['EXT_SOURCE_1'] / df['payment_rate']
    df['app EXT_SOURCE_2 / PM_RATIO'] = df['EXT_SOURCE_2'] / df['payment_rate']
    df['app EXT_SOURCE_3 / PM_RATIO'] = df['EXT_SOURCE_3'] / df['payment_rate']
    df['phone_to_birth_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['phone_to_employ_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['external_sources_weighted'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
    df['cnt_non_child'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['child_to_non_child_ratio'] = df['CNT_CHILDREN'] / df['cnt_non_child']
    df['income_per_non_child'] = df['AMT_INCOME_TOTAL'] / df['cnt_non_child']
    df['credit_per_person'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['credit_per_child'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    df['credit_per_non_child'] = df['AMT_CREDIT'] / df['cnt_non_child']
    df['short_employment'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)
    df['young_age'] = (df['DAYS_BIRTH'] < -14000).astype(int)




    # Flag
    df["NEW_REG_IND_SUM"] = df[
                ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                 "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].sum(axis=1)
    df["NEW_REG_IND_KURT"] = df[
                ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                 "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].kurtosis(axis=1)

    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # External sources
    df['external_sources_weighted'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4

    # Try to aggregate
    NEW_AGGREGATION_RECIPIES = [
            (["CODE_GENDER",
              "NAME_EDUCATION_TYPE"], [("AMT_ANNUITY", "max"),
                                       ("AMT_CREDIT", "max"),
                                       ("EXT_SOURCE_1", "median"),
                                       ("EXT_SOURCE_2", "median"),
                                       ("OWN_CAR_AGE", "max"),
                                       ("OWN_CAR_AGE", "sum"),
                                       ("credit_to_annuity_ratio", "median"),
                                       ("external_sources_mean", "median"),
                                       ("credit_to_goods_ratio", "median"),
                                       ("NEW_SOURCES_PROD", "median"),
                                       ("phone_to_birth_ratio", "median"),
                                       ("car_to_employ_ratio", "median"),
                                       ("NEW_SCORES_STD", "median"),
                                       ("phone_to_employ_ratio", "median")]),

            (["CODE_GENDER",
              "ORGANIZATION_TYPE"], [("AMT_ANNUITY", "median"),
                                     ("AMT_INCOME_TOTAL", "median"),
                                     ("DAYS_REGISTRATION", "median"),
                                     ("EXT_SOURCE_1", "median"),
                                     ("credit_to_annuity_ratio", "median"),
                                     ("external_sources_mean", "median"),
                                     ("credit_to_goods_ratio", "median"),
                                     ("NEW_SOURCES_PROD", "median"),
                                     ("car_to_employ_ratio", "median"),
                                     ("phone_to_birth_ratio", "median"),
                                     ("NEW_SCORES_STD", "median"),
                                     ("annuity_income_percentage", "median"),
                                     ("days_employed_percentage", "median"),
                                     ("phone_to_employ_ratio", "median")]),

            (["CODE_GENDER",
              "REG_CITY_NOT_WORK_CITY"], [("AMT_ANNUITY", "median"),
                                          ("DAYS_ID_PUBLISH", "median"),
                                          ("EXT_SOURCE_1", "median"),
                                          ("credit_to_annuity_ratio", "median"),
                                          ("external_sources_mean", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("car_to_employ_ratio", "median"),
                                          ("phone_to_birth_ratio", "median"),
                                          ("NEW_SCORES_STD", "median"),
                                          ("annuity_income_percentage", "median"),
                                          ("days_employed_percentage", "median"),
                                          ("phone_to_employ_ratio", "median")]),

            (["CODE_GENDER",
              "NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE",
              "REG_CITY_NOT_WORK_CITY"], [("EXT_SOURCE_1", "median"),
                                          ("EXT_SOURCE_2", "median"),
                                          ("credit_to_annuity_ratio", "median"),
                                          ("external_sources_mean", "median"),
                                          ("credit_to_goods_ratio", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("car_to_employ_ratio", "median"),
                                          ("phone_to_birth_ratio", "median"),
                                          ("NEW_SCORES_STD", "median"),
                                          ("annuity_income_percentage", "median"),
                                          ("days_employed_percentage", "median"),
                                          ("phone_to_employ_ratio", "median")]),
            (["NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE"], [("AMT_CREDIT", "median"),
                                   ("AMT_REQ_CREDIT_BUREAU_YEAR", "median"),
                                   ("APARTMENTS_AVG", "median"),
                                   ("BASEMENTAREA_AVG", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("NONLIVINGAREA_AVG", "median"),
                                   ("OWN_CAR_AGE", "median"),
                                   ("YEARS_BUILD_AVG", "median"),
                                   ("credit_to_annuity_ratio", "median"),
                                   ("external_sources_mean", "median"),
                                   ("credit_to_goods_ratio", "median"),
                                   ("NEW_SOURCES_PROD", "median"),
                                   ("car_to_employ_ratio", "median"),
                                   ("phone_to_birth_ratio", "median"),
                                   ("NEW_SCORES_STD", "median"),
                                   ("annuity_income_percentage", "median"),
                                   ("days_employed_percentage", "median"),
                                   ("phone_to_employ_ratio", "median")]),

            (["NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE",
              "REG_CITY_NOT_WORK_CITY"], [("ELEVATORS_AVG", "median"),
                                          ("EXT_SOURCE_1", "median"),
                                          ("credit_to_annuity_ratio", "median"),
                                          ("external_sources_mean", "median"),
                                          ("credit_to_goods_ratio", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("car_to_employ_ratio", "median"),
                                          ("phone_to_birth_ratio", "median"),
                                          ("NEW_SCORES_STD", "median"),
                                          ("annuity_income_percentage", "median"),
                                          ("days_employed_percentage", "median"),
                                          ("phone_to_employ_ratio", "median")]),

            (["OCCUPATION_TYPE"], [("AMT_ANNUITY", "median"),
                                   ("DAYS_BIRTH", "median"),
                                   ("DAYS_EMPLOYED", "median"),
                                   ("DAYS_ID_PUBLISH", "median"),
                                   ("DAYS_REGISTRATION", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("EXT_SOURCE_3", "median"),
                                   ("credit_to_annuity_ratio", "median"),
                                   ("external_sources_mean", "median"),
                                   ("credit_to_goods_ratio", "median"),
                                   ("NEW_SOURCES_PROD", "median"),
                                   ("car_to_employ_ratio", "median"),
                                   ("phone_to_birth_ratio", "median"),
                                   ("NEW_SCORES_STD", "median"),
                                   ("annuity_income_percentage", "median"),
                                   ("days_employed_percentage", "median"),
                                   ("phone_to_employ_ratio", "median")]),
        ]
    for groupby_cols, specs in NEW_AGGREGATION_RECIPIES:
        group_object = df.groupby(groupby_cols)
        for select, agg in specs:
            groupby_aggregate_name = "{}_{}_{}_{}".format("NEW", "_".join(groupby_cols), agg, select)
            df = df.merge(
                group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name}),
                left_on=groupby_cols,
                right_on=groupby_cols,
                how="left")


    cols_to_agg = ['AMT_CREDIT',
               'AMT_ANNUITY',
               'AMT_INCOME_TOTAL',
               'AMT_GOODS_PRICE',
               'EXT_SOURCE_1',
               'EXT_SOURCE_2',
               'EXT_SOURCE_3',
               'OWN_CAR_AGE',
               'REGION_POPULATION_RELATIVE',
               'DAYS_REGISTRATION',
               'CNT_CHILDREN',
               'CNT_FAM_MEMBERS',
               'DAYS_ID_PUBLISH',
               'DAYS_BIRTH',
               'DAYS_EMPLOYED'
    ]

    aggs = ['min', 'mean', 'max', 'var']
    aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

    APPLICATION_AGGREGATION_RECIPIES = [
        (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
        (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
        (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
     ]

    for groupby_cols, specs in APPLICATION_AGGREGATION_RECIPIES:
        group_object = df.groupby(groupby_cols)
        for select, agg in specs:
            groupby_aggregate_name = "{}_{}_{}_{}".format("NEW", "_".join(groupby_cols), agg, select)
            df = df.merge(
                group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name}),
                left_on=groupby_cols,
                right_on=groupby_cols,
                how="left")


    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df["NEW_DOC_IND_SUM"] = df[
                [col for col in df.columns if re.search(r"FLAG_DOCUMENT", col)]].sum(axis=1)
    df["NEW_DOC_IND_KURT"] = df[
                [col for col in df.columns if re.search(r"FLAG_DOCUMENT", col)]].kurtosis(axis=1)
    df["NEW_LIVE_IND_SUM"] = df[
                ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].sum(axis=1)
    df["NEW_CONTACT_IND_SUM"] = df[
                ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                 "FLAG_EMAIL"]].sum(axis=1)
    df["NEW_CONTACT_IND_KURT"] = df[
                ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                 "FLAG_EMAIL"]].kurtosis(axis=1)
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return reduce_mem_usage(df)

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bb = reduce_mem_usage(pd.read_csv('bureau_balance.csv', nrows = num_rows), verbose = False)

     # Some new features in bureau_balance set
    tmp = bb[['SK_ID_BUREAU', 'STATUS']].groupby('SK_ID_BUREAU')
    tmp_last = tmp.last()
    tmp_last.columns = ['First_status']
    bb = bb.join(tmp_last, how = 'left', on = 'SK_ID_BUREAU')
    tmp_first = tmp.first()
    tmp_first.columns = ['Last_status']
    bb = bb.join(tmp_first, how = 'left', on = 'SK_ID_BUREAU')
    del tmp, tmp_first, tmp_last
    gc.collect()

    tmp = bb[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['Month']
    bb = bb.join(tmp, how = 'left', on = 'SK_ID_BUREAU')
    del tmp
    gc.collect()

    tmp = bb.loc[bb['STATUS'] == 'C', ['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['When_closed']
    bb = bb.join(tmp, how = 'left', on = 'SK_ID_BUREAU')
    del tmp
    gc.collect()

    bb['Month_closed_to_end'] = bb['Month'] - bb['When_closed']

    for c in range(6):
        tmp = bb.loc[bb['STATUS'] == str(c), ['SK_ID_BUREAU', 'MONTHS_BALANCE']] .groupby('SK_ID_BUREAU').count()
        tmp.columns = ['DPD_' + str(c) + '_cnt']
        bb = bb.join(tmp, how = 'left', on = 'SK_ID_BUREAU')
        bb['DPD_' + str(c) + ' / Month'] = bb['DPD_' + str(c) + '_cnt'] / bb['Month']
        del tmp
        gc.collect()
    bb['Non_zero_DPD_cnt'] = bb[['DPD_1_cnt', 'DPD_2_cnt', 'DPD_3_cnt', 'DPD_4_cnt', 'DPD_5_cnt']].sum(axis = 1)
    bb= reduce_mem_usage(bb)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    del bb
    gc.collect()

    bureau = reduce_mem_usage(pd.read_csv('bureau.csv', nrows = num_rows), verbose = False)
    # Attempt to clean up
    bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del  bb_agg
    gc.collect()

    def correct(x):
        if x < 0:
            return 0
        elif x > 1:
            return 1
        else:
            return x

    bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"] = bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"] = bureau["AMT_CREDIT_SUM_LIMIT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"] = bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_ANNUITY"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"] = bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"] = bureau["AMT_CREDIT_SUM_LIMIT"] / bureau["AMT_ANNUITY"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"] = bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_ANNUITY"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"] = bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"].apply(lambda x: correct(x))
    bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"] = bureau["AMT_CREDIT_MAX_OVERDUE"] / bureau["AMT_CREDIT_SUM_OVERDUE"].replace(0, np.nan)
    bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"] = bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"].apply(lambda x: correct(x))
    #  A new batch of variables
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_OVERDUE']
    bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = bureau['DAYS_CREDIT'] - bureau['CREDIT_DAY_OVERDUE']
    bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']


    # Bureau and bureau_balance numeric features
    num_aggregations = {
        "DAYS_CREDIT": [ 'mean', 'var','min','max','size'],
        "DAYS_CREDIT_ENDDATE": [ 'mean'],
        "DAYS_ENDDATE_FACT": ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean','var','min','max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean','max'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum','max','min','var'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum', 'max','min','var'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean','var','min','max'],
        'AMT_CREDIT_SUM_LIMIT': ['min', 'max','mean','var'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum','max','mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        "NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY":['max', 'mean'],
        "NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY":['max', 'mean'],
        "NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT":['max', 'mean'],
        'bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT':[ 'mean', 'sum', 'max'],
        'bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT':[ 'mean', 'sum', 'min'],
        'bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE':[ 'mean', 'sum', 'min'],
        'bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE':[ 'mean', 'max', 'min'],
        'bureau DAYS_CREDIT - DAYS_ENDDATE_FACT':[ 'mean',  'max'],
        'bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE':[ 'mean','max','min'],
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return reduce_mem_usage(bureau_agg)

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    prev.loc[prev['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
    prev.loc[prev['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # A new batch
    prev['prev missing'] = prev.isnull().sum(axis = 1).values
    prev['prev AMT_APPLICATION - AMT_CREDIT'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['prev AMT_APPLICATION / AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] / prev['AMT_GOODS_PRICE']
    prev['prev AMT_GOODS_PRICE / AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
    prev['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DRAWING'] - prev['DAYS_FIRST_DUE']
    prev['prev DAYS_TERMINATION less -500'] = (prev['DAYS_TERMINATION'] < -500).astype(int)
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','min','mean','sum'],
        'AMT_CREDIT': [ 'max','min', 'mean','sum'],
        'APP_CREDIT_PERC': [ 'max', 'mean','min'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean','min'],
        'AMT_GOODS_PRICE': [ 'max', 'mean','min'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean','min'],
        'DAYS_DECISION': [ 'max', 'min', 'mean'],
        'CNT_PAYMENT': ['mean','max','min','sum'],
        'DAYS_LAST_DUE': [ 'max','mean','min'],
        'DAYS_TERMINATION': [ 'max','min'],
        'DAYS_FIRST_DRAWING':[ 'max','min'],
        'DAYS_FIRST_DUE':[ 'max','min'],
        'DAYS_LAST_DUE_1ST_VERSION':[ 'max','min'],
        'prev missing':['mean'],
        'prev AMT_APPLICATION - AMT_CREDIT':['mean','max','sum'],
        'prev AMT_APPLICATION / AMT_GOODS_PRICE':['mean','max','sum'],
        'prev AMT_GOODS_PRICE / AMT_CREDIT':['mean','max','sum'],
        'prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE': ['mean'],
        'prev DAYS_TERMINATION less -500':['mean'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return reduce_mem_usage(prev_agg)

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    pos['SK_DPD_NOTIGNORED'] = pos['SK_DPD'] - pos['SK_DPD_DEF']
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'SK_DPD_NOTIGNORED':['max','mean','sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return reduce_mem_usage(pos_agg)

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return reduce_mem_usage(ins_agg)

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc['AMT_DRAWINGS_ATM_CURRENT'][cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    cc['AMT_DRAWINGS_CURRENT'][cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)

    # Replace some outliers
    cc.loc[cc['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
    cc.loc[cc['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan

    cc['card missing'] = cc.isnull().sum(axis = 1).values
    cc['card SK_DPD - MONTHS_BALANCE'] = cc['SK_DPD'] - cc['MONTHS_BALANCE']
    cc['card SK_DPD_DEF - MONTHS_BALANCE'] = cc['SK_DPD_DEF'] - cc['MONTHS_BALANCE']
    cc['card SK_DPD - SK_DPD_DEF'] = cc['SK_DPD'] - cc['SK_DPD_DEF']
    cc['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_RECIVABLE']
    cc['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = cc['AMT_RECIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']

    cc['card AMT_BALANCE - AMT_RECIVABLE'] = cc['AMT_BALANCE'] - cc['AMT_RECIVABLE']
    cc['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = cc['AMT_BALANCE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = cc['AMT_BALANCE'] - cc['AMT_TOTAL_RECEIVABLE']

    cc['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_ATM_CURRENT']
    cc['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_OTHER_CURRENT']
    cc['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_POS_CURRENT']
    cc["NEW_AMT_BALANCE_DIVIDE_AMT_CREDIT_LIMIT_ACTUAL"] = cc["AMT_BALANCE"] /cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["NEW_AMT_BALANCE_DIVIDE_AMT_TOTAL_RECEIVABLE"] = cc["AMT_BALANCE"] /cc["AMT_TOTAL_RECEIVABLE"].replace(0, np.nan)
    cc["NEW_AMT_DRAWINGS_ATM_CURRENT_DIVIDE_CNT_DRAWINGS_ATM_CURRENT"] = cc["AMT_DRAWINGS_ATM_CURRENT"] /cc["CNT_DRAWINGS_ATM_CURRENT"].replace(0, np.nan)
    cc["NEW_AMT_DRAWINGS_CURRENT_DIVIDE_CNT_DRAWINGS_CURRENT"] = cc["AMT_DRAWINGS_CURRENT"] /cc["CNT_DRAWINGS_CURRENT"].replace(0, np.nan)
    cc["NEW_AMT_DRAWINGS_OTHER_CURRENT_DIVIDE_CNT_DRAWINGS_OTHER_CURRENT"] = cc["AMT_DRAWINGS_OTHER_CURRENT"] /cc["CNT_DRAWINGS_OTHER_CURRENT"].replace(0, np.nan)
    cc["NEW_AMT_DRAWINGS_POS_CURRENT_DIVIDE_CNT_DRAWINGS_POS_CURRENT"] = cc["AMT_DRAWINGS_POS_CURRENT"] /cc["CNT_DRAWINGS_POS_CURRENT"].replace(0, np.nan)
    cc["NEW_AMT_INST_MIN_REGULARITY_DIVIDE_AMT_PAYMENT_TOTAL_CURRENT"] = cc["AMT_INST_MIN_REGULARITY"] /cc["AMT_PAYMENT_TOTAL_CURRENT"].replace(0, np.nan)
    aggregations = {
        'MONTHS_BALANCE': ['max','mean'],
        'AMT_BALANCE': ['max', 'mean', 'sum','var' ],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum','var' ],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'sum','var' ],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum','var' ],
        'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum','var' ],
        'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean', 'sum','var' ],
        'AMT_RECIVABLE': ['max', 'mean', 'sum','var' ],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean', 'sum','var' ],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum','var' ],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum','var' ],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum','var' ],
        'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum','var' ],
        'SK_DPD': ['max', 'mean', 'sum','var' ],
        'NEW_AMT_BALANCE_DIVIDE_AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'var'],
        "NEW_AMT_BALANCE_DIVIDE_AMT_TOTAL_RECEIVABLE": ['max', 'mean', 'min' ],
        "NEW_AMT_DRAWINGS_ATM_CURRENT_DIVIDE_CNT_DRAWINGS_ATM_CURRENT": ['max', 'mean','std' ],
        "NEW_AMT_DRAWINGS_CURRENT_DIVIDE_CNT_DRAWINGS_CURRENT": ['max', 'mean','std'],
        "NEW_AMT_DRAWINGS_OTHER_CURRENT_DIVIDE_CNT_DRAWINGS_OTHER_CURRENT": ['max', 'mean'],
        "NEW_AMT_DRAWINGS_POS_CURRENT_DIVIDE_CNT_DRAWINGS_POS_CURRENT": ['max', 'mean', 'std'],
        'NEW_AMT_INST_MIN_REGULARITY_DIVIDE_AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum','var' ],
        'card missing': ['mean'],
        'card SK_DPD - MONTHS_BALANCE': ['median'],
        'card SK_DPD_DEF - MONTHS_BALANCE': ['median'],
        'card SK_DPD - SK_DPD_DEF':['median'],
        'card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE': ['median'],
        'card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL':['median'],
        'card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL' :['median'],
        'card AMT_BALANCE - AMT_RECIVABLE': ['median'],
        'card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL' :['median'],
        'card AMT_BALANCE - AMT_TOTAL_RECEIVABLE' :['median'],
        'card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT': ['median'],
        'card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT': ['median'],
        'card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT': ['median'],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return reduce_mem_usage(cc_agg)

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            is_unbalance=False,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=30,
            colsample_bytree=0.05,
            subsample=1,
            max_depth= 8,
            reg_alpha=0,
            reg_lambda=100,
            min_split_gain=0.5,
            min_child_weight=70,
            silent= -1,
            verbose= -1,
            max_bin= 300,
            subsample_freq= 1
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        # Second batch
        df['app EXT_SOURCE_1 / ACTIVE_AMT_CREDIT_SUM_SUM'] = df['EXT_SOURCE_1'] / df['ACTIVE_AMT_CREDIT_SUM_SUM']
        df['app EXT_SOURCE_2 / ACTIVE_AMT_CREDIT_SUM_SUM'] = df['EXT_SOURCE_2'] / df['ACTIVE_AMT_CREDIT_SUM_SUM']
        df['app EXT_SOURCE_3 / ACTIVE_AMT_CREDIT_SUM_SUM'] = df['EXT_SOURCE_3'] / df['ACTIVE_AMT_CREDIT_SUM_SUM']
        df['app EXT_SOURCE_1 * INSTAL_DBD_SUM'] = df['EXT_SOURCE_1'] * df['INSTAL_DBD_SUM']
        df['app EXT_SOURCE_2 * INSTAL_DBD_SUM'] = df['EXT_SOURCE_2'] * df['INSTAL_DBD_SUM']
        df['app EXT_SOURCE_3 * INSTAL_DBD_SUM'] = df['EXT_SOURCE_3'] * df['INSTAL_DBD_SUM']
        df['app CNT_CHILDREN * REGION_POPULATION_RELATIVE'] = df['CNT_CHILDREN'] * df['REGION_POPULATION_RELATIVE']
        df['AMT_GOODS_PRICE / DAYS_BIRTH'] = df['AMT_GOODS_PRICE'] / df['DAYS_BIRTH']
        df['DAYS_BIRTH / REGION_POPULATION_RELATIVE'] =  df['DAYS_BIRTH'] / df['REGION_POPULATION_RELATIVE']


        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
