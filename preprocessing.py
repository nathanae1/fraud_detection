
# coding: utf-8


import pandas as pd
import numpy as np
from datetime import datetime
import json


def intersect(df, columns):
    return [col for col in columns if col in df.columns]

def create_target(df):
    df['category'] = df.acct_type.apply(lambda x: x.split('_')[0])
    df['is_fraud'] = (df.category == 'fraudster')
    return df

def drop_columns(df, columns):
    cols = intersect(df, columns)
    df = df.drop(cols, axis=1)
    return df

def build_booleans(df, columns):
    """
    Build boolean columns assessing if value is present.
    Default to false.
    """
    for col in columns:
        try:
            new_col = 'has_' + col
            if df[col].dtype == 'object':
                df[new_col] = ~((df[col] == '') | (df[col].isnull()))

            elif col == 'body_length':
                df[new_col] = ~((df[col] > 0) | (df[col].isnull()))
            else:
                df[new_col] = ~df[col].isnull()
        except KeyError:
            df[new_col] = False

    return df

def build_popularity(df, columns, train, drop=False):
    """
    Build count columns for high cardinality categorical features.
    Store mapping during training in pickle files.
    During testing, default to 1 if value not found in mapping.
    Set drop to true if input columns should be dropped after processing is done.
    """
    cols = intersect(df, columns)
    for col in cols:
        filename = 'data/' + col + '_counts.json'
        if train:
            counts = df[col].value_counts().to_dict()
            with open(filename, 'w') as f:
                json.dump(counts, f)
        else:
            with open(filename, 'r') as f:
                counts = json.load(f)

        new_col = col + '_count'
        df[new_col] = df[col].apply(lambda x: counts.get(x, 1))

    if drop:
        df = drop_columns(df, cols)

    return df

def log_transform(df, columns, drop=True):
    cols = intersect(df, columns)
    for col in cols:
        new_col = 'log_' + col
        df[new_col] = np.log(df[col].astype(int)).replace(-np.inf, -1)

    if drop:
        df = drop_columns(df, cols)

    return df


# In[141]:
def build_event_features(df):

    try:
        df['event_duration'] = (pd.to_datetime(df.event_end, unit='s') - pd.to_datetime(df.event_start, unit='s')) / np.timedelta64(1, 'D')
        df[df.event_duration < 0].event_duration = 0
    except AttributeError:
        df['event_duration'] = 0

    try:
        df['days_to_event_start'] = (pd.to_datetime(df.event_start, unit='s') - pd.to_datetime(df.event_created, unit='s')) / np.timedelta64(1, 'D')
        df[df.days_to_event_start < 0].days_to_event_start = 0
    except AttributeError:
        df['days_to_event_start'] = 0

    return df


def build_dummies(df, columns, drop=True, train_cols=[]):
    """
    Create dummy columns. Add constraints to below dictionary if necessary.
    In predict phase will pick up from the train columns.
    """

    constraints = {'currency': ['US', 'AUD', 'CAD', 'EUR', 'GBP'],
                    'listed': ['y'],
                    'payout_type': ['CHECK'],
                    'delivery_method': ['1', '0']}
    for col in columns:
        if col in df:
            dummies = pd.get_dummies(df[col], prefix=col)
            if col in constraints:
                wanted_cols = ['_'.join([col, val]) for val in constraints[col]]
                dummies = dummies[intersect(dummies, wanted_cols)]
        else:
            dummies = df[[]]
        if train_cols:
            train_dummy_cols = [d_col for d_col in train_cols if col + '_' in d_col]
            missing_cols = [train_col for train_col in train_dummy_cols if not train_col in dummies.columns]
            for new_col in missing_cols:
                dummies[new_col] = 0

        df = df.join(dummies)

    if drop:
        df = drop_columns(df, columns)

    return df

def fill_missing_values(df, column_types, train):
    """
    Filling missing values.
    For categorical variables, might require to input default value below.
    """
    default_values = {'country': 'US'}

    for col_type in column_types:
        col, type = col_type
        if type == 'object':
            filling_value = df[col].mode()[0] if train else default_values.get(col, '')
            try:
                df[col] = df[col].fillna(filling_value)
                df[col] = df[col].replace('', filling_value)
            except KeyError:
                df[col] = default_values.get(col, 0)

        else:
            try:
                df[col] = df[col].fillna(0)
                if col == 'delivery_method':
                    df[col] = df[col].astype(int)
            except KeyError:
                df[col] = 0


#        except:
#

    return df

def preprocess(json_file, train=False):
    print datetime.now(), "Start: Preprocessing %s data" % ('train' if train else 'predict')
    if train:
        df = pd.read_json(json_file)
    else:
        df = pd.DataFrame(pd.Series(json_file)).T

    print df.columns
    print datetime.now(), "File loaded."

    if train:
        # create and separate target
        df = create_target(df)
        y = df.pop('is_fraud')
        df = drop_columns(df, ['sale_duration', 'sale_duration2', 'event_published',
                                'acct_type'])
        print datetime.now(), "Target created."
    else:
        # loading columns from train dataset for comparison
        with open('data/columns.json', 'r') as f:
            train_cols = json.load(f)
        print datetime.now(), "Columns fetched."


    # careful: order is important
    df = build_booleans(df, ['country', 'body_length', 'venue_name'])
    print datetime.now(), "Boolean columns: OK."
    df = fill_missing_values(df, [('country', 'object'), ('delivery_method', 'int64')], train)
    print datetime.now(), "Missing values: OK."
    df = build_popularity(df, ['country', 'email_domain'], train, drop=True)
    print datetime.now(), "Popularity columns: OK."
    df = log_transform(df, ['country_count', 'body_length', 'user_age'], drop=True)
    print datetime.now(), "Log transforms: OK."
    df = build_event_features(df)
    print datetime.now(), "Event time features: OK."

    # make dummies
    dummie_cols = ['payout_type', 'currency', 'listed', 'delivery_method']
    if train:
        df = build_dummies(df, dummie_cols, drop=True)
    else:
        df = build_dummies(df, dummie_cols, drop=True, train_cols=list(zip(*train_cols)[0]))
    print datetime.now(), "Dummies: OK."

    df = drop_columns(df, ['object_id', 'event_created', 'event_end', 'event_start', 'user_created',
        'venue_latitude', 'venue_longitude', 'approx_payout_date'])

    object_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = drop_columns(df, object_cols)

    if not train:
        df = fill_missing_values(df, train_cols, train)

    X = df.values

    if train:
        with open('data/columns.json', 'w') as f:
            json.dump(list((col, str(df[col].dtype)) for col in df.columns), f)
        return X, y

    else:
        return df[list(zip(*train_cols)[0])]

if __name__ == '__main__':
    X, y = preprocess('data/train_new.json', train=True)
