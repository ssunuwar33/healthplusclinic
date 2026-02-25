# feature_engineering.py

import pandas as pd
import numpy as np


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Dropping last column of dataframe as it is not needed
    df.drop('SILVER_LOADED_AT',axis=1, inplace=True)

    # lowercasing all the columns name
    df.columns = df.columns.str.lower()

    # lowercasing all categorical columns
    cat_features = df.select_dtypes(include=["object", "category", "bool"]).columns
    df[cat_features] = df[cat_features].apply(lambda x: x.str.lower())

    # Pre-process dates (Required for your .dt and .hour calls to work)
    df['appointment_date'] = pd.to_datetime(df['appointment_date'])
    df['appointment_time'] = pd.to_datetime(df['appointment_time'])


    # Day of week
    df['days_week'] = df['appointment_date'].dt.isocalendar().day

    # Weekend
    df['is_weekend'] = df['days_week'].apply(lambda x: x in [6, 7])

    # Hour
    df['hour'] = df['appointment_time'].apply(lambda x: x.hour)

    # Time of day
    bins = [0, 12, 17, 24]
    labels = ['Morning', 'Afternoon', 'Evening']
    df['time_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

    df.drop(columns='hour', inplace=True)

    # Open hours
    df['open_hours'] = (
        pd.to_timedelta(df['hours_end'].astype(str)) -
        pd.to_timedelta(df['hours_start'].astype(str))
    )
    df['open_hours'] = (df['open_hours'].dt.total_seconds() / 3600).astype(int)

    df.drop(['hours_start', 'hours_end'], axis=1, inplace=True)

    # Age group
    bins = [0, 3, 17, 31, 45, np.inf]
    labels = ['babies', 'children', 'young', 'middle_age', 'old']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Sort for historical features
    df = df.sort_values(['patient_id', 'appointment_date'])

    # Total appointments
    df['total_appointments'] = df.groupby('patient_id').cumcount()

    # No show rate history
    df['patient_no_show_rate'] = (
        df.groupby('patient_id')['is_no_show_0_1']
        .transform(lambda x: x.shift().expanding().mean())
    )
    df['patient_no_show_rate'] = df['patient_no_show_rate'].fillna(0)

    # Weekdays
    df['is_weekdays'] = df['days_week'].apply(lambda x: 1 <= x <= 5)

    # Clinic frequency
    clinic_counts = df['clinic_name'].value_counts().to_dict()
    df['clinic_count'] = df['clinic_name'].map(clinic_counts)

    bins = [0, 5000, 10000, np.inf]
    labels = ['low_volume_clinic', 'medium_volume_clinic', 'high_volume_clinic']
    df['patient_clinic_frequency_visit'] = pd.cut(
        df['clinic_count'], bins=bins, labels=labels, right=False
    )

    # Drop unused columns
    cols = [
        'appointment_id', 'patient_id', 'provider_id',
        'appointment_date', 'appointment_time',
        'days_week', 'provider_clinic_id',
        'specialty', 'clinic_name', 'clinic_count'
    ]

    df = df.drop(columns=cols)

    

    return df

#data = pd.read_csv('D:/healthplusclinic/data/01-rawdata/rawdata.csv')
#pre = feature_engineering(data)
#print(pre.head())
