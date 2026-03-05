# create_dataset.py

import pandas as pd
from datetime import datetime

dataframes = []
patient_paths = ["Data/AP01",
                 "Data/AP02",
                 "Data/AP03",
                 "Data/AP04",
                 "Data/AP05"]

for i, path in enumerate(patient_paths):
    # Define dict key for each patient
    var_name = path[5:]

    # ---- 1. SIGNALS ---- 
    # Raw ingest nasal
    nasal_airflow = pd.read_csv(f"{path}/Flow_Signals.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)
    
    # print(nasal_airflow.head(4))
    # Raw ingest thoracic
    thoracic_movement = pd.read_csv(f"{path}/Thorac.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)

    # Adjusting sampling rate for nasal and thoracic readings
    # nasal and thoracic: 32 Hz, spo2: 4 Hz
    # nasal and thoracic data are sampled 8 times higher
    nasal_airflow["group"] = nasal_airflow.index // 8
    thoracic_movement["group"] = thoracic_movement.index // 8

    # Currently, the columns are named 0 and 1
    nasal_airflow = nasal_airflow.groupby("group").agg({
        0: "last",
        1: "mean"
    }).reset_index(drop=True)
    thoracic_movement = thoracic_movement.groupby("group").agg({
        0: "last",
        1: "mean"
    }).reset_index(drop=True)

    spo2 = pd.read_csv(f"{path}/SPO2.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)

    signals_combined = pd.DataFrame({
        "Timestamp": spo2[0],
        "nasal_airflow": nasal_airflow[1],
        "thoracic_movement": thoracic_movement[1],
        "spo2": spo2[1]
    })

    # Converting times to standard datetime object
    signals_combined["Timestamp"] = pd.to_datetime(signals_combined["Timestamp"], 
                                                   format="%d.%m.%Y %H:%M:%S,%f")

    signals_combined = signals_combined.dropna(subset=["Timestamp"])
    signals_combined = signals_combined.sort_values("Timestamp")
    # print(signals_combined.head(4))

    # ---- 2. ANNOTATIONS ----
    # Sleep Profile annotations
    sleep_profile = pd.read_csv(f"{path}/Sleep_Profile.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)
    sleep_profile[0] = pd.to_datetime(sleep_profile[0], 
                                      format="%d.%m.%Y %H:%M:%S,%f")
    
    # Breathing Irregularity annotations
    breathing_irregularity = pd.read_csv(f"{path}/Flow_Events.txt", 
                                skiprows=5,
                                sep=";",
                                header=None)
    
    # Converting to standard datetime objects
    breathing_irregularity[["Timestamp_Start", "Timestamp_End"]] = breathing_irregularity[0].str.split("-",
                                                                                                       expand=True)
    breathing_irregularity[["Date", "Timestamp_Start"]] = breathing_irregularity["Timestamp_Start"].str.split(" ",
                                                                                                              expand=True)
    breathing_irregularity["Date"] = pd.to_datetime(breathing_irregularity["Date"],
                                              format="%d.%m.%Y")
    
    breathing_irregularity["Timestamp_Start"] = pd.to_datetime(breathing_irregularity["Timestamp_Start"], 
                                                   format="%H:%M:%S,%f")
    breathing_irregularity["Timestamp_End"] = pd.to_datetime(breathing_irregularity["Timestamp_End"], 
                                                   format="%H:%M:%S,%f")
    
    breathing_irregularity["Timestamp_Start"] = breathing_irregularity["Date"] + pd.to_timedelta(breathing_irregularity["Timestamp_Start"].dt.strftime("%H:%M:%S.%f"))
    breathing_irregularity["Timestamp_End"] = breathing_irregularity["Date"] + pd.to_timedelta(breathing_irregularity["Timestamp_End"].dt.strftime("%H:%M:%S.%f"))
    
    # print(breathing_irregularity.head(4))

    # Appending annotations to our dataset
    # If constituent dataframes are not sorted by timestamps, sort them
    signals_combined = pd.merge_asof(
        left=signals_combined, 
        right=sleep_profile,
        left_on="Timestamp",
        right_on=0,
        direction="backward"    # Align rows with the lower value of Timestamp
    )
    signals_combined.drop(columns=[0], inplace=True)

    # Using IntervalIndex for left outer join
    intervals = pd.IntervalIndex.from_arrays(
        breathing_irregularity["Timestamp_Start"],
        breathing_irregularity["Timestamp_End"],
        closed="left"
    )
    idx = intervals.get_indexer(signals_combined["Timestamp"])
    signals_combined["breathing_irregularity"] = pd.Series(idx).map(
        lambda i: breathing_irregularity.iloc[i, 2] if i != -1 else None
    )

    signals_combined.drop(columns=[1], inplace=True)

    # Annotating patient name
    signals_combined["patient_code"] = var_name

    print(signals_combined.head(4))

    dataframes.append(signals_combined)