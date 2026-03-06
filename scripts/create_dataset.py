# create_dataset.py

import pandas as pd
from datetime import datetime
from scipy.signal import butter, filtfilt
import numpy as np

import argparse
import os

# Handle parsed directories
parser = argparse.ArgumentParser(description="Process health data.")
parser.add_argument("-in_dir", type=str, required=True, help="Path to the input data folder.")
parser.add_argument("-out_dir", type=str, required=True, help="Path to the output dataset folder.")
args = parser.parse_args()

input_folder = args.in_dir
output_folder = args.out_dir

# Traverse input path and collect patient data directories
patient_paths = [f.path for f in os.scandir(input_folder) if f.is_dir()]
# print(patient_paths)

dataframes = []

for i, path in enumerate(patient_paths):
    
    var_name = path[5:]

    # ---- 1. SIGNALS ------ 

    # Raw ingest nasal
    nasal_airflow = pd.read_csv(f"{path}/Flow_Signals.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)
    b, a = butter(N=2, 
                  Wn=[0.17, 0.4],
                  btype="bandpass",
                  fs=32
                 )  # butter describes the architecture of the filter
    nasal_airflow[2] = filtfilt(b, a, nasal_airflow[1]) # filtfilt actually filters the signal

    # Converting to standard timestamp
    nasal_airflow[0] = pd.to_datetime(nasal_airflow[0],
                                      format="%d.%m.%Y %H:%M:%S,%f")
    nasal_airflow = nasal_airflow.sort_values(0)
    
    # print(nasal_airflow.head(4))

    # Raw ingest thoracic
    thoracic_movement = pd.read_csv(f"{path}/Thorac.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)
    thoracic_movement[2] = filtfilt(b, a, thoracic_movement[1]) # filtfilt actually filters the signal

    # Converting to standard timestamp
    thoracic_movement[0] = pd.to_datetime(thoracic_movement[0],
                                      format="%d.%m.%Y %H:%M:%S,%f")
    thoracic_movement = thoracic_movement.sort_values(0)

    spo2 = pd.read_csv(f"{path}/SPO2.txt", 
                                skiprows=8,
                                sep=";",
                                header=None)
    spo2[0] = pd.to_datetime(spo2[0], 
                             format="%d.%m.%Y %H:%M:%S,%f")
    spo2 = spo2.sort_values(0)
    
    # Aligning data in a dataframe
    combined_signals = pd.DataFrame({
        "timestamp": nasal_airflow[0],
        "nasal_airflow": nasal_airflow[2],
        "thoracic_movement": thoracic_movement[2]
    })

    combined_signals = pd.merge_asof(
        left=combined_signals, 
        right=spo2,
        left_on="timestamp",
        right_on=0,
        direction="backward"    # Align rows with the lower value of Timestamp
    )

    combined_signals.drop(columns=[0], inplace=True)
    combined_signals.rename(columns={1: "spo2"}, inplace=True)

    combined_signals["spo2"] = combined_signals["spo2"].bfill()     # Backfilling any remaining NaN
    # NaN could have occurred if recording for spo2 started a bit later than the other two

    # ---- 2. ANNOTATIONS ------

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

    # Edge Case: Day Change
    midnight_mask = breathing_irregularity["Timestamp_End"] < breathing_irregularity["Timestamp_Start"]
    breathing_irregularity.loc[midnight_mask, "Timestamp_End"] += pd.Timedelta(days=1)
    
    # ---- 3. TENSOR ------

    # Creating windows of 30 s with 15 s overlap
    # Logic: If a data is sampled at 32 Hz, then clusters of 32 * 30 rows constitute a 30 s window
    # Also, upsampling spo2 data from 4 Hz to 32 Hz to prevent sparsity.

    # Total number of windows
    N = (len(nasal_airflow[2]) - 960) // 480 + 1
    tensor = np.empty((N, 960, 3))

    # y-variables to store annotation for each window
    y_sleep = np.empty((N), dtype=object)   
    y_breath = np.empty((N), dtype=object)
    # Lesson: by default, "empty" initialises the type as float64.
    # Trying to assign text to float64 array unit throws error.

    sleep_profile_counter = 0   # counter to traverse through the sleep profile dataset
    breathing_irregularity_counter = 0    # counter to traverse through the flow events dataset
    for i in range(N):
        # Grabbing timestamps before dropping them!
        window_start = combined_signals.iloc[i * 480, 0]
        window_end = combined_signals.iloc[i * 480 + 959, 0]

        combined_slice = combined_signals.iloc[i * 480 : i * 480 + 960, 1:]
        tensor[i] = combined_slice

        # Adding sleep profile annotation
        try:
            while window_start > sleep_profile.iloc[sleep_profile_counter, 0]:
                sleep_profile_counter += 1

            first_part = abs(window_start - sleep_profile.iloc[sleep_profile_counter, 0])
            second_part = abs(window_end - sleep_profile.iloc[sleep_profile_counter, 0])

            # Assign the sleep annotation with the higher overlap with our window
            if first_part >= second_part:
                y_sleep[i] = sleep_profile.iloc[sleep_profile_counter, 1]
            else:
                y_sleep[i] = sleep_profile.iloc[sleep_profile_counter + 1, 1]

        except IndexError:
            # If index goes out of bounds, assign the last recorded sleep state
            y_sleep[i] = sleep_profile.iloc[-1, 1]  

        # Adding breathing irregularity annotation
        while breathing_irregularity_counter < len(breathing_irregularity) and \
            breathing_irregularity.loc[breathing_irregularity_counter, "Timestamp_End"] <= window_start:
            breathing_irregularity_counter += 1

        temp_counter = breathing_irregularity_counter
        while temp_counter < len(breathing_irregularity):
            try:
                overlap = min(window_end, breathing_irregularity.loc[temp_counter, "Timestamp_End"]) \
                            - max(window_start, breathing_irregularity.loc[temp_counter, "Timestamp_Start"])
                if overlap > pd.Timedelta(seconds=0) and overlap >= abs(window_end - window_start) / 2:
                    y_breath[i] = breathing_irregularity.iloc[temp_counter, 2]

            except (IndexError, KeyError):
                y_breath[i] = "Normal"
                break

            if breathing_irregularity.loc[temp_counter, "Timestamp_Start"] >= window_end:
                break

            temp_counter += 1

    y_breath[pd.isnull(y_breath)] = "Normal"
    y_sleep[pd.isnull(y_sleep)] = "Unknown" # Just in case

    # Bundling output dataset
    out_file = os.path.join(output_folder, f"{var_name}_dataset.npz")
    np.savez_compressed(
        out_file,
        X=tensor,
        y_sleep=y_sleep,
        y_breath=y_breath
    )

    print(f"Saved {var_name} with {N} windows to {out_file}")