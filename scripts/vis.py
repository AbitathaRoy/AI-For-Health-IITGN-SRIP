# vis.py

import pandas as pd
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

import argparse

# Handle parsed directories
parser = argparse.ArgumentParser(description="Visualize health data.")
parser.add_argument("-name", type=str, required=True, help="Path to the input data folder.")
# parser.add_argument("-out_dir", type=str, required=True, help="Path to the output dataset folder.")
args = parser.parse_args()

path = args.name
# output_folder = args.out_dir

# Traverse input path and collect patient data directories
# patient_paths = [f.path for f in os.scandir(input_folder) if f.is_dir()]
# print(patient_paths)

# dataframes = []

# for i, path in enumerate(patient_paths):

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

# ---- 3. PLOT ------

window_duration = pd.Timedelta(minutes=5)   # mastering Timedelta() is only going to help us

# Get the absolute start and end of the recording
recording_start = combined_signals['timestamp'].iloc[0]
recording_end = combined_signals['timestamp'].iloc[-1]  # another way of writing iloc

# Initialize the multi-page PDF document
pdf_filename = f"Visualizations/{var_name}_report.pdf"

# Using PdfPages to create multi-page PDF
with PdfPages(pdf_filename) as pdf:

    current_start = recording_start
    while current_start < recording_end:
        current_end = current_start + window_duration

        # Part 1: Slice signals for the current 5-min window
        # flag: change to combined_signals["timestamp"] <= current_end if breaks show in graph (it won't)
        mask = (combined_signals["timestamp"] >= current_start) & (combined_signals["timestamp"] < current_end)
        window_signals = combined_signals.loc[mask]

        # Skip if no data, like a gap or eof
        if window_signals.empty:
            current_start = current_end
            continue

        # Part 2: Set up the plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(30, 10), sharex=True)

        str_start = current_start.strftime("%d/%m/%Y %H:%M:%S")
        str_end = current_end.strftime("%H:%M:%S")
        fig.suptitle(f"Patient {var_name} | {str_start} to {str_end}",
                     fontsize=18,
                     fontweight="bold"
                     )  # supertitle? interesting!
        
        # Plot nasal airflow
        ax1.plot(window_signals["timestamp"], 
                window_signals["nasal_airflow"], 
                color="blue",
                label="Nasal Flow")
        ax1.set_ylabel("Nasal Airflow (L/min)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot thoracic movement
        ax2.plot(window_signals["timestamp"], 
                window_signals["thoracic_movement"], 
                color="orange",
                label="Thoracic/Abdominal Resp.")
        ax2.set_ylabel("Resp. Amplitude")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot SPO2
        ax3.plot(window_signals["timestamp"], 
                window_signals["spo2"], 
                color="darkgreen",
                label="SpO2")
        ax3.set_ylabel("SpO2 (%)")
        ax3.set_xlabel("Time")
        ax3.xaxis.set_major_locator(mdates.SecondLocator(interval=30))  # Places a major tick every 30 seconds
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax3.tick_params(axis='x', labelrotation=90)     # 
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Find events that overlap with this SPECIFIC 5-minute window
        overlapping_events = breathing_irregularity[
            (breathing_irregularity["Timestamp_Start"] < current_end) &
            (breathing_irregularity["Timestamp_End"] > current_start)
        ]

        # Overlay annotations
        for _, event in overlapping_events.iterrows():
            # Bound the highlight box so it doesn't bleed off the edges of the 5-min plot
            highlight_start = max(event['Timestamp_Start'], current_start)
            highlight_end = min(event['Timestamp_End'], current_end)
            event_type = str(event[2])

            if event_type == "Hypopnea":
                color = "yellow"
            elif event_type == "Obstructive Apnea":
                color = "red"
            else:
                color = "purple"
            
            ax1.text(highlight_start,
                     ax1.get_ylim()[1] + 0.15,   # get_ylim() returns (bottom, top) positional limits of current y-axis
                     event_type,
                     color="black",
                     fontsize=11,
                     verticalalignment="top",
                     horizontalalignment="left"
                     )
            ax1.axvspan(xmin=highlight_start,
                        xmax=highlight_end,
                        color=color,
                        alpha=0.2)
            
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Move to the next 5-minute chunk
        current_start = current_end

    print(f"Successfully generated multi-page PDF: {pdf_filename}")