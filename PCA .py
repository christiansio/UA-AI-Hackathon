import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.detector import PcaAD
from adtk.detector import SeasonalAD

def process_dataset(file_path):
    # Load and validate the entire DataFrame as a multivariate time series
    df = pd.read_csv(file_path, index_col="DATE/TIME READ", parse_dates=True, dayfirst=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = validate_series(df)

    # Interpolate to handle missing data
    df.interpolate(method='linear', inplace=True)
    df = df.dropna()

    # Resample the data to daily frequency
    df = df.resample('D').mean()

    # Forward fill any NaN values that might have resulted from resampling
    df.ffill(inplace=True)

    # Set the frequency
    df.index.freq = 'D'

    # Apply PCA-based anomaly detection
    pca_ad = PcaAD()
    anomalies = pca_ad.fit_detect(df)

    # Check if any anomalies were detected
    if anomalies.sum() == 0:
        print("No anomalies detected.")
    else:
        print(f"Number of anomalies detected: {anomalies.sum()}")

    # Filter out rows with anomalies
    df_clean = df[~anomalies]  # This keeps only the rows where anomalies are False

    # Extract anomalies data
    anomaly_data = df[anomalies].copy()
    anomaly_data['isAnomaly'] = 'Anomaly'  # Add a column to indicate anomaly

    # Save anomalies to CSV at specified path
    # Modify depending on where you want the anomaly file to go
    output_path = r'C:\Users\Chris\Documents\UA-AI Hackathon\PCA-Anomalies\HYDROMET_6_ILOILO_OTON,ILOILO_LAMBUYAO-ABILAYBRIDGE-SPLIT (2).csv'
    anomaly_data.to_csv(output_path, index=True)

    print(f"Anomalies saved to: {output_path}")

    # Plotting the original data with anomalies and the filtered data side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Plot original data with anomalies
    axes[0].set_title('Original Data with Anomalies')
    for col in df.columns:
        axes[0].plot(df.index, df[col], label=f'{col}')
        # Plot anomalies for rainfall amount only
        if col == 'RAINFALL AMOUNT (mm)':
            axes[0].plot(df.index[anomalies], df[col][anomalies], 'ro', label='Anomalies (RAINFALL AMOUNT)')

    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Values')
    axes[0].legend(loc='best')

    # Plot filtered data without anomalies
    axes[1].set_title('Filtered Data without Anomalies')
    for col in df_clean.columns:
        axes[1].plot(df_clean.index, df_clean[col], label=f'{col}')

    axes[1].set_xlabel('Date')
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.show()

# Main loop to allow different datasets to be added
while True:
    # Prompt user for file path
    file_path = input("Enter the path to your dataset (or type 'exit' to quit): ")

    if file_path.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break

    try:
        # Process the dataset
        process_dataset(file_path)
    except Exception as e:
        print(f"An error occurred: {e}. Please try again with a valid file.")
