PCA-BASED ANOMALY DETECTION AND ISOLATION FOREST ALGORITHM FOR HYDROLOGICAL DATA: ENHANCING WATER LEVEL ACCURACY THROUGH AI

By
Ghyzyl Mae D. Martinez
Anna Mae V. Serdeña
Christian Bernard P. Sio
John Carlo M. Udani
Dave C. Vegafria

University of Antique

1. Project Overview
This project aims to develop a robust system for detecting false data readings in water level datasets, particularly focusing on anomalies such as unrealistic sudden increases or decreases in water levels relative to rainfall. Two machine learning algorithms are employed: Principal Component Analysis (PCA)-based Anomaly Detection and Isolation Forest. The use of these two distinct methods allows for a comprehensive analysis, comparing the strengths of both to ensure reliable detection of anomalies in the data.

2. Installation Instructions
To set up the project environment and install all necessary dependencies, follow the steps below:

2.1. Clone the repository
2.2. Create a virtual environment (optional but recommended)
2.3. Install the required Python packages
2.4. Prepare your datasets
     Ensure that your water level datasets are formatted correctly, with date-time values and relevant measurements such as rainfall amount and water level.

3. Usage Instructions

Running PCA-Based Anomaly Detection:
1. provide the path where the anomaly data will be saved.
2. Run the PCA Anomaly Detection script
3. Follow the prompts
   - Provide the path to your dataset when prompted.
   - The script will process the data, detect anomalies, and save the results-anomalies, and data including their timestamps-to a specified CSV file.
   - The script will also generate plots showing the original data with detected anomalies and the cleaned data without anomalies, which you can then save for later use.

Running Isolation Forest Anomaly Detection
 1. Provide the file path of the dataset to be scanned.
 2. Provide the file path where you want the anomaly data to be saved.
   - Specify the file name for the anomaly data.
 3. Run the script
   - The script will process the data, detect anomalies, and save the results to a specified CSV file.
   - The script will also generate plots showing the data with and without anomalies, which you can then save for later use.

4. Dependencies
The following Python packages are required to run the scripts:

- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- matplotlib: For plotting graphs and visualizing data.
- seaborn: For enhanced data visualizations.
- scikit-learn: For implementing the Isolation Forest algorithm.
- adtk: For implementing the PCA-based anomaly detection.
- os: For handling file paths and directories.
- warnings: To manage warnings during script execution.
