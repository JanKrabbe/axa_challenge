import os
import zipfile
import pandas as pd
import pyproj
import math
import numpy as np

class CitibikeDataset():
    """
    Load, clean, and preprocess Citibike trip data from CSV files.

    Supports loading data from a single CSV file, a directory containing
    multiple CSV files, or ZIP files containing CSVs.
    """
    def __init__(self, path):
        """
        Initializes the CitibikeDataset by loading data from a file, directory, or ZIP archive.

        Arguments:
            path (str): Path to a CSV file, a directory containing CSV files, or a ZIP file.
        """
        self.df_rides = None
        self.dropped_rows = None
        self.stations = None
        self.duration_mean = None
        self.duration_std = None
        self.x_center = None
        self.y_center = None

        if not os.path.exists(path):
            raise FileNotFoundError(f"The provided path '{path}' does not exist.")

        
        if os.path.isfile(path):
            if path.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    raise ValueError(f"Error reading CSV file '{path}': {e}")
            else:
                raise ValueError("Unsupported file format. Please provide a CSV file if a file is given.")      
        elif os.path.isdir(path):
            # Search for zip files, extract them, and delete the zip files
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.zip'):
                        zip_path = os.path.join(root, file)
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                z.extractall(root)
                            os.remove(zip_path)
                        except Exception as e:
                            print(f"Error extracting ZIP file '{zip_path}': {e}")

            # Load all CSV files in directory
            df_list = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(csv_path)
                            df_list.append(df)
                        except Exception as e:
                            print(f"Error reading CSV file '{csv_path}': {e}")
            if not df_list:
                raise ValueError("No CSV files found in the provided directory.")

            try:
                df = pd.concat(df_list, ignore_index=True)
            except Exception as e:
                raise ValueError(f"Error concatenating DataFrames: {e}")

        required_columns = [
            'ride_id', 'rideable_type', 'started_at', 'ended_at',
            'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id',
            'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError("Does the CSV file contain the Citibike dataset?" 
                f"The following required columns are missing from the dataset: {missing_columns}")

        original_df = df.copy()
        cleaned_df = df.dropna().reset_index(drop=True)
        dropped_mask = ~original_df.index.isin(cleaned_df.index)
        self.dropped_rows = original_df.loc[dropped_mask]

        # Ensure end_station_id is of type string (was not always the case when exploring the data)
        cleaned_df['end_station_id'] = cleaned_df['end_station_id'].astype(str)

        # Compute ride duration in seconds
        cleaned_df['started_at'] = pd.to_datetime(cleaned_df['started_at'], errors='coerce')
        cleaned_df['ended_at'] = pd.to_datetime(cleaned_df['ended_at'], errors='coerce')

        cleaned_df['ride_duration'] = (cleaned_df['ended_at'] - cleaned_df['started_at']).dt.total_seconds()

        self.duration_mean = cleaned_df['ride_duration'].mean()
        self.duration_std = cleaned_df['ride_duration'].std()
        cleaned_df['ride_duration_normalized'] = (cleaned_df['ride_duration'] - self.duration_mean) / self.duration_std


        # Create station information
        self._process_stations(cleaned_df)

        # Compute straight-line distance (in meters) between start and end points
        # Removed due since it takes too long. Needs to be improved if distance should be used
        #cleaned_df['straight_line_distance'] = cleaned_df.apply(self._compute_distance, axis=1)

        # Normalisierte Werte berechnen und mittelwerte und standardabweichungen als attribute speichern

        self.df_rides = cleaned_df

        
    def _compute_distance(self, row):
        """
        Computes the straight-line (great-circle) distance between start and end coordinates.

        Parameters:
            row (pd.Series): A row containing 'start_station_id', 'end_station_id'
        
        Returns:
            float: The computed distance in meters.
        """

        start_station = self.stations[self.stations['station_id'] == row['start_station_id']]
        end_station = self.stations[self.stations['station_id'] == row['end_station_id']]

        start = np.array(start_station.iloc[0][['x', 'y']])
        end = np.array(end_station.iloc[0][['x', 'y']])

        return np.linalg.norm(start - end)

    def _process_stations(self, df):
        """
        Processes the ride DataFrame to extract unique station information and compute usage counts.

        The resulting DataFrame is stored in the 'stations' attribute.
        """
  
        start_stations = df[['start_station_id', 'start_station_name', 'start_lat', 'start_lng']].dropna()
        start_stations = start_stations.drop_duplicates(subset='start_station_id')
        start_stations = start_stations.rename(columns={
            'start_station_id': 'station_id',
            'start_station_name': 'station_name',
            'start_lat': 'lat',
            'start_lng': 'lng'
        })

        end_stations = df[['end_station_id', 'end_station_name', 'end_lat', 'end_lng']].dropna()
        end_stations = end_stations.drop_duplicates(subset='end_station_id')
        end_stations = end_stations.rename(columns={
            'end_station_id': 'station_id',
            'end_station_name': 'station_name',
            'end_lat': 'lat',
            'end_lng': 'lng'
        })

        stations = pd.concat([start_stations, end_stations], ignore_index=True)
        stations = stations.drop_duplicates(subset='station_id')

        # Calculate usage counts
        start_counts = df['start_station_id'].value_counts().rename('start_count')
        end_counts = df['end_station_id'].value_counts().rename('end_count')

        stations = stations.merge(start_counts, left_on='station_id', right_index=True, how='left')
        stations = stations.merge(end_counts, left_on='station_id', right_index=True, how='left')
        stations['start_count'] = stations['start_count'].fillna(0).astype(int)
        stations['end_count'] = stations['end_count'].fillna(0).astype(int)

        # Transform geographic coordinates to Web Mercator (EPSG:3857).
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        stations['x'], stations['y'] = transformer.transform(stations['lng'].values, stations['lat'].values)

        self.x_center = stations['x'].mean()
        self.y_center = stations['y'].mean()

        stations['x_centered'] = (stations['x'] - self.x_center) 
        stations['y_centered'] = (stations['y'] - self.y_center) 

        self.stations = stations