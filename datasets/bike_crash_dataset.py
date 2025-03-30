import pandas as pd
import pyproj

class BikeCrashDataset():
    """
    Preprocess and standardize NYPD crash data to bike crash dataset.
    """

    REQUIRED_COLUMNS = {
        'CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE', 'CRASH_DATETIME', 'x', 'y'
    }

    def __init__(self, path):
        """
        Initialize the BikeCrashDataset with a DataFrame.

        Arguments:
            path (str): Path to the CSV file containing the crash data.
        """
        self.df = pd.read_csv(path)

        if not self._has_required_columns():
            self._process_dataset()

    def _has_required_columns(self) -> bool:
        """
        Check if the DataFrame already has the required columns.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """
        return self.REQUIRED_COLUMNS.issubset(self.df.columns)

    def _process_dataset(self) -> None:
        """
        Extract bike crahes from data and apply preprocessing.
        """
        condition_cyclist = (
            self.df.get('NUMBER OF CYCLIST INJURED', 0) > 0
        ) | (
            self.df.get('NUMBER OF CYCLIST KILLED', 0) > 0
        )

        vehicle_cols = [
            'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
            'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4',
            'VEHICLE TYPE CODE 5'
        ]
        vehicle_conditions = []
        for col in vehicle_cols:
            if col in self.df.columns:
                cond = self.df[col].astype(str).str.lower().str.contains('bic|bik', na=False)
                vehicle_conditions.append(cond)

        if vehicle_conditions:
            vehicle_condition = pd.concat(vehicle_conditions, axis=1).any(axis=1)
        else:
            vehicle_condition = pd.Series([False] * len(self.df), index=self.df.index)

        combined_condition = condition_cyclist | vehicle_condition
        self.df = self.df[combined_condition]

        essential_cols = ['CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE']
        self.df = self.df.dropna(subset=essential_cols)

        self.df = self.df[
            (self.df['LATITUDE'] != 0) & (self.df['LONGITUDE'] != 0)
        ]

        self.df['CRASH_DATETIME'] = pd.to_datetime(
            self.df['CRASH DATE'] + ' ' + self.df['CRASH TIME'],
            errors='coerce'
        )

        # Transform coordinates from EPSG:4326 to EPSG:3857 (Web Mercator).
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_coords, y_coords = transformer.transform(
            self.df['LONGITUDE'].values,
            self.df['LATITUDE'].values
        )
        self.df['x'] = x_coords
        self.df['y'] = y_coords

        self.df = self.df[['CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE', 'CRASH_DATETIME', 'x', 'y']]

    def citibike_alignment(self, citibike_dataset) :
        """
        Calculates centered the 'x' and 'y' coordinates of the bike crash dataset given
        Citibike dataset and constrains Data to relevant area.

        Parameters:
            citibike_dataset: The Citibike dataset which stores its center values
        """

        self.df['x_centered'] = self.df['x'] - citibike_dataset.x_center
        self.df['y_centered'] = self.df['y'] - citibike_dataset.y_center

        min_x = citibike_dataset.stations["x_centered"].min()
        max_x = citibike_dataset.stations["x_centered"].max()
        min_y = citibike_dataset.stations["y_centered"].min()
        max_y = citibike_dataset.stations["y_centered"].max()
   
        self.df = self.df[(self.df["x_centered"] >= min_x) & 
                          (self.df["x_centered"] <= max_x) & 
                          (self.df["y_centered"] >= min_y) & 
                          (self.df["y_centered"] <= max_y)
                        ]

    def spatio_temporal_rasterization(self, bins=100, time_bin_size=15):
        """
        Aggregates the crash data into a spatio-temporal grid (raster) and returns a new DataFrame that can be used to fit crash models.
        
        Arguments:
            bins (int): Number of spatial bins along each axis (x and y)
            time_bin_size (int): Size of each temporal bin in minutes
        
        Returns:
            DataFrame: A new DataFrame with the following columns:
                - 'x_center': Center x-coordinate of the spatial bin
                - 'y_center': Center y-coordinate of the spatial bin
                - 'time_center': Center time of the temporal bin (in minutes since midnight)
                - 'crash_count': Number of crashes in that spatio-temporal bin
        """

        df_temp = self.df.copy()
        
        def time_to_minutes(time_str):
            hour, minute = map(int, time_str.split(':'))
            return hour * 60 + minute

        df_temp['time_numeric'] = df_temp['CRASH TIME'].apply(time_to_minutes)
        
        min_x = df_temp['x_centered'].min()
        max_x = df_temp['x_centered'].max()
        min_y = df_temp['y_centered'].min()
        max_y = df_temp['y_centered'].max()

        x_bin_size = (max_x - min_x) / bins
        y_bin_size = (max_y - min_y) / bins
        
        df_temp['x_bin'] = ((df_temp['x_centered'] - min_x) // x_bin_size).astype(int)
        df_temp['y_bin'] = ((df_temp['y_centered'] - min_y) // y_bin_size).astype(int)

        df_temp['time_bin'] = (df_temp['time_numeric'] // time_bin_size).astype(int)
        
        df_raster = df_temp.groupby(['x_bin', 'y_bin', 'time_bin']).size().reset_index(name='crash_count')
        
        df_raster['x_center'] = min_x + (df_raster['x_bin'] + 0.5) * x_bin_size
        df_raster['y_center'] = min_y + (df_raster['y_bin'] + 0.5) * y_bin_size
        df_raster['time_center'] = df_raster['time_bin'] * time_bin_size + time_bin_size / 2
        
        return df_raster[['x_center', 'y_center', 'time_center', 'crash_count']]

