import unittest
import os
import tempfile
import pandas as pd
from datasets.citibike_dataset import CitibikeDataset

class TestCitibikeDataset(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file with minimal Citibike ride data.
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df = pd.DataFrame({
            'ride_id': ['1', '2', '3'],
            'rideable_type': ['electric_bike', 'classic_bike', 'classic_bike'],
            'started_at': ['2023-01-01 08:00:00', '2023-01-01 09:00:00', '2023-01-01 10:00:00'],
            'ended_at': ['2023-01-01 08:15:00', '2023-01-01 09:20:00', '2023-01-01 10:20:00'],
            'start_station_name': ['Station A', 'Station B', 'Station C'],
            'start_station_id': ['A1', 'B1', 'C1'],
            'end_station_name': ['Station A', 'Station B', 'Station C'],
            'end_station_id': ['A1', 'B1', None],  # One row has missing end_station_id
            'start_lat': [40.7128, 0, 40.7148],    # One row has invalid latitude (0)
            'start_lng': [-74.0060, -74.0050, -74.0030],
            'end_lat': [40.7128, 0, 40.7148],
            'end_lng': [-74.0060, -74.0050, -74.0030],
            'member_casual': ['member', 'casual', 'member']
        })
        df.to_csv(self.temp_file.name, index=False)

    def tearDown(self):
        # Close and remove the temporary file.
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def test_file_not_found(self):
        # Test that a non-existent file raises FileNotFoundError.
        with self.assertRaises(FileNotFoundError):
            CitibikeDataset("non_existent_file.csv")

    def test_unsupported_file_format(self):
        # Test that providing an unsupported file format (e.g., .txt) raises ValueError.
        temp_txt = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        with open(temp_txt.name, 'w') as f:
            f.write("Not a CSV file")
        temp_txt.close()
        with self.assertRaises(ValueError):
            CitibikeDataset(temp_txt.name)
        os.unlink(temp_txt.name)

    def test_load_csv(self):
        # Test that a proper CSV file is loaded correctly.
        dataset = CitibikeDataset(self.temp_file.name)
        self.assertIsNotNone(dataset.df_rides)
        # Check that required ride columns are present.
        required_cols = [
            'ride_id', 'rideable_type', 'started_at', 'ended_at',
            'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id',
            'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual'
        ]
        for col in required_cols:
            self.assertIn(col, dataset.df_rides.columns)

    def test_no_empty_values_and_dropped_rows(self):
        # Ensure that df_rides and stations contain no NaN values,
        # and that the number of dropped rows equals the difference between the original CSV rows and df_rides.
        dataset = CitibikeDataset(self.temp_file.name)
        self.assertFalse(dataset.df_rides.isnull().values.any(), "df_rides contains NaN values.")
        self.assertFalse(dataset.stations.isnull().values.any(), "stations contains NaN values.")
        original_df = pd.read_csv(self.temp_file.name)
        self.assertEqual(len(dataset.dropped_rows), 1, "There should be one dropped rows.")
        self.assertEqual(len(dataset.df_rides) + len(dataset.dropped_rows), len(original_df), "Number of rides should equal the number of original rows + number of dropped rows.")

    def test_process_stations(self):
        # Test that _process_stations creates a stations DataFrame with expected columns.
        dataset = CitibikeDataset(self.temp_file.name)
        self.assertIsNotNone(dataset.df_rides)
        self.assertIsNotNone(dataset.stations)
        expected_station_cols = ['station_id', 'station_name', 'lat', 'lng', 
                                 'start_count', 'end_count', 'x', 'y', 
                                 'x_centered', 'y_centered']
        for col in expected_station_cols:
            self.assertIn(col, dataset.stations.columns)

    def test_duration_calculation(self):
        # Test that ride duration and normalized ride duration are computed.
        dataset = CitibikeDataset(self.temp_file.name)
        df_rides = dataset.df_rides
        self.assertIn('ride_duration', df_rides.columns)
        self.assertIn('ride_duration_normalized', df_rides.columns)
        # For the first ride: 08:00 to 08:15 should be about 900 seconds.
        self.assertAlmostEqual(df_rides.loc[0, 'ride_duration'], 900, delta=1)

    def test_duration_statistics(self):
        # For our test CSV, ride 1 is from 08:00 to 08:15 (900 seconds) and ride 2 is from 09:00 to 09:20 (1200 seconds).
        # Check that duration_mean and duration_std are computed correctly.
        dataset = CitibikeDataset(self.temp_file.name)
        expected_mean = (900 + 1200) / 2
        expected_std = pd.Series([900, 1200]).std()
        self.assertAlmostEqual(dataset.duration_mean, expected_mean, delta=1, msg="Duration mean is incorrect.")
        self.assertAlmostEqual(dataset.duration_std, expected_std, delta=1, msg="Duration std is incorrect.")

    def test_station_x_y_center_and_centered_values(self):
        # Test that x_center and y_center exist, are computed correctly, and that each station's x_centered and y_centered
        # equal x - x_center and y - y_center respectively.
        dataset = CitibikeDataset(self.temp_file.name)
        computed_x_center = dataset.stations['x'].mean()
        computed_y_center = dataset.stations['y'].mean()
        self.assertAlmostEqual(dataset.x_center, computed_x_center, delta=1, msg="x_center not computed correctly.")
        self.assertAlmostEqual(dataset.y_center, computed_y_center, delta=1, msg="y_center not computed correctly.")
        # For each station, verify the centered values.
        for _, row in dataset.stations.iterrows():
            self.assertAlmostEqual(row['x_centered'], row['x'] - dataset.x_center, delta=1, msg=f"x_centered incorrect for station {row['station_id']}.")
            self.assertAlmostEqual(row['y_centered'], row['y'] - dataset.y_center, delta=1, msg=f"y_centered incorrect for station {row['station_id']}.")

if __name__ == '__main__':
    unittest.main()
