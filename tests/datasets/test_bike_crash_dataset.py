import unittest
import pandas as pd
import os
import tempfile
import sys
from datasets.bike_crash_dataset import BikeCrashDataset


class TestBikeCrashDataset(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file for testing.
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

    def tearDown(self):
        # Remove the temporary file after each test.
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def test_has_required_columns(self):
        # Test with a CSV that already has all required columns.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01'],
            'CRASH TIME': ['12:00'],
            'LATITUDE': [40.7128],
            'LONGITUDE': [-74.0060],
            'CRASH_DATETIME': ['2023-01-01 12:00'],
            'x': [1000],
            'y': [2000]
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        self.assertTrue(dataset._has_required_columns())
        # Check that the dataframe contains all required columns.
        for col in BikeCrashDataset.REQUIRED_COLUMNS:
            self.assertIn(col, dataset.df.columns)

    def test_has_required_columns_false(self):
        # Create a DataFrame missing some required columns.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01'],
            'CRASH TIME': ['12:00'],
            'SomeOtherColumn': [123]
        })

        df_init = pd.DataFrame({
            'CRASH DATE': ['2023-01-01'],
            'CRASH TIME': ['12:00'],
            'LATITUDE': [40.7128],
            'LONGITUDE': [-74.0060],
            'CRASH_DATETIME': ['2023-01-01 12:00'],
            'x': [1000],
            'y': [2000]
        })
        df_init.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)

        dataset.df = df
        # _has_required_columns should return False
        self.assertFalse(dataset._has_required_columns())

    def test_process_dataset(self):
        # Test with a CSV missing some required columns.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', '2023-01-02'],
            'CRASH TIME': ['08:00', '14:00'],
            'LATITUDE': [40.7128, 40.7138],
            'LONGITUDE': [-74.0060, -74.0050],
            'NUMBER OF CYCLIST INJURED': [1, 0],
            'VEHICLE TYPE CODE 1': ['bike', 'car']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        # After processing, the required columns should be present.
        for col in BikeCrashDataset.REQUIRED_COLUMNS:
            self.assertIn(col, dataset.df.columns)
        # Also check that CRASH_DATETIME is a datetime type.
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(dataset.df['CRASH_DATETIME']))

    def test_process_dataset_cyclist_filter(self):
        # Create a dataset with some entries with cyclist involvement and some without.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'CRASH TIME': ['10:00', '10:05', '10:10'],
            'LATITUDE': [40.7128, 40.7128, 40.7128],
            'LONGITUDE': [-74.0060, -74.0060, -74.0060],
            'NUMBER OF CYCLIST INJURED': [0, 1, 0],
            'NUMBER OF CYCLIST KILLED': [0, 0, 0],
            'VEHICLE TYPE CODE 1': ['car', 'bike', 'truck']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        # Only the second row should be kept (cyclist involved).
        self.assertEqual(len(dataset.df), 1)

        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'CRASH TIME': ['10:00', '10:05', '10:10'],
            'LATITUDE': [40.7128, 40.7128, 40.7128],
            'LONGITUDE': [-74.0060, -74.0060, -74.0060],
            'NUMBER OF CYCLIST INJURED': [0, 1, 1],
            'NUMBER OF CYCLIST KILLED': [0, 0, 0],
            'VEHICLE TYPE CODE 1': ['car', 'bike', 'truck']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        # Only the second row should be kept (cyclist involved).
        self.assertEqual(len(dataset.df), 2)

    def test_process_dataset_essential_values(self):
        # Create dataset with missing essential values and zeros in LATITUDE/LONGITUDE.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', None, '2023-01-01'],
            'CRASH TIME': ['10:00', '10:05', None],
            'LATITUDE': [40.7128, 0, 40.7128],
            'LONGITUDE': [-74.0060, -74.0060, 0],
            'NUMBER OF CYCLIST INJURED': [1, 1, 1],
            'VEHICLE TYPE CODE 1': ['bike', 'bike', 'bike']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        # Check that no essential column has missing values and that LATITUDE and LONGITUDE are non-zero.
        self.assertTrue(dataset.df[['CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE']].notna().all().all())
        self.assertTrue((dataset.df['LATITUDE'] != 0).all())
        self.assertTrue((dataset.df['LONGITUDE'] != 0).all())

    def test_citibike_alignment(self):
        # Create a simple CSV to simulate crash data.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', None, '2023-01-01'],
            'CRASH TIME': ['10:00', '10:05', None],
            'LATITUDE': [40.7128, 0, 40.7128],
            'LONGITUDE': [-74.0060, -74.0060, 0],
            'NUMBER OF CYCLIST INJURED': [1, 1, 1],
            'VEHICLE TYPE CODE 1': ['bike', 'bike', 'bike']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        
        # Create a dummy citibike_dataset object with necessary attributes.
        class DummyCitibike:
            pass
        dummy = DummyCitibike()
        dummy.x_center = 1000
        dummy.y_center = 2000
        dummy.stations = pd.DataFrame({
            'station_id': [1, 2],
            'x_centered': [50, 150],
            'y_centered': [60, 160]
        })
        dataset.citibike_alignment(dummy)
        
        # Check that all x_centered and y_centered values fall within the bounds from dummy.stations.
        min_x = dummy.stations['x_centered'].min()
        max_x = dummy.stations['x_centered'].max()
        min_y = dummy.stations['y_centered'].min()
        max_y = dummy.stations['y_centered'].max()
        self.assertTrue(((dataset.df['x_centered'] >= min_x) & (dataset.df['x_centered'] <= max_x)).all())
        self.assertTrue(((dataset.df['y_centered'] >= min_y) & (dataset.df['y_centered'] <= max_y)).all())

        # Check that for every row: x_centered == x - dummy.x_center and similarly for y.
        pd.testing.assert_series_equal(
            dataset.df['x_centered'],
            dataset.df['x'] - dummy.x_center,
            check_names=False
        )
        pd.testing.assert_series_equal(
            dataset.df['y_centered'],
            dataset.df['y'] - dummy.y_center,
            check_names=False
        )

    def test_get_spatio_temporal_rasterization(self):
        # Create a simple CSV that simulates crash data.
        df = pd.DataFrame({
            'CRASH DATE': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'CRASH TIME': ['10:00', '10:05', '10:10'],
            'LATITUDE': [40.7128, 42.7128, 41.7128],
            'LONGITUDE': [-74.0060, -76.0060, -75.0060],
            'NUMBER OF CYCLIST INJURED': [0, 1, 1],
            'NUMBER OF CYCLIST KILLED': [0, 0, 0],
            'VEHICLE TYPE CODE 1': ['bike', 'bike', 'truck']
        })
        df.to_csv(self.temp_file.name, index=False)
        dataset = BikeCrashDataset(self.temp_file.name)
        # Simulate that the dataset is already aligned by adding x_centered and y_centered.
        dataset.df['x_centered'] = dataset.df['x'] - 1000  # arbitrary center
        dataset.df['y_centered'] = dataset.df['y'] - 2000
        raster_df = dataset.get_spatio_temporal_rasterization(bins=10, time_bin_size=15)
        expected_cols = ['x_center', 'y_center', 'time_center', 'crash_count']
        for col in expected_cols:
            self.assertIn(col, raster_df.columns)
        # Ensure that crash_count is integer and non-negative.
        self.assertTrue((raster_df['crash_count'] >= 0).all())
        self.assertTrue(pd.api.types.is_integer_dtype(raster_df['crash_count']))
        self.assertEqual(raster_df['crash_count'].sum(), 3)

if __name__ == '__main__':
    unittest.main()
