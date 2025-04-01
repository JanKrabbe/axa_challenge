import unittest
import tempfile
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from modeling.price_calculator import PriceCalculator

# Dummy model that always predicts a crash count of 2.0.
class DummyModel:
    def predict(self, X):
        return np.array([2.0])

# Dummy CitibikeDataset with minimal required attributes.
class DummyCitibikeDataset:
    def __init__(self):
        # Stations DataFrame with one station.
        self.stations = pd.DataFrame({
            'station_id': ['A1'],
            'x_centered': [100.0],
            'y_centered': [200.0]
        })
        # Rides DataFrame with one ride that starts and ends at station A1.
        # The ride's times are chosen so that the time bin (with time_bin_size=30)
        # for both start and end fall into the same bin.
        self.df_rides = pd.DataFrame({
            'start_station_id': ['A1'],
            'started_at': [datetime(2023, 3, 1, 8, 5)],   # 8:05 -> 485 minutes, bin index=16
            'end_station_id': ['A1'],
            'ended_at': [datetime(2023, 3, 1, 8, 10)]       # 8:10 -> 490 minutes, bin index=16
        })

class TestPriceCalculator(unittest.TestCase):
    def setUp(self):
        # Create a temporary pickle file containing the dummy model.
        self.temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        dummy_model = DummyModel()
        with open(self.temp_model_file.name, 'wb') as f:
            pickle.dump(dummy_model, f)
        
        # Create a dummy CitibikeDataset instance.
        self.dummy_citibike = DummyCitibikeDataset()

    def tearDown(self):
        # Close the temporary file handle to release the lock on Windows.
        self.temp_model_file.close()
        os.unlink(self.temp_model_file.name)

    def test_predict_insurance_price(self):
        """
        Tests that predict_insurance_price() returns the expected insurance price and risk per ride.
        
        For a ride with start time 8:07, the bin index will be:
            8:07 -> 8*60+7 = 487 minutes, bin index = 487 // 30 = 16.
        The corresponding bin center is:
            16*30 + 15 = 495 minutes.
        The dummy model always predicts 2.0 crashes.
        The dummy rides DataFrame has one ride starting at A1 (bin 16) and one ride ending at A1 (bin 16),
        so total traffic = 2.
        Expected risk per ride = 2.0 / 2 = 1.0.
        Given cost_per_accident = 5000 and traffic_adjustment = 0.001, expected price = 1.0 * 5000 * 0.001 = 5.0.
        """
        calculator = PriceCalculator(
            model_path=self.temp_model_file.name,
            citibike_dataset=self.dummy_citibike,
            time_bin_size=30,
            cost_per_accident=5000,
            traffic_adjustment=0.001
        )
        
        ride_start = datetime(2023, 3, 1, 8, 7)  # 8:07 falls in the same bin as our dummy rides.
        insurance_price, risk_per_ride = calculator.predict_insurance_price(ride_start, "A1")
        
        self.assertAlmostEqual(risk_per_ride, 1.0, places=3)
        self.assertAlmostEqual(insurance_price, 5.0, places=3)

if __name__ == '__main__':
    unittest.main()
