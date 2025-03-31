import numpy as np
from datetime import datetime
import pickle

class PriceCalculator:
    """
    Calculator to estimate insurance prices for individual rides based on crash risk.
    
    The risk per ride is computed by dividing the predicted crash count by the number of rides
    starting at the concidered station in that considered time bin.
    """
    
    def __init__(self, model_path, citibike_dataset, time_bin_size=30, cost_per_accident=5000, traffic_adjustment=0.001):
        """
        Initializes the CrashRiskCalculator.
        
        Arguments:
            model_path (str): Path to the saved model (e.g., "fitted_models/my_model.pkl")
            citibike_dataset (CitibikeDataset): Dataset to calculate traffic 
            time_bin_size (int): Size of the time bin in minutes (default is 30)
            cost_per_accident (float): Fixed estimated average cost per crash (default is 5000)
            traffic_adjustment (float): Tuneable factor to account for mismatch of crash counts and traffic and for varying traffic numbers due to dataset size. 
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.citibike_dataset = citibike_dataset
        self.time_bin_size = time_bin_size
        self.cost_per_accident = cost_per_accident
        self.traffic_adjustment = traffic_adjustment
    
    def convert_time_to_minutes(self, dt):
        """
        Converts a datetime object to minutes since midnight.
        
        Arguments:
            dt (datetime): A datetime object
            
        Returns:
            int: Minutes since midnight
        """
        return dt.hour * 60 + dt.minute
    
    def get_time_bin_center(self, minutes):
        """
        Given a time in minutes, computes the center of the corresponding time bin.
        
        Arguments:
            minutes (int): Time in minutes since midnight
            
        Returns:
            float: The center of the time bin (in minutes)
        """

        time_bin = minutes // self.time_bin_size

        return time_bin * self.time_bin_size + self.time_bin_size / 2
    
    def predict_insurance_price(self, started_at, start_station_id):
        """
        Predicts the insurance price for a single ride.
        
        Arguments:
            started_at (datetime): The ride's start time
            start_station_id (str): Citibike id of the start station
            
        Returns:
            tuple: A tuple (insurance_price, risk_per_ride) where:
                - insurance_price (float): The calculated insurance price
                - risk_per_ride (float): The predicted risk per ride (crashes per start)
        """

        minutes = self.convert_time_to_minutes(started_at)
        time_center = self.get_time_bin_center(minutes)
        
        station_row = self.citibike_dataset.stations[self.citibike_dataset.stations["station_id"] == start_station_id]
        x_centered = station_row['x_centered'].iloc[0]
        y_centered = station_row['y_centered'].iloc[0]
        X = np.array([[x_centered, y_centered, time_center]])

        predicted_crash_count = self.model.predict(X)[0]

        station_id = station_row['station_id']
        bin_index = minutes // self.time_bin_size

        def time_bin(dt):
            return (dt.hour * 60 + dt.minute) // self.time_bin_size

        rides_station = self.citibike_dataset.df_rides[self.citibike_dataset.df_rides['start_station_id'] == start_station_id]
        num_starts = rides_station['started_at'].apply(time_bin).eq(bin_index).sum()

        rides_station = self.citibike_dataset.df_rides[self.citibike_dataset.df_rides['end_station_id'] == start_station_id]
        num_ends = rides_station['ended_at'].apply(time_bin).eq(bin_index).sum()

        traffic = num_starts + num_ends

        if traffic > 0:
            risk_per_ride = predicted_crash_count / traffic
        else:
            risk_per_ride = predicted_crash_count
        
        insurance_price = risk_per_ride * self.cost_per_accident * self.traffic_adjustment

        return insurance_price, risk_per_ride