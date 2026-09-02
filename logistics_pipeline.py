import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

class LogisticsDataPipeline:
    """Handles Phase 1 & 2: Ingestion, schema verification, and chronological data cleaning."""
    @staticmethod
    def generate_and_clean_data(size=500):
        np.random.seed(42)
        raw_data = {
            'Order_ID': range(10001, 10001 + size),
            'Sales_Value_INR': np.random.randint(500, 7500, size=size),
            'Delivery_Status': np.random.choice(['On-Time', 'Delayed', None], size=size, p=[0.82, 0.14, 0.04]),
            'Customer_Lat': np.random.uniform(12.90, 13.10, size=size),
            'Customer_Lon': np.random.uniform(77.50, 77.70, size=size),
            'Historical_Demand': np.random.randint(20, 150, size=size)
        }
        df = pd.DataFrame(raw_data)
        # Handle duplicates and perform temporal chronological forward-fill imputation
        df = df.drop_duplicates(subset=['Order_ID'])
        df['Delivery_Status'] = df['Delivery_Status'].ffill().fillna('On-Time')
        return df

class SpatialTerritoryOptimizer:
    """Handles Phase 4: Machine Learning Geospatial Route Partitioning."""
    def __init__(self, zones=3):
        self.zones = zones
        self.model = KMeans(n_clusters=zones, random_state=42, n_init='auto')
        
    def partition_routes(self, df):
        coords = df[['Customer_Lat', 'Customer_Lon']]
        df['Delivery_Zone_ID'] = self.model.fit_predict(coords)
        return df, self.model.cluster_centers_

# Execute the complete core architecture verification run
if __name__ == "__main__":
    print("[INFO] Initializing Yuva Intern Corporate Logistics Pipeline Run...")
    df_raw = LogisticsDataPipeline.generate_and_clean_data(size=200)
    
    # Calculate initial KPIs
    otif = (df_raw['Delivery_Status'] == 'On-Time').sum() / len(df_raw) * 100
    print(f"[KPI METRIC] Baseline Pipeline Operational OTIF Rate: {otif:.2f}%")
    
    # Apply ML Clustering
    optimizer = SpatialTerritoryOptimizer(zones=4)
    df_optimized, hubs = optimizer.partition_routes(df_raw)
    print(f"[ML ENGINE] Successfully localized {len(hubs)} optimal distribution hubs.")
    print(df_optimized[['Order_ID', 'Delivery_Zone_ID']].head(5))
