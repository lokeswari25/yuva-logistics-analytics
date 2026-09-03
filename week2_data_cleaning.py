import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LogisticsDataCleaner:
    """Production-grade ETL pipeline handling Week 2 data cleansing, anomaly 
    detection, null-imputation, and scaling validation for logistics telemetry.
    """
    def __init__(self, sample_size=1000):
        np.random.seed(42)
        # 1. Simulating Raw, Uncleaned Logistics Telemetry Data with Built-in Anomalies
        tracking_ids = range(500001, 500001 + sample_size)
        weights = np.random.uniform(10, 1500, size=sample_size)
        distances = np.random.uniform(5, 500, size=sample_size)
        
        # Inject intentional errors: Outliers (-999), Missing rows (NaN), and string case variances
        weights[np.random.choice(sample_size, 40, replace=False)] = np.nan
        distances[np.random.choice(sample_size, 20, replace=False)] = -999.0  # Systemic sensor error code
        
        regions = np.random.choice(['North', 'south', 'EAST', 'West', None], size=sample_size, p=[0.25, 0.30, 0.20, 0.20, 0.05])
        base_costs = (distances * 45) + (weights * 12) + np.random.normal(0, 800, size=sample_size)
        
        self.raw_df = pd.DataFrame({
            'Tracking_ID': tracking_ids,
            'Shipment_Weight_Kg': weights,
            'Delivery_Distance_Miles': distances,
            'Region_Zone': regions,
            'Transportation_Cost_INR': base_costs
        })

    def execute_preprocessing_pipeline(self):
        """Executes a 4-step structural data cleansing framework."""
        print("=== [PHASE 1] Initializing Telemetry Health Audit ===")
        print(f"Total Imported Shipments: {len(self.raw_df)}")
        print("Detected Null Cells Per Column:\n", self.raw_df.isnull().sum())
        
        cleaned_df = self.raw_df.copy()
        
        # Step 1: Standardize Categorical Formats and Repair Gaps
        cleaned_df['Region_Zone'] = cleaned_df['Region_Zone'].astype(str).str.upper().str.strip()
        cleaned_df['Region_Zone'] = cleaned_df['Region_Zone'].replace('NONE', np.nan)
        # Categorical forward-fill imputation to maintain sequence
        cleaned_df['Region_Zone'] = cleaned_df['Region_Zone'].ffill().fillna('SOUTH')
        
        # Step 2: Handle Outlier Sensor Telemetry codes (-999)
        cleaned_df['Delivery_Distance_Miles'] = cleaned_df['Delivery_Distance_Miles'].replace(-999.0, np.nan)
        
        # Step 3: Apply Statistical Median Imputation for Numeric Nulls
        weight_median = cleaned_df['Shipment_Weight_Kg'].median()
        distance_median = cleaned_df['Delivery_Distance_Miles'].median()
        
        cleaned_df['Shipment_Weight_Kg'] = cleaned_df['Shipment_Weight_Kg'].fillna(weight_median)
        cleaned_df['Delivery_Distance_Miles'] = cleaned_df['Delivery_Distance_Miles'].fillna(distance_median)
        
        # Recalculate cost bounds for any rows corrected to avoid math mismatches
        cleaned_df['Transportation_Cost_INR'] = (cleaned_df['Delivery_Distance_Miles'] * 45) + (cleaned_df['Shipment_Weight_Kg'] * 12)
        
        # Step 4: MinMax Normalization Feature Scaling for Machine Learning Models
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(cleaned_df[['Shipment_Weight_Kg', 'Delivery_Distance_Miles']])
        cleaned_df['Scaled_Weight'] = scaled_features[:, 0]
        cleaned_df['Scaled_Distance'] = scaled_features[:, 1]
        
        print("\n=== [PHASE 2] Post-Pipeline Quality Validation ===")
        print("Remaining Missing Rows:\n", cleaned_df.isnull().sum())
        print("\n=== [PHASE 3] Sample Cleansed Database Output ===")
        print(cleaned_df[['Tracking_ID', 'Shipment_Weight_Kg', 'Delivery_Distance_Miles', 'Region_Zone', 'Scaled_Weight']].head(5))
        
        return cleaned_df

if __name__ == "__main__":
    pipeline = LogisticsDataCleaner()
    cleansed_data = pipeline.execute_preprocessing_pipeline()
