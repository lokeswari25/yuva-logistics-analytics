import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LogisticsPredictivePipeline:
    """Enterprise Pipeline for Week 4: Multi-Variate Target Simulation, 
    Categorical Feature Encoding, Machine Learning Training, and Operational Metrics Valuation.
    """
    def __init__(self, size=1000):
        np.random.seed(42)
        # 1. Generating High-Dimensional Enterprise Logistics Datasets
        weight = np.random.uniform(20, 2000, size=size)
        distance = np.random.uniform(10, 500, size=size)
        traffic = np.random.uniform(1.0, 5.0, size=size)
        priority = np.random.choice(['Low', 'Medium', 'High'], size=size, p=[0.4, 0.4, 0.2])
        
        # Build complex, non-linear system interactions simulating real road bottlenecks
        base_time = (distance / 45) + (traffic * 1.5) + (weight * 0.001)
        interaction_delay = (traffic ** 2) * (weight / 1000)
        transit_time = base_time + interaction_delay + np.random.normal(0, 0.8, size=size)
        
        self.df = pd.DataFrame({
            'Tracking_ID': range(400001, 400001 + size),
            'Shipment_Weight_Kg': weight,
            'Delivery_Distance_Miles': distance,
            'Traffic_Density_Index': traffic,
            'Priority_Level': priority,
            'Transit_Time_Hours': np.clip(transit_time, 0.5, None)
        })

    def preprocess_and_split(self):
        """Phase 2: Transforming categorical attributes into ordinal numeric features."""
        encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
        self.df['Priority_Encoded'] = encoder.fit_transform(self.df[['Priority_Level']])
        
        X = self.df[['Shipment_Weight_Kg', 'Delivery_Distance_Miles', 'Traffic_Density_Index', 'Priority_Encoded']]
        y = self.df['Transit_Time_Hours']
        
        # Chronological Split
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self):
        """Phase 3: Training model suite and extracting backtesting error statistics."""
        X_train, X_test, y_train, y_test = self.preprocess_and_split()
        
        models = {
            "Decision_Tree_Base": DecisionTreeRegressor(max_depth=6, random_state=42),
            "Random_Forest_Ensemble": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        }
        
        print("=== 🤖 PHASE 3: EVALUATING MACHINE LEARNING INFERENCE ===")
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"\\nModel Framework: {name}")
            print(f" -> Root Mean Squared Error (RMSE): {rmse:.2f} Hours")
            print(f" -> Mean Absolute Error (MAE): {mae:.2f} Hours")
            print(f" -> Coefficient of Determination (R2 Score): {r2:.3f}")

if __name__ == "__main__":
    pipeline = LogisticsPredictivePipeline()
    pipeline.train_and_evaluate()
