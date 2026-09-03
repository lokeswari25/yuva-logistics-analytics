import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticsVisualEngine:
    """Enterprise Pipeline for EDA, Statistical Analytics and Logistics Plots."""
    def __init__(self, size=100):
        np.random.seed(42)
        # 1. Simulating Week 3 Cleansed Logistics Logs
        weight = np.random.uniform(50, 1500, size=size)
        distance = np.random.uniform(10, 500, size=size)
        
        # Adding real operational rules: longer distances = longer times & fuel costs
        transit_time = (distance / 45) + (weight * 0.003) + np.random.normal(0, 1, size=size)
        transit_time = np.clip(transit_time, 1, None) # Floor bound at 1 hour
        cost_inr = (distance * 35) + (weight * 15) + np.random.normal(0, 300, size=size)
        
        priority = np.random.choice(['Low-SLA', 'Medium-SLA', 'High-SLA'], size=size, p=[0.4, 0.4, 0.2])
        
        self.df = pd.DataFrame({
            'Tracking_ID': range(300001, 300001 + size),
            'Shipment_Weight_Kg': weight,
            'Delivery_Distance_Miles': distance,
            'Transit_Time_Hours': transit_time,
            'Transportation_Cost_INR': cost_inr,
            'SLA_Priority': priority
        })

    def run_exploratory_analysis(self):
        print("=== 📊 STEP 1: CALCULATING CENTRAL TENDENCY MARKS ===")
        # Calculate Mean, Median (50%), and Standard Deviation
        summary = self.df[['Shipment_Weight_Kg', 'Delivery_Distance_Miles', 'Transit_Time_Hours', 'Transportation_Cost_INR']].describe().T[['mean', '50%', 'std']]
        summary.columns = ['Arithmetic_Mean', 'Statistical_Median', 'Standard_Deviation']
        print(summary.to_string())
        
        print("\n=== 📈 STEP 2: GENERATING SYSTEM VISUALIZATIONS ===")
        sns.set_theme(style="whitegrid")
        
        # Create a beautiful composite chart grid containing your delivery insights
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Chart A: Distribution Profile of Transportation Spendings
        sns.histplot(data=self.df, x='Transportation_Cost_INR', kde=True, color='#1B365D', ax=axes[0])
        axes[0].set_title('Distribution of Transportation Costs (INR)', fontweight='bold')
        axes[0].set_xlabel('Cost Metrics (INR)')
        
        # Chart B: Strategic Multi-Variable Bottleneck Evaluation Scatter
        sns.scatterplot(data=self.df, x='Delivery_Distance_Miles', y='Transit_Time_Hours', 
                        hue='SLA_Priority', palette='dark', size='Shipment_Weight_Kg', sizes=(20, 200), alpha=0.8, ax=axes[1])
        axes[1].set_title('Transit Durations Across Logistics Distance Channels', fontweight='bold')
        axes[1].set_xlabel('Trip Distance (Miles)')
        axes[1].set_ylabel('Total Transit Duration (Hours)')
        
        plt.tight_layout()
        print("[SUCCESS] Operational logistics plots mapped to visual canvases.")
        plt.close()

if __name__ == "__main__":
    # Fire up Week 3 Run!
    engine = LogisticsVisualEngine()
    engine.run_exploratory_analysis()
