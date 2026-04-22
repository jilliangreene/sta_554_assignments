import pandas as pd
import time
import os

# Load full dataset
df = pd.read_csv("power_streaming_data.csv")

output_dir = "powerStreaming"
os.makedirs(output_dir, exist_ok=True)

# Loop 20 times
for i in range(20):
    
    # Random sample of 5 rows
    sample = df.sample(n=5)
    
    # Write to CSV (no index)
    file_path = f"{output_dir}/batch_{i}.csv"
    sample.to_csv(file_path, index=False)
    
    print(f"Wrote {file_path}")
    
    # Wait 10 seconds
    time.sleep(10)