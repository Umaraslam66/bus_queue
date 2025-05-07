import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_arrival_times():
    # Parameters
    total_buses = 150
    start_time = datetime.strptime('07:00', '%H:%M')
    end_time = datetime.strptime('09:00', '%H:%M')
    total_minutes = (end_time - start_time).total_seconds() / 60
    
    # Generate arrival times with a bimodal distribution
    # Higher frequency around 7:30 and 8:30
    times = []
    
    # First peak (7:00-7:45)
    peak1_buses = int(total_buses * 0.4)  # 40% of buses
    peak1_times = np.random.normal(30, 15, peak1_buses)  # Mean at 30 minutes, std dev 15
    peak1_times = np.clip(peak1_times, 0, 45)  # Clip to 0-45 minutes
    
    # Second peak (7:45-9:00)
    peak2_buses = total_buses - peak1_buses
    peak2_times = np.random.normal(90, 20, peak2_buses)  # Mean at 90 minutes, std dev 20
    peak2_times = np.clip(peak2_times, 45, 120)  # Clip to 45-120 minutes
    
    # Combine and sort
    all_times = np.concatenate([peak1_times, peak2_times])
    all_times.sort()
    
    # Convert to datetime objects
    arrival_times = [start_time + timedelta(minutes=float(t)) for t in all_times]
    
    # Create DataFrame
    df = pd.DataFrame({
        'arrival_time': [t.strftime('%H:%M:%S') for t in arrival_times],
        'minutes_from_start': all_times
    })
    
    # Save to CSV
    df.to_csv('arrival_distribution.csv', index=False)
    print(f"Generated {len(arrival_times)} arrival times and saved to arrival_distribution.csv")
    
    # Print some statistics
    print("\nArrival Distribution Statistics:")
    print(f"First bus arrives at: {arrival_times[0].strftime('%H:%M:%S')}")
    print(f"Last bus arrives at: {arrival_times[-1].strftime('%H:%M:%S')}")
    print(f"Average interval between buses: {total_minutes/len(arrival_times):.2f} minutes")
    
    # Count buses in 15-minute intervals
    intervals = pd.cut(df['minutes_from_start'], 
                      bins=range(0, int(total_minutes)+15, 15),
                      labels=[f"{start_time + timedelta(minutes=i)}" for i in range(0, int(total_minutes), 15)])
    interval_counts = intervals.value_counts().sort_index()
    
    print("\nBuses per 15-minute interval:")
    for interval, count in interval_counts.items():
        print(f"{interval}: {count} buses")

if __name__ == "__main__":
    generate_arrival_times() 