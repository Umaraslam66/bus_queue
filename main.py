import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_test1_arrivals():
    """Generate arrival times based on the specific 5-minute interval distribution."""
    intervals = {
        '07:00': 6, '07:05': 6, '07:10': 11, '07:15': 11,
        '07:20': 3, '07:25': 3, '07:30': 6, '07:35': 6,
        '07:40': 10, '07:45': 10, '07:50': 12, '07:55': 12,
        '08:00': 3, '08:05': 3, '08:10': 8, '08:15': 8,
        '08:20': 6, '08:25': 6, '08:30': 8, '08:35': 8,
        '08:40': 5, '08:45': 5, '08:50': 2, '08:55': 2
    }
    
    arrival_times = []
    for time_str, num_buses in intervals.items():
        base_time = datetime.strptime(time_str, '%H:%M')
        for _ in range(num_buses):
            # Add random seconds within the 5-minute interval
            random_seconds = np.random.uniform(0, 300)
            arrival_time = base_time + timedelta(seconds=random_seconds)
            arrival_times.append(arrival_time)
    
    # Sort arrival times
    arrival_times.sort()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'arrival_time': [t.strftime('%H:%M:%S') for t in arrival_times],
        'minutes_from_start': [(t - datetime.strptime('07:00', '%H:%M')).total_seconds() / 60 
                              for t in arrival_times]
    })
    return df

class BusSimulation:
    def __init__(self, config_path):
        self.config = json.load(open(config_path))
        self.arrival_data = pd.read_csv(self.config['data_path'])
        
        # Convert arrival times to minutes from start
        self.arrival_data['arrival_time'] = pd.to_datetime(self.arrival_data['arrival_time'])
        
        # Simulation parameters
        self.dwell_time = 6  # minutes
        self.max_queue_alt1 = 20  # Maximum queue length
        self.max_unloading_alt1 = 20  # Maximum number of buses that can unload simultaneously
        self.max_queue_alt2 = 10
        self.max_unloading_alt2 = 10
        
        # Metrics
        self.reset_metrics()
    
    def reset_metrics(self):
        self.metrics = {
            'queue_length': [],
            'buses_served': 0,
            'waiting_times': [],
            'total_time': 0,
            'dwelling_buses': [],  # New metric for buses currently dwelling
            'cumulative_served': []  # New metric for cumulative buses served
        }
    
    def simulate_alternative1(self, arrival_times):
        """One lane road with space for 20 buses, no overtaking"""
        self.reset_metrics()
        queue = []
        unloading = []  # List of (start_time, end_time) tuples
        current_time = 0
        arrival_idx = 0
        cumulative_served = 0
        
        while arrival_idx < len(arrival_times) or queue or unloading:
            # Process arrivals
            while arrival_idx < len(arrival_times) and arrival_times[arrival_idx] <= current_time:
                if len(queue) < self.max_queue_alt1:
                    queue.append(arrival_times[arrival_idx])
                arrival_idx += 1
            
            # Process unloading
            unloading = [(start, end) for start, end in unloading if end > current_time]
            
            # Start new unloading if possible
            while len(unloading) < self.max_unloading_alt1 and queue:
                unloading.append((current_time, current_time + self.dwell_time))
                queue.pop(0)
                self.metrics['buses_served'] += 1
                cumulative_served += 1
            
            # Record metrics
            self.metrics['queue_length'].append(len(queue))
            if queue:
                self.metrics['waiting_times'].append(current_time - queue[0])
            
            # Record dwelling buses (number of buses currently unloading)
            self.metrics['dwelling_buses'].append(len(unloading))
            
            # Record cumulative served buses
            self.metrics['cumulative_served'].append(cumulative_served)
            
            current_time += 0.5  # Half-minute time steps
            
        self.metrics['total_time'] = current_time
        return self.metrics
    
    def simulate_alternative2(self, arrival_times):
        """Road with space for 10 buses on side, allowing overtaking"""
        self.reset_metrics()
        queue = []
        unloading = []  # List of (start_time, end_time) tuples
        current_time = 0
        arrival_idx = 0
        cumulative_served = 0
        
        while arrival_idx < len(arrival_times) or queue or unloading:
            # Process arrivals
            while arrival_idx < len(arrival_times) and arrival_times[arrival_idx] <= current_time:
                if len(queue) < self.max_queue_alt2:
                    queue.append(arrival_times[arrival_idx])
                arrival_idx += 1
            
            # Process unloading
            unloading = [(start, end) for start, end in unloading if end > current_time]
            
            # Start new unloading if possible
            while len(unloading) < self.max_unloading_alt2 and queue:
                unloading.append((current_time, current_time + self.dwell_time))
                queue.pop(0)
                self.metrics['buses_served'] += 1
                cumulative_served += 1
            
            # Record metrics
            self.metrics['queue_length'].append(len(queue))
            if queue:
                self.metrics['waiting_times'].append(current_time - queue[0])
            
            # Record dwelling buses (number of buses currently unloading)
            self.metrics['dwelling_buses'].append(len(unloading))
            
            # Record cumulative served buses
            self.metrics['cumulative_served'].append(cumulative_served)
            
            current_time += 0.5  # Half-minute time steps
            
        self.metrics['total_time'] = current_time
        return self.metrics
    
    def plot_comparison(self, alt1_metrics, alt2_metrics, test_name):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Queue length comparison
        time_points = np.arange(0, max(len(alt1_metrics['queue_length']), 
                                     len(alt2_metrics['queue_length']))) * 0.5
        ax1.plot(time_points[:len(alt1_metrics['queue_length'])], 
                 alt1_metrics['queue_length'], label='Alternative 1')
        ax1.plot(time_points[:len(alt2_metrics['queue_length'])], 
                 alt2_metrics['queue_length'], label='Alternative 2')
        ax1.set_title(f'Queue Length Over Time - {test_name}')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Queue Length')
        ax1.legend()
        
        # Waiting time comparison
        ax2.plot(time_points[:len(alt1_metrics['waiting_times'])], 
                 alt1_metrics['waiting_times'], label='Alternative 1')
        ax2.plot(time_points[:len(alt2_metrics['waiting_times'])], 
                 alt2_metrics['waiting_times'], label='Alternative 2')
        ax2.set_title('Waiting Time Over Time')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Waiting Time (minutes)')
        ax2.legend()
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Generate Test 1 arrival distribution
    df = generate_test1_arrivals()
    df.to_csv("arrival_distribution.csv", index=False)
    
    sim = BusSimulation("config.json")
    arrival_times = df['minutes_from_start'].values
    
    # Run simulations
    print("Running Alternative 1 (Single Lane, 20 buses)...")
    alt1_metrics = sim.simulate_alternative1(arrival_times.copy())
    print(f"Total time: {alt1_metrics['total_time']:.1f} minutes")
    print(f"Buses served: {alt1_metrics['buses_served']}")
    print(f"Maximum queue length: {max(alt1_metrics['queue_length'])}")
    print(f"Average waiting time: {np.mean(alt1_metrics['waiting_times']):.1f} minutes")
    
    print("\nRunning Alternative 2 (Side Road, 10 buses)...")
    alt2_metrics = sim.simulate_alternative2(arrival_times.copy())
    print(f"Total time: {alt2_metrics['total_time']:.1f} minutes")
    print(f"Buses served: {alt2_metrics['buses_served']}")
    print(f"Maximum queue length: {max(alt2_metrics['queue_length'])}")
    print(f"Average waiting time: {np.mean(alt2_metrics['waiting_times']):.1f} minutes")
    
    # Plot comparison
    sim.plot_comparison(alt1_metrics, alt2_metrics, "Test 1 - Specific Distribution")