import streamlit as st
import pandas as pd
import numpy as np
from main import BusSimulation, generate_test1_arrivals
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_test1_arrivals_custom(intervals):
    """Generate arrival times based on custom 5-minute interval distribution."""
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

def generate_arrival_times(total_buses, start_time_str, end_time_str, distribution_type, custom_params=None):
    if distribution_type == "Test 1 - Specific Distribution":
        if custom_params:
            return generate_test1_arrivals_custom(custom_params)
        return generate_test1_arrivals()
        
    start_time = datetime.strptime(start_time_str, '%H:%M')
    end_time = datetime.strptime(end_time_str, '%H:%M')
    total_minutes = (end_time - start_time).total_seconds() / 60
    
    if distribution_type == "Random":
        times = sorted(np.random.uniform(0, total_minutes, total_buses))
    elif distribution_type == "Fixed Rate":
        rate = custom_params.get('rate', 2) if custom_params else 2
        times = [i * (1/rate) for i in range(total_buses)]
    else:  # Bimodal
        # Get custom parameters or use defaults
        peak1_ratio = custom_params.get('peak1_ratio', 0.4) if custom_params else 0.4
        peak1_time_ratio = custom_params.get('peak1_time_ratio', 0.3) if custom_params else 0.3
        peak2_time_ratio = custom_params.get('peak2_time_ratio', 0.7) if custom_params else 0.7
        std_dev = custom_params.get('std_dev', 0.1) if custom_params else 0.1
        
        # First peak
        peak1_buses = int(total_buses * peak1_ratio)
        peak1_time = total_minutes * peak1_time_ratio
        peak1_times = np.random.normal(peak1_time, total_minutes * std_dev, peak1_buses)
        
        # Second peak
        peak2_buses = total_buses - peak1_buses
        peak2_time = total_minutes * peak2_time_ratio
        peak2_times = np.random.normal(peak2_time, total_minutes * std_dev, peak2_buses)
        
        times = sorted(np.concatenate([peak1_times, peak2_times]))
        times = np.clip(times, 0, total_minutes)
    
    # Create DataFrame
    arrival_times = [start_time + timedelta(minutes=float(t)) for t in times]
    df = pd.DataFrame({
        'arrival_time': [t.strftime('%H:%M:%S') for t in arrival_times],
        'minutes_from_start': times
    })
    return df

def get_distribution_parameters(distribution_type):
    custom_params = {}
    
    if distribution_type == "Fixed Rate":
        st.sidebar.subheader("Fixed Rate Parameters")
        custom_params['rate'] = st.sidebar.number_input(
            "Buses per minute",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
    
    elif distribution_type == "Bimodal":
        st.sidebar.subheader("Bimodal Distribution Parameters")
        custom_params['peak1_ratio'] = st.sidebar.slider(
            "First peak ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.1
        )
        custom_params['peak1_time_ratio'] = st.sidebar.slider(
            "First peak time ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.1
        )
        custom_params['peak2_time_ratio'] = st.sidebar.slider(
            "Second peak time ratio",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.1
        )
        custom_params['std_dev'] = st.sidebar.slider(
            "Standard deviation ratio",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01
        )
    
    return custom_params

def get_test1_parameters():
    st.sidebar.subheader("Test 1 Distribution Parameters")
    
    # Create time slots from 07:00 to 08:55 in 5-minute intervals
    time_slots = [f"{h:02d}:{m:02d}" for h in range(7, 9) for m in range(0, 60, 5)]
    
    # Initialize default values
    default_intervals = {
        '07:00': 6, '07:05': 6, '07:10': 11, '07:15': 11,
        '07:20': 3, '07:25': 3, '07:30': 6, '07:35': 6,
        '07:40': 10, '07:45': 10, '07:50': 12, '07:55': 12,
        '08:00': 3, '08:05': 3, '08:10': 8, '08:15': 8,
        '08:20': 6, '08:25': 6, '08:30': 8, '08:35': 8,
        '08:40': 5, '08:45': 5, '08:50': 2, '08:55': 2
    }
    
    # Create input fields for each time slot
    intervals = {}
    for time_slot in time_slots:
        intervals[time_slot] = st.sidebar.number_input(
            f"Buses at {time_slot}",
            min_value=0,
            max_value=20,
            value=default_intervals.get(time_slot, 0),
            step=1
        )
    
    return intervals

def plot_comparison(alt1_metrics, alt2_metrics, test_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert time points to datetime
    start_time = datetime.strptime("07:00", "%H:%M")
    time_points = [start_time + timedelta(minutes=0.5 * i) 
                  for i in range(max(len(alt1_metrics['queue_length']), 
                                   len(alt2_metrics['queue_length'])))]
    
    # Queue length comparison
    ax1.plot(time_points[:len(alt1_metrics['queue_length'])], 
             alt1_metrics['queue_length'], label='Alternative 1')
    ax1.plot(time_points[:len(alt2_metrics['queue_length'])], 
             alt2_metrics['queue_length'], label='Alternative 2')
    ax1.set_title(f'Queue Length Over Time - {test_name}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Queue Length')
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Waiting time comparison
    ax2.plot(time_points[:len(alt1_metrics['waiting_times'])], 
             alt1_metrics['waiting_times'], label='Alternative 1')
    ax2.plot(time_points[:len(alt2_metrics['waiting_times'])], 
             alt2_metrics['waiting_times'], label='Alternative 2')
    ax2.set_title('Waiting Time Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Waiting Time (minutes)')
    ax2.legend()
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Dwelling buses comparison
    ax3.plot(time_points[:len(alt1_metrics['dwelling_buses'])], 
             alt1_metrics['dwelling_buses'], label='Alternative 1')
    ax3.plot(time_points[:len(alt2_metrics['dwelling_buses'])], 
             alt2_metrics['dwelling_buses'], label='Alternative 2')
    ax3.set_title('Number of Buses Dwelling Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Buses Dwelling')
    ax3.legend()
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Cumulative served buses comparison
    ax4.plot(time_points[:len(alt1_metrics['cumulative_served'])], 
             alt1_metrics['cumulative_served'], label='Alternative 1')
    ax4.plot(time_points[:len(alt2_metrics['cumulative_served'])], 
             alt2_metrics['cumulative_served'], label='Alternative 2')
    ax4.set_title('Cumulative Buses Served Over Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Buses Served')
    ax4.legend()
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    st.title("Bus Terminal Simulation Dashboard")
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # General parameters
    total_buses = st.sidebar.number_input("Total number of buses", 
                                        min_value=10, max_value=300, value=150)
    dwell_time = st.sidebar.number_input("Average dwell time (minutes)", 
                                        min_value=1, max_value=20, value=6)
    
    # Time window
    st.sidebar.subheader("Time Window")
    start_time = st.sidebar.time_input("Start time", datetime.strptime("07:00", "%H:%M"))
    end_time = st.sidebar.time_input("End time", datetime.strptime("09:00", "%H:%M"))
    
    # Distribution type
    distribution_type = st.sidebar.selectbox(
        "Arrival Distribution",
        ["Test 1 - Specific Distribution", "Bimodal", "Random", "Fixed Rate"]
    )
    
    # Get distribution-specific parameters
    custom_params = None
    if distribution_type == "Test 1 - Specific Distribution":
        custom_params = get_test1_parameters()
    else:
        custom_params = get_distribution_parameters(distribution_type)
    
    # Alternative 1 parameters
    st.sidebar.subheader("Alternative 1 Parameters")
    alt1_capacity = st.sidebar.number_input("Single Lane Capacity", 
                                          min_value=5, max_value=50, value=20)
    
    # Alternative 2 parameters
    st.sidebar.subheader("Alternative 2 Parameters")
    alt2_capacity = st.sidebar.number_input("Side Road Capacity", 
                                          min_value=5, max_value=30, value=10)
    
    # Generate data and run simulation when button is clicked
    if st.sidebar.button("Run Simulation"):
        # Generate arrival times
        df = generate_arrival_times(
            total_buses,
            start_time.strftime("%H:%M"),
            end_time.strftime("%H:%M"),
            distribution_type,
            custom_params
        )
        
        # Save to CSV for simulation
        df.to_csv("arrival_distribution.csv", index=False)
        
        # Initialize and run simulation
        sim = BusSimulation("config.json")
        sim.dwell_time = dwell_time
        sim.max_queue_alt1 = alt1_capacity
        sim.max_queue_alt2 = alt2_capacity
        
        arrival_times = df['minutes_from_start'].values
        
        # Run simulations
        with st.spinner("Running simulation..."):
            alt1_metrics = sim.simulate_alternative1(arrival_times.copy())
            alt2_metrics = sim.simulate_alternative2(arrival_times.copy())
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alternative 1 Results")
            st.write(f"Total time: {alt1_metrics['total_time']:.1f} minutes")
            st.write(f"Buses served: {alt1_metrics['buses_served']}")
            st.write(f"Maximum queue length: {max(alt1_metrics['queue_length'])}")
            st.write(f"Average waiting time: {np.mean(alt1_metrics['waiting_times']):.1f} minutes")
        
        with col2:
            st.subheader("Alternative 2 Results")
            st.write(f"Total time: {alt2_metrics['total_time']:.1f} minutes")
            st.write(f"Buses served: {alt2_metrics['buses_served']}")
            st.write(f"Maximum queue length: {max(alt2_metrics['queue_length'])}")
            st.write(f"Average waiting time: {np.mean(alt2_metrics['waiting_times']):.1f} minutes")
        
        # Plot comparison
        st.subheader("Comparison Plots")
        fig = plot_comparison(alt1_metrics, alt2_metrics, distribution_type)
        st.pyplot(fig)
        
        # Display efficiency metrics
        st.subheader("Efficiency Comparison")
        efficiency_data = {
            'Metric': ['Total Time (min)', 'Buses Served', 'Max Queue Length', 'Avg Wait Time (min)'],
            'Alternative 1': [
                f"{alt1_metrics['total_time']:.1f}",
                alt1_metrics['buses_served'],
                max(alt1_metrics['queue_length']),
                f"{np.mean(alt1_metrics['waiting_times']):.1f}"
            ],
            'Alternative 2': [
                f"{alt2_metrics['total_time']:.1f}",
                alt2_metrics['buses_served'],
                max(alt2_metrics['queue_length']),
                f"{np.mean(alt2_metrics['waiting_times']):.1f}"
            ]
        }
        st.table(pd.DataFrame(efficiency_data))
        
        # Display arrival distribution
        st.subheader("Arrival Distribution")
        arrival_hist = plt.figure(figsize=(10, 4))
        plt.hist(arrival_times, bins=30)
        plt.title(f"Bus Arrival Distribution ({distribution_type})")
        plt.xlabel("Minutes from Start")
        plt.ylabel("Number of Buses")
        st.pyplot(arrival_hist)

if __name__ == "__main__":
    main() 