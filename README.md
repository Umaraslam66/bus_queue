# Bus Terminal Simulation System

This project simulates and compares two different bus terminal unloading configurations using a discrete-event simulation approach. The system models bus arrivals, queue management, and unloading processes to evaluate the efficiency of different terminal layouts.

## Overview

The simulation compares two alternative bus terminal configurations:

1. **Alternative 1**: Single lane with space for 20 buses
   - Buses cannot overtake each other
   - Sequential unloading process
   - Limited to one bus unloading at a time

2. **Alternative 2**: Side road with space for 10 buses
   - Buses can overtake unloading buses
   - Parallel unloading process
   - Multiple buses can unload simultaneously

## System Components

### 1. Core Simulation (`main.py`)

The main simulation logic is implemented in the `BusSimulation` class with the following key components:

#### Parameters
- `dwell_time`: Average time (in minutes) a bus spends unloading
- `max_queue_alt1`: Maximum queue length for Alternative 1 (default: 20)
- `max_queue_alt2`: Maximum queue length for Alternative 2 (default: 10)

#### Key Methods
- `simulate_alternative1()`: Implements the single-lane simulation
- `simulate_alternative2()`: Implements the side-road simulation
- `reset_metrics()`: Initializes simulation metrics
- `plot_comparison()`: Generates comparison visualizations

### 2. Interactive Dashboard (`app.py`)

The Streamlit dashboard provides an interactive interface for:
- Adjusting simulation parameters
- Selecting arrival distributions
- Viewing results and visualizations

#### Features
- Real-time parameter adjustment
- Multiple arrival distribution options
- Interactive visualizations
- Performance metrics comparison

## Simulation Logic

### 1. Time Management
- Simulation runs in discrete time steps (0.5-minute intervals)
- Events are processed in chronological order
- System state is updated at each time step

### 2. Bus Arrival Process
Three arrival distribution patterns are implemented:

1. **Bimodal Distribution**
   - Simulates morning rush hour patterns
   - Two peak periods (40% and 60% of buses)
   - Normal distribution around peak times
   - Parameters:
     - Peak 1: 30% of time window
     - Peak 2: 70% of time window
     - Standard deviation: 10% of time window

2. **Random Distribution**
   - Uniform distribution across time window
   - Simulates irregular arrival patterns
   - No specific clustering

3. **Fixed Rate**
   - Constant arrival rate (2 buses per minute)
   - Simulates scheduled service
   - Regular intervals between arrivals

### 3. Queue Management

#### Alternative 1 (Single Lane)
```python
# Key logic
if unloading is None and queue:
    unloading = current_time
    queue.pop(0)
```

- First-in-first-out (FIFO) queue
- One bus unloads at a time
- Queue limited to 20 buses
- No overtaking possible

#### Alternative 2 (Side Road)
```python
# Key logic
while len(unloading) < self.max_queue_alt2 and queue:
    unloading.append((current_time, current_time + self.dwell_time))
    queue.pop(0)
```

- Parallel unloading process
- Up to 10 buses can unload simultaneously
- Overtaking possible
- More efficient queue management

### 4. Performance Metrics

The simulation tracks several key performance indicators:

1. **Queue Length**
   - Number of buses waiting at each time step
   - Maximum queue length reached
   - Queue length over time

2. **Waiting Time**
   - Time each bus spends in queue
   - Average waiting time
   - Maximum waiting time

3. **Service Metrics**
   - Total simulation time
   - Number of buses served
   - System throughput

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

3. Adjust parameters in the sidebar:
   - Total number of buses
   - Dwell time
   - Time window
   - Arrival distribution
   - Queue capacities

4. Click "Run Simulation" to see results

## Configuration

The simulation can be configured through:
- `config.json`: Basic configuration file
- Dashboard parameters: Interactive adjustments
- Code parameters: Direct modification in source files

## Visualization

The system generates several visualizations:
1. Queue length over time
2. Waiting time over time
3. Arrival distribution histogram
4. Efficiency comparison table

## Performance Considerations

- Simulation uses discrete time steps (0.5 minutes)
- Memory efficient queue management
- Optimized for real-time parameter adjustment
- Scalable for different arrival patterns

## Future Improvements

Potential enhancements:
1. More arrival distribution patterns
2. Dynamic dwell times
3. Weather impact simulation
4. Cost-benefit analysis
5. 3D visualization of terminal layout

## Contributing

Feel free to contribute by:
1. Adding new features
2. Improving documentation
3. Optimizing performance
4. Adding more test scenarios

## License

This project is open source and available under the MIT License. 