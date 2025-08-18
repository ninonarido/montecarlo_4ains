# SIMULATION - 3AINS


## Python Simulation Tool Setup Guide

### Prerequisites

Before starting, ensure you have:
- A computer running Windows, macOS, or Linux
- Basic understanding of Python programming
- Administrator/sudo access for installations

### Step 1: Install Python

#### Windows
1. Visit [python.org](https://www.python.org/downloads/)
2. Download the latest Python 3.x version (3.9 or higher recommended)
3. Run the installer with "Add Python to PATH" checked
4. Verify installation by opening Command Prompt and typing:
   ```
   python --version
   ```

#### macOS
1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python
   ```
3. Verify installation:
   ```bash
   python3 --version
   ```

#### Linux (Ubuntu/Debian)
1. Update package list:
   ```bash
   sudo apt update
   ```
2. Install Python:
   ```bash
   sudo apt install python3 python3-pip python3-venv
   ```
3. Verify installation:
   ```bash
   python3 --version
   ```

### Step 2: Set Up Virtual Environment

Creating a virtual environment keeps your simulation project dependencies isolated.

1. Create a project directory:
   ```bash
   mkdir python-simulation
   cd python-simulation
   ```

2. Create virtual environment:
   ```bash
   # Windows
   python -m venv simulation_env
   
   # macOS/Linux
   python3 -m venv simulation_env
   ```

3. Activate virtual environment:
   ```bash
   # Windows
   simulation_env\Scripts\activate
   
   # macOS/Linux
   source simulation_env/bin/activate
   ```

You should see `(simulation_env)` in your terminal prompt when activated.

### Step 3: Install Essential Simulation Libraries

Install the core libraries needed for most simulation work:

```bash
pip install --upgrade pip
pip install numpy scipy matplotlib pandas jupyter
pip install simpy  # For discrete event simulation
pip install mesa   # For agent-based modeling
pip install networkx  # For network simulations
pip install plotly  # For interactive visualizations
```

For specific simulation types, also consider:
```bash
# Physics simulations
pip install pymunk pygame

# Monte Carlo simulations  
pip install random2

# Statistical analysis
pip install statsmodels seaborn

# Machine learning simulations
pip install scikit-learn
```

### Step 4: Set Up Development Environment

#### Option A: Jupyter Notebook (Recommended for beginners)
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Your browser will open with the Jupyter interface
3. Create a new Python 3 notebook

#### Option B: VS Code
1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install Python extension
3. Open your project folder
4. Select your virtual environment as the Python interpreter

#### Option C: PyCharm
1. Install [PyCharm Community Edition](https://www.jetbrains.com/pycharm/)
2. Create new project and select your virtual environment
3. Configure interpreter to use your `simulation_env`

### Step 5: Create Your First Simulation

Let's create a simple population growth simulation to test your setup:

Create a file called `population_simulation.py`:

```python
import numpy as np
import matplotlib.pyplot as plt

def population_growth_simulation(initial_pop, growth_rate, time_steps):
    """
    Simulate exponential population growth
    """
    population = [initial_pop]
    
    for t in range(1, time_steps):
        new_pop = population[-1] * (1 + growth_rate)
        population.append(new_pop)
    
    return population

## Run simulation
initial_population = 100
growth_rate = 0.05  # 5% growth per time step
time_steps = 50

results = population_growth_simulation(initial_population, growth_rate, time_steps)

## Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(time_steps), results, 'b-', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('Population Growth Simulation')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Initial population: {initial_population}")
print(f"Final population: {results[-1]:.2f}")
print(f"Growth factor: {results[-1]/initial_population:.2f}x")
```

### Step 6: Test Your Setup

1. Save the file and run it:
   ```bash
   python population_simulation.py
   ```

2. You should see:
   - A graph showing exponential growth
   - Printed statistics about the simulation

If this works, your simulation environment is ready!

### Step 7: Advanced Setup (Optional)

#### Performance Optimization
For computationally intensive simulations:

```bash
pip install numba  # JIT compilation for faster execution
pip install dask   # Parallel computing
```

#### GPU Acceleration
For simulations that can benefit from GPU:

```bash
pip install cupy   # NumPy-like library for GPU
# or
pip install tensorflow  # For machine learning simulations
```

#### Visualization Enhancements
For advanced plotting and animations:

```bash
pip install bokeh      # Interactive web-based plots
pip install mayavi     # 3D scientific visualization
pip install vpython    # 3D animations
```

### Step 8: Project Structure

Organize your simulation projects with this structure:

```
python-simulation/
├── simulation_env/          # Virtual environment
├── src/                    # Source code
│   ├── __init__.py
│   ├── models/            # Simulation models
│   ├── utils/             # Utility functions
│   └── visualizations/    # Plotting functions
├── data/                  # Input/output data
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```

Create `requirements.txt` to track dependencies:
```bash
pip freeze > requirements.txt
```

### Common Simulation Types & Libraries

#### 1. Agent-Based Modeling
```python
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
```

#### 2. Discrete Event Simulation
```python
import simpy

def process(env):
    yield env.timeout(1)  # Wait for 1 time unit
```

#### 3. Monte Carlo Simulation
```python
import random
import numpy as np

# Generate random samples
samples = np.random.normal(0, 1, 10000)
```

#### 4. Network Simulation
```python
import networkx as nx

# Create network
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])
```

### Troubleshooting

#### Common Issues:

1. **Import errors**: Ensure virtual environment is activated
2. **Plot not showing**: Install matplotlib backend or use `plt.show()`
3. **Permission errors**: Run terminal as administrator (Windows) or use `sudo` (Linux/macOS)
4. **Slow simulations**: Consider using NumPy arrays instead of Python lists

#### Performance Tips:

1. Use NumPy for numerical computations
2. Vectorize operations when possible
3. Profile your code with `cProfile`
4. Consider parallel processing for independent simulations

### Next Steps

1. **Learn simulation-specific libraries** based on your needs
2. **Practice with example projects** from GitHub repositories
3. **Join communities** like Stack Overflow, Reddit r/Python
4. **Read documentation** for libraries you'll use most
5. **Take online courses** in computational modeling

### Additional Resources

- **Books**: "Modeling and Simulation in Python" by Nino Narido
- **Courses**: Quantitative Method: Modeling and Simulation
- **Documentation**: Official docs for NumPy, SciPy, Matplotlib
- **Examples**: GitHub repositories with simulation examples

Your Python simulation environment is now ready! 
Start with simple models and gradually increase complexity as you become more comfortable with the tools.
