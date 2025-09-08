import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from datetime import datetime, timedelta

class Machine:
    """Individual teller/machine with breakdown capability"""
    def __init__(self, env, machine_id, mtbf=50, mttr=5):
        self.env = env
        self.machine_id = machine_id
        self.mtbf = mtbf  # Time Between Failures
        self.mttr = mttr  # Time To Repair
        self.is_working = True
        self.total_downtime = 0
        self.breakdown_count = 0
        self.breakdown_events = []  # (time, event_type)
        
        
        env.process(self.breakdown_process())
    
    def breakdown_process(self):
        """Process that handles machine breakdowns"""
        while True:
            time_to_failure = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(time_to_failure)
            
            # Machine breaks down
            if self.is_working:
                self.is_working = False
                self.breakdown_count += 1
                breakdown_time = self.env.now
                self.breakdown_events.append((breakdown_time, 'breakdown'))
                print(f"*** BREAKDOWN: Teller {self.machine_id} failed at {breakdown_time:.2f} ***")
                
                # Repair time
                repair_time = random.expovariate(1.0 / self.mttr)
                yield self.env.timeout(repair_time)
                
                # Machine is repaired
                self.is_working = True
                repair_complete_time = self.env.now
                self.total_downtime += repair_complete_time - breakdown_time
                self.breakdown_events.append((repair_complete_time, 'repair'))
                print(f"*** REPAIR: Teller {self.machine_id} fixed at {repair_complete_time:.2f} ***")

class VisualizedBankWithBreakdowns:
    def __init__(self, env, num_tellers, service_time, mtbf=50, mttr=5):
        self.env = env
        self.num_tellers = num_tellers
        self.service_time = service_time
        
        # Create individual machines
        self.machines = [Machine(env, i, mtbf, mttr) for i in range(num_tellers)]
        self.tellers = simpy.Resource(env, num_tellers)
        
        # Statistics tracking
        self.customers_served = 0
        self.total_wait_time = 0
        self.customers_in_system = 0
        self.customers_waiting = 0
        self.customers_balked = 0  # Customers who left due to long queues
        
    
        self.max_data_points = 200
        self.time_data = deque(maxlen=self.max_data_points)
        self.queue_length_data = deque(maxlen=self.max_data_points)
        self.customers_in_system_data = deque(maxlen=self.max_data_points)
        self.working_tellers_data = deque(maxlen=self.max_data_points)
        self.utilization_data = deque(maxlen=self.max_data_points)
        
        # Complete history for final analysis
        self.wait_times = []
        self.service_events = []  # (time, event_type, customer_id, details)
        
    def get_working_tellers(self):
        """Get number of currently working tellers"""
        return sum(1 for machine in self.machines if machine.is_working)
    
    def get_available_teller(self):
        """Get an available working teller, or None if all are busy/broken"""
        working_machines = [m for m in self.machines if m.is_working]
        return len(working_machines) > len(self.tellers.queue)
    
    def serve_customer(self, customer_id):
        """Process for serving a customer with breakdown awareness"""
        arrival_time = self.env.now
        self.customers_in_system += 1
        self.service_events.append((arrival_time, 'arrival', customer_id, 'normal'))
        
        print(f"Customer {customer_id} arrives at {arrival_time:.2f}")
        
    
        working_tellers = self.get_working_tellers()
        if working_tellers == 0 or len(self.tellers.queue) >= 10:  # Balking condition
            if random.random() < 0.3:  # 30% chance of balking
                self.customers_balked += 1
                self.customers_in_system -= 1
                self.service_events.append((self.env.now, 'balk', customer_id, f'queue_length={len(self.tellers.queue)}'))
                print(f"Customer {customer_id} balked at {self.env.now:.2f} (queue too long or no working tellers)")
                return
        
        with self.tellers.request() as request:
            yield request
            
            # Wait until we have a working teller
            while self.get_working_tellers() == 0:
                print(f"Customer {customer_id} waiting for teller repair at {self.env.now:.2f}")
                yield self.env.timeout(0.1)  # Small delay to check again
            
            wait_time = self.env.now - arrival_time
            self.total_wait_time += wait_time
            self.wait_times.append(wait_time)
            
            print(f"Customer {customer_id} starts service at {self.env.now:.2f} (waited {wait_time:.2f})")
            
            # Service time (potentially interrupted by breakdown)
            service_duration = random.expovariate(1.0 / self.service_time)
            service_start = self.env.now
            
            # Find which teller is serving (simplified - assume first available working teller)
            serving_teller = next((m for m in self.machines if m.is_working), None)
            
            try:
                yield self.env.timeout(service_duration)
                
                # Check if teller broke down during service
                if serving_teller and not serving_teller.is_working:
                    print(f"Service interrupted for Customer {customer_id} due to teller breakdown!")
                    # Customer needs to wait for repair or another teller
                    additional_wait = random.uniform(1, 3)
                    yield self.env.timeout(additional_wait)
                
            except simpy.Interrupt:
                print(f"Customer {customer_id} service interrupted!")
                yield self.env.timeout(1)  # Additional delay due to interruption
            
            # Customer leaves
            self.customers_served += 1
            self.customers_in_system -= 1
            self.service_events.append((self.env.now, 'departure', customer_id, f'service_time={self.env.now-service_start:.2f}'))
            print(f"Customer {customer_id} finished at {self.env.now:.2f}")
    
    def record_stats(self):
        """Record current statistics for real-time visualization"""
        current_time = self.env.now
        queue_length = len(self.tellers.queue)
        working_tellers = self.get_working_tellers()
        
        self.time_data.append(current_time)
        self.queue_length_data.append(queue_length)
        self.customers_in_system_data.append(self.customers_in_system)
        self.working_tellers_data.append(working_tellers)
        
        # Calculate utilization
        if working_tellers > 0:
            busy_tellers = min(working_tellers, self.customers_in_system)
            utilization = (busy_tellers / working_tellers) * 100
        else:
            utilization = 0
        self.utilization_data.append(utilization)

class RealTimeAnimator:
    """Class to handle real-time animated visualization"""
    def __init__(self, bank):
        self.bank = bank
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-Time Bank Simulation with Machine Breakdowns', fontsize=16)
        
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the four subplots"""
        # Plot 1: Queue Length
        self.ax1 = self.axes[0, 0]
        self.ax1.set_title('Queue Length Over Time')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Customers in Queue')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        
        # Plot 2: Working Tellers
        self.ax2 = self.axes[0, 1]
        self.ax2.set_title('Working Tellers Over Time')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Number of Working Tellers')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, self.bank.num_tellers + 0.5)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        
        # Plot 3: System Utilization
        self.ax3 = self.axes[1, 0]
        self.ax3.set_title('System Utilization (%)')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Utilization %')
        self.ax3.set_ylim(0, 105)
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2)
        
        # Plot 4: Customers in System
        self.ax4 = self.axes[1, 1]
        self.ax4.set_title('Total Customers in System')
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Customers')
        self.ax4.grid(True, alpha=0.3)
        self.line4, = self.ax4.plot([], [], 'm-', linewidth=2)
        
    def animate(self, frame):
        """Animation function called by matplotlib"""
        if len(self.bank.time_data) > 0:
            times = list(self.bank.time_data)
            
            # Update Plot 1: Queue Length
            queues = list(self.bank.queue_length_data)
            self.line1.set_data(times, queues)
            if times:
                self.ax1.set_xlim(max(0, times[-1] - 20), times[-1] + 1)
                if queues:
                    self.ax1.set_ylim(0, max(queues) + 1)
            
            # Update Plot 2: Working Tellers
            working = list(self.bank.working_tellers_data)
            self.line2.set_data(times, working)
            if times:
                self.ax2.set_xlim(max(0, times[-1] - 20), times[-1] + 1)
            
            # Update Plot 3: Utilization
            utilization = list(self.bank.utilization_data)
            self.line3.set_data(times, utilization)
            if times:
                self.ax3.set_xlim(max(0, times[-1] - 20), times[-1] + 1)
            
            # Update Plot 4: Customers in System
            customers = list(self.bank.customers_in_system_data)
            self.line4.set_data(times, customers)
            if times:
                self.ax4.set_xlim(max(0, times[-1] - 20), times[-1] + 1)
                if customers:
                    self.ax4.set_ylim(0, max(customers) + 1)
        
        return self.line1, self.line2, self.line3, self.line4

def customer_generator(env, bank, inter_arrival_time):
    """Generate customers with variable arrival rates"""
    customer_id = 1
    while True:
        env.process(bank.serve_customer(customer_id))
        
        # Variable arrival rate based on time of day
        time_factor = 1 + 0.5 * np.sin(env.now * 0.1)  # Sine wave variation
        next_arrival = random.expovariate(1.0 / (inter_arrival_time * time_factor))
        yield env.timeout(next_arrival)
        customer_id += 1

def stats_recorder(env, bank):
    """Process to record statistics periodically"""
    while True:
        bank.record_stats()
        yield env.timeout(0.2)  # Record every 0.2 time units for smooth animation

def run_simulation_with_real_time_viz():
    """Run simulation with real-time animated visualization"""
    # Parameters
    SIMULATION_TIME = 100.0
    NUM_TELLERS = 3
    AVERAGE_SERVICE_TIME = 4.0
    AVERAGE_INTER_ARRIVAL_TIME = 2.5
    MTBF = 30.0  # Time between failures
    MTTR = 8.0   # Time to repair
    
    print("=== Real-Time Bank Simulation with Machine Breakdowns ===")
    print(f"Simulation will run for {SIMULATION_TIME} time units")
    print(f"Tellers: {NUM_TELLERS}, MTBF: {MTBF}, MTTR: {MTTR}")
    
    # Create environment and bank
    env = simpy.Environment()
    bank = VisualizedBankWithBreakdowns(env, NUM_TELLERS, AVERAGE_SERVICE_TIME, MTBF, MTTR)
    
    # Start processes
    env.process(customer_generator(env, bank, AVERAGE_INTER_ARRIVAL_TIME))
    env.process(stats_recorder(env, bank))
    
    # Create animator
    animator = RealTimeAnimator(bank)
    
    # Create animation
    def run_simulation_step():
        try:
            env.run(until=env.now + 0.1)  # Run simulation in small steps
            return env.now < SIMULATION_TIME
        except:
            return False
    
    def combined_animate(frame):
        if run_simulation_step():
            return animator.animate(frame)
        else:
            # Simulation finished, show final results
            print_final_statistics(bank, SIMULATION_TIME)
            return animator.animate(frame)
    
    # Start animation
    anim = animation.FuncAnimation(
        animator.fig, combined_animate, interval=100, blit=False, cache_frame_data=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return bank, anim

def print_final_statistics(bank, sim_time):
    """Print comprehensive final statistics"""
    print(f"\n=== Final Statistics ===")
    print(f"Simulation Time: {sim_time}")
    print(f"Customers served: {bank.customers_served}")
    print(f"Customers balked: {bank.customers_balked}")
    
    if bank.wait_times:
        print(f"Average wait time: {np.mean(bank.wait_times):.2f}")
        print(f"Max wait time: {max(bank.wait_times):.2f}")
    
    if bank.queue_length_data:
        print(f"Average queue length: {np.mean(bank.queue_length_data):.2f}")
        print(f"Max queue length: {max(bank.queue_length_data)}")
    
    # Machine breakdown statistics
    total_breakdowns = sum(m.breakdown_count for m in bank.machines)
    total_downtime = sum(m.total_downtime for m in bank.machines)
    
    print(f"\n=== Machine Breakdown Statistics ===")
    print(f"Total breakdowns: {total_breakdowns}")
    print(f"Total downtime: {total_downtime:.2f}")
    
    for i, machine in enumerate(bank.machines):
        availability = ((sim_time - machine.total_downtime) / sim_time) * 100
        print(f"Teller {i}: {machine.breakdown_count} breakdowns, "
              f"{machine.total_downtime:.2f} downtime, {availability:.1f}% availability")
    
    # Overall system utilization
    avg_working_tellers = np.mean(bank.working_tellers_data) if bank.working_tellers_data else 0
    system_availability = (avg_working_tellers / bank.num_tellers) * 100
    print(f"Average system availability: {system_availability:.1f}%")

def run_static_analysis():
    """Run simulation and create static plots for detailed analysis"""
    print("Running simulation for static analysis...")
    
    env = simpy.Environment()
    bank = VisualizedBankWithBreakdowns(env, 3, 4.0, 25.0, 6.0)
    
    env.process(customer_generator(env, bank, 2.0))
    env.process(stats_recorder(env, bank))
    
    env.run(until=50.0)
    
    create_detailed_plots(bank, 50.0)
    return bank

def create_detailed_plots(bank, sim_time):
    """Create detailed static analysis plots"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Detailed Bank Simulation Analysis with Machine Breakdowns', fontsize=16)
    
    times = list(bank.time_data)
    
    # Plot 1: Queue Length and Working Tellers
    ax1 = axes[0, 0]
    ax1.plot(times, list(bank.queue_length_data), 'b-', label='Queue Length', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, list(bank.working_tellers_data), 'r-', label='Working Tellers', linewidth=2)
    ax1.set_title('Queue Length vs Working Tellers')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Queue Length', color='b')
    ax1_twin.set_ylabel('Working Tellers', color='r')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: System Utilization
    ax2 = axes[0, 1]
    ax2.plot(times, list(bank.utilization_data), 'g-', linewidth=2)
    ax2.set_title('System Utilization Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Utilization %')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wait Time Distribution
    ax3 = axes[1, 0]
    if bank.wait_times:
        ax3.hist(bank.wait_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(np.mean(bank.wait_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(bank.wait_times):.2f}')
        ax3.set_title('Customer Wait Time Distribution')
        ax3.set_xlabel('Wait Time')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Machine Breakdown Timeline
    ax4 = axes[1, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, machine in enumerate(bank.machines):
        breakdown_times = [t for t, event in machine.breakdown_events if event == 'breakdown']
        repair_times = [t for t, event in machine.breakdown_events if event == 'repair']
        
        # Plot breakdown periods as horizontal bars
        for j in range(len(breakdown_times)):
            if j < len(repair_times):
                ax4.barh(i, repair_times[j] - breakdown_times[j], 
                        left=breakdown_times[j], height=0.6, 
                        color=colors[i % len(colors)], alpha=0.7,
                        label=f'Teller {i}' if j == 0 else "")
    
    ax4.set_title('Machine Breakdown Timeline')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Teller ID')
    ax4.set_yticks(range(len(bank.machines)))
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Cumulative Statistics
    ax5 = axes[2, 0]
    cumulative_served = []
    cumulative_balked = []
    served_count = 0
    balked_count = 0
    
    for time, event, customer_id, details in bank.service_events:
        if event == 'departure':
            served_count += 1
        elif event == 'balk':
            balked_count += 1
        cumulative_served.append(served_count)
        cumulative_balked.append(balked_count)
    
    event_times = [t for t, _, _, _ in bank.service_events]
    ax5.plot(event_times, cumulative_served, 'g-', label='Customers Served', linewidth=2)
    ax5.plot(event_times, cumulative_balked, 'r-', label='Customers Balked', linewidth=2)
    ax5.set_title('Cumulative Customer Statistics')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: System Performance Metrics
    ax6 = axes[2, 1]
    if len(times) > 10:
        window_size = len(times) // 10
        avg_queue = np.convolve(list(bank.queue_length_data), 
                               np.ones(window_size)/window_size, mode='valid')
        avg_utilization = np.convolve(list(bank.utilization_data), 
                                     np.ones(window_size)/window_size, mode='valid')
        time_window = times[window_size-1:]
        
        ax6.plot(time_window, avg_queue, 'b-', label='Avg Queue Length', linewidth=2)
        ax6_twin = ax6.twinx()
        ax6_twin.plot(time_window, avg_utilization, 'r-', label='Avg Utilization %', linewidth=2)
        
        ax6.set_title('Moving Averages')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Queue Length', color='b')
        ax6_twin.set_ylabel('Utilization %', color='r')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with menu"""
    print("Bank Simulation with Machine Breakdowns")
    print("1. Real-time animated simulation")
    print("2. Static analysis with detailed plots")
    print("3. Both")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    try:
        if choice == '1':
            bank, anim = run_simulation_with_real_time_viz()
        elif choice == '2':
            bank = run_static_analysis()
        elif choice == '3':
            print("Running real-time simulation first...")
            bank, anim = run_simulation_with_real_time_viz()
            print("\nNow running static analysis...")
            bank2 = run_static_analysis()
        else:
            print("Invalid choice. Running real-time simulation...")
            bank, anim = run_simulation_with_real_time_viz()
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed: pip install matplotlib numpy simpy")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()