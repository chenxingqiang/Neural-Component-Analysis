import numpy as np
import matplotlib.pyplot as plt
from run_secom_fault_detection import balanced_two_stage_detector, load_secom_data

def visualize_balanced_detector_metrics():
    """
    Visualize the separate T² and SPE metrics from the Balanced TwoStage detector
    """
    # Load SECOM data
    X_train, X_test, happen, y_test, normal_indices, fault_indices = load_secom_data()
    
    # Run the balanced two-stage detector
    print("Running Balanced Two-Stage Detector...")
    results = balanced_two_stage_detector(X_train, X_test, happen, [37, 38, 34, 36])
    
    # Extract metrics
    t2_statistics = results['t2_statistics']
    spe_statistics = results['spe_statistics']
    t2_threshold = results['t2_threshold']
    spe_threshold = results['spe_threshold']
    t2_alarms = results['t2_alarms']
    spe_alarms = results['spe_alarms']
    
    # Calculate real miss rates (not percentage)
    t2_misses = np.sum(t2_alarms[happen:] == 0)
    t2_miss_rate = t2_misses / (len(t2_alarms) - happen) if len(t2_alarms) > happen else 0
    t2_miss_percent = 100 * t2_miss_rate
    
    spe_misses = np.sum(spe_alarms[happen:] == 0)
    spe_miss_rate = spe_misses / (len(spe_alarms) - happen) if len(spe_alarms) > happen else 0
    spe_miss_percent = 100 * spe_miss_rate
    
    # Calculate false alarm rates
    t2_false_alarms = np.sum(t2_alarms[:happen] > 0)
    t2_false_rate = 100 * t2_false_alarms / happen if happen > 0 else 0
    
    spe_false_alarms = np.sum(spe_alarms[:happen] > 0)
    spe_false_rate = 100 * spe_false_alarms / happen if happen > 0 else 0
    
    # Create the visualization
    plt.figure(figsize=(15, 10))
    
    # Plot T² statistics
    plt.subplot(2, 2, 1)
    plt.plot(range(1, happen+1), t2_statistics[:happen], 'g-', alpha=0.7, label='T² Normal')
    plt.plot(range(happen+1, len(t2_statistics)+1), t2_statistics[happen:], 'm-', alpha=0.7, label='T² Fault')
    plt.axhline(y=t2_threshold, color='r', linestyle='--', label='T² Threshold')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Occurrence')
    plt.title('T² Statistics (Average Deviation)')
    plt.xlabel('Sample')
    plt.ylabel('T² Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot SPE statistics
    plt.subplot(2, 2, 2)
    plt.plot(range(1, happen+1), spe_statistics[:happen], 'g-', alpha=0.7, label='SPE Normal')
    plt.plot(range(happen+1, len(spe_statistics)+1), spe_statistics[happen:], 'm-', alpha=0.7, label='SPE Fault')
    plt.axhline(y=spe_threshold, color='r', linestyle='--', label='SPE Threshold')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Occurrence')
    plt.title('SPE Statistics (Maximum Deviation)')
    plt.xlabel('Sample')
    plt.ylabel('SPE Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot T² Alarms
    plt.subplot(2, 2, 3)
    plt.plot(range(1, happen+1), t2_alarms[:happen], 'b-', label='T² Alarms (Normal)')
    plt.plot(range(happen+1, len(t2_alarms)+1), t2_alarms[happen:], 'r-', label='T² Alarms (Fault)')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Occurrence')
    plt.title(f'T² Alarms ({t2_false_rate:.2f}% False, {t2_miss_percent:.2f}% Miss)')
    plt.xlabel('Sample')
    plt.ylabel('Alarm Status')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot SPE Alarms
    plt.subplot(2, 2, 4)
    plt.plot(range(1, happen+1), spe_alarms[:happen], 'b-', label='SPE Alarms (Normal)')
    plt.plot(range(happen+1, len(spe_alarms)+1), spe_alarms[happen:], 'r-', label='SPE Alarms (Fault)')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Occurrence')
    plt.title(f'SPE Alarms ({spe_false_rate:.2f}% False, {spe_miss_percent:.2f}% Miss)')
    plt.xlabel('Sample')
    plt.ylabel('Alarm Status')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('t2_spe_comparison.png')
    print("Plot saved as t2_spe_comparison.png")
    
    # Print summary
    print("\nBalanced Two-Stage Detector Summary:")
    print(f"T² False Alarm Rate: {t2_false_rate:.2f}%")
    print(f"T² Miss Rate: {t2_miss_percent:.2f}%")
    print(f"SPE False Alarm Rate: {spe_false_rate:.2f}%")
    print(f"SPE Miss Rate: {spe_miss_percent:.2f}%")

if __name__ == "__main__":
    visualize_balanced_detector_metrics() 