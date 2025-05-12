import numpy as np
import matplotlib.pyplot as plt
import torch
from src.detectors.spe_fault_detector import SPEFaultDetector
from src.detectors.enhanced_transformer_detection import load_data
import time

def integrate_spe_detector():
    """
    Example of integrating the SPE fault detector with existing process monitoring system
    using data from the Tennessee Eastman process
    """
    print("Loading Tennessee Eastman process data...")

    # Load data using the existing data loader
    X_train, X_test, happen = load_data(is_mock=False)

    print(f"Data loaded - Training: {X_train.shape}, Testing: {X_test.shape}")
    print(f"Fault occurs at sample {happen} in test data")

    # Create SPE fault detector
    print("\nInitializing SPE fault detector...")
    input_dim = X_train.shape[1]
    detector = SPEFaultDetector(input_dim=input_dim, hidden_dim=32)

    # Train the model or load pre-trained model
    try:
        print("Trying to load pre-trained model...")
        success = detector.load_model('results/models/te_spe_detector.pth')
        if not success:
            raise FileNotFoundError("Pre-trained model not found")
    except Exception as e:
        print(f"Could not load pre-trained model: {str(e)}")
        print("Training new model on normal operating data...")

        # Track training time
        start_time = time.time()

        # Train the model
        losses = detector.train(
            X_train,
            epochs=100,
            batch_size=32,
            lr=0.001,
            validation_split=0.1
        )

        training_time = time.time() - start_time
        print(f"Model trained in {training_time:.2f} seconds")

        # Save model for future use
        detector.save_model('results/models/te_spe_detector.pth')

        # Plot training loss
        plt.figure(figsize=(10, 4))
        plt.plot(losses[0], label='Training Loss')
        plt.plot(losses[1], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SPE Detector Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/plots/spe_detector_training_loss.png')
        plt.close()
        print("Training loss plot saved to 'results/plots/spe_detector_training_loss.png'")

    # Evaluate performance on test data
    print("\nEvaluating model performance...")
    metrics = detector.evaluate_performance(X_test, happen)

    # Print detection results
    print("\n===== SPE-Based Fault Detection Results =====")
    print(f"False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")
    print(f"Miss Rate: {metrics['miss_rate']:.2f}%")

    if metrics['detection_time'] is not None:
        print(f"Detection Time: {metrics['detection_time']} samples")
    else:
        print("Detection Time: Not detected")

    # Plot results
    detector.plot_results(X_test, happen, save_path='results/plots/spe_detector_results.png')
    print("Results plot saved to 'results/plots/spe_detector_results.png'")

    # Compare with other methods
    compare_with_other_methods(metrics)

    return detector, metrics


def compare_with_other_methods(spe_metrics):
    """Compare SPE results with other fault detection methods"""
    print("\n===== Performance Comparison with Other Methods =====")

    # Create comparison table
    methods = [
        "Enhanced Transformer T²",  # Previous T² approach
        "SPE-Based Detection",      # Our SPE method
    ]

    false_rates = [3.75, spe_metrics['false_alarm_rate']]
    miss_rates = [97.94, spe_metrics['miss_rate']]

    detection_times = ["Not detected",
                      str(spe_metrics['detection_time']) if spe_metrics['detection_time'] is not None else "Not detected"]

    # Print table
    print(f"{'Method':<25} | {'False Alarm (%)':<15} {'Miss Rate (%)':<15} | {'Detection Time':<15}")
    print("-" * 75)

    for i, method in enumerate(methods):
        print(f"{method:<25} | {false_rates[i]:<15.2f} {miss_rates[i]:<15.2f} | {detection_times[i]:<15}")


def real_time_monitoring_simulation():
    """
    Simulate real-time process monitoring using the SPE fault detector
    """
    print("\n===== Real-Time Monitoring Simulation =====")

    # Load detector
    try:
        detector = SPEFaultDetector(input_dim=52)  # Adjust input dimension as needed
        success = detector.load_model('results/models/te_spe_detector.pth')
        if not success:
            print("Could not load model for real-time simulation")
            return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load test data
    _, X_test, happen = load_data(is_mock=False)

    # Parameters for monitoring
    buffer_size = 10  # Number of recent samples to consider
    alarm_threshold = 3  # Consecutive alarms to trigger fault detection

    print(f"Starting simulation with {buffer_size} sample buffer and {alarm_threshold} consecutive alarms threshold")
    print("Processing data samples sequentially...")

    # Monitoring variables
    alarm_count = 0
    detection_time = None
    fault_detected = False

    # Store results for plotting
    timestamps = []
    spe_values = []
    alarm_status = []

    # Process samples one by one to simulate real-time data
    for i in range(len(X_test)):
        # Get current sample
        current_sample = X_test[i:i+1]  # Keep 2D shape

        # Detect potential fault in current sample
        is_fault, spe_value = detector.detect_faults(current_sample)

        # Update monitoring variables
        if is_fault[0]:
            alarm_count += 1
        else:
            alarm_count = 0

        # Check if we've reached alarm threshold
        if alarm_count >= alarm_threshold and not fault_detected:
            fault_detected = True
            detection_time = i - happen if i >= happen else None

        # Store results
        timestamps.append(i)
        spe_values.append(spe_value[0])
        alarm_status.append(1 if is_fault[0] else 0)

        # Print update every 50 samples
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(X_test)} samples...")

        # Print when fault is detected
        if fault_detected and detection_time is not None and i == happen + detection_time:
            print(f"FAULT DETECTED at sample {i+1}, {detection_time} samples after actual fault")

    # Plot real-time monitoring results
    plt.figure(figsize=(12, 8))

    # Plot SPE values
    plt.subplot(2, 1, 1)
    plt.plot(timestamps[:happen], spe_values[:happen], 'g-', label='Normal')
    plt.plot(timestamps[happen:], spe_values[happen:], 'r-', label='Fault')
    plt.axhline(y=detector.spe_limit, color='k', linestyle='--', label='Control Limit')
    plt.axvline(x=happen, color='m', linestyle='-', label='Actual Fault')
    plt.title('Real-Time SPE Monitoring')
    plt.xlabel('Sample')
    plt.ylabel('SPE Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot alarm status
    plt.subplot(2, 1, 2)
    plt.step(timestamps, alarm_status, 'b-', where='post', label='Alarm Status')
    plt.axvline(x=happen, color='m', linestyle='-', label='Actual Fault')
    if detection_time is not None:
        plt.axvline(x=happen+detection_time, color='r', linestyle='--', label='Detection')
    plt.title('Alarm Status (1=Alarm, 0=Normal)')
    plt.xlabel('Sample')
    plt.ylabel('Status')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/real_time_monitoring.png')
    plt.close()

    print("\nReal-time monitoring simulation completed")
    print(f"Results saved to 'results/plots/real_time_monitoring.png'")

    # Print summary
    print("\n===== Real-Time Monitoring Results =====")
    if fault_detected and detection_time is not None:
        print(f"Fault detected {detection_time} samples after occurrence")
    elif fault_detected:
        print("False alarm: fault detected before actual fault occurred")
    else:
        print("No fault detected during simulation")

    # Calculate statistics
    false_alarms = sum(alarm_status[:happen])
    false_alarm_rate = 100 * false_alarms / happen if happen > 0 else 0

    misses = alarm_threshold - sum(alarm_status[happen:]) > 0
    miss_rate = 100 if misses else 0

    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
    print(f"Miss Rate: {miss_rate:.2f}%")
    print(f"Detection Delay: {detection_time if detection_time is not None else 'Not detected'}")


if __name__ == "__main__":
    # Run the integration example
    detector, metrics = integrate_spe_detector()

    # Run real-time monitoring simulation
    real_time_monitoring_simulation()
