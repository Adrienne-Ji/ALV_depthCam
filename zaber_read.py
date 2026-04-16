import time
import csv
from datetime import datetime
from zaber_motion import Units
from zaber_motion.ascii import Connection

# Settings for 10 Hz
INTERVAL = 0.1  # 100ms between samples
data_log = []
start_time_iso = None

try:
    with Connection.open_network_share("localhost", 11421, "COM4") as connection:
        device = connection.detect_devices()[0]
        # Accessing all 3 axes from your X-MCC4
        axes = [device.get_axis(i) for i in [1, 2, 3]]
        
        print(f"Connected to {device.identify().device_id}")
        print("Logging at 10 Hz. Press Ctrl+C to Stop and Save.")

        data_log = []
        # Use a high-precision start time
        start_time = time.perf_counter()
        start_time_iso = datetime.now().isoformat(timespec="milliseconds")
        next_sample_time = start_time

        while True:
            now = time.perf_counter()
            
            # Trigger the read only when the 100ms 'bucket' is hit
            if now >= next_sample_time:
                # Get the 'Truth' for all three motor positions
                p1 = axes[0].settings.get("encoder.pos", Units.LENGTH_MILLIMETRES)
                p2 = axes[1].settings.get("encoder.pos", Units.LENGTH_MILLIMETRES)
                p3 = axes[2].settings.get("encoder.pos", Units.LENGTH_MILLIMETRES)
                
                # Use elapsed time for the CSV timestamp
                elapsed = round(now - start_time, 3)
                data_log.append([elapsed, p1, p2, p3])
                
                # Schedule the next sample exactly 0.1s after the PREVIOUS one
                # This prevents 'time drift' across your experiment
                next_sample_time += INTERVAL
            
            # Give the CPU a tiny rest so it doesn't run at 100%
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\n[EXIT] Saving Data...")
    if data_log:
        filename = f"ALV_10Hz_Log_{int(time.time())}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"start_time_iso={start_time_iso}"])
            writer.writerow(["time_sec", "Endo_mm", "Trans_mm", "Epi_mm"])
            writer.writerows(data_log)
        print(f"Success! {len(data_log)} samples saved to {filename}")