import dv_processing as dv
import serial
import time
import csv
import threading
import signal
import sys

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 2000000
AEDAT_FILE = "events.aedat4"
CSV_FILE = "encoder.csv"
FLUSH_INTERVAL_SEC = 1.0
# ----------------------------------------


running = True


def signal_handler(sig, frame):
    global running
    print("\nStopping acquisition...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


# ---------------- Encoder Setup ----------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
ser.flushInput()

csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_ns", "angle_rad"])


def encoder_thread():
    last_flush = time.time()

    while running:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        try:
            angle = float(line)
            timestamp = time.perf_counter_ns()
            csv_writer.writerow([timestamp, angle])
        except ValueError:
            continue

        # Periodic flush
        if time.time() - last_flush > FLUSH_INTERVAL_SEC:
            csv_file.flush()
            last_flush = time.time()


# ---------------- Event Camera Setup ----------------
camera = dv.io.CameraCapture()

if not camera.isEventStreamAvailable():
    print("No event stream available.")
    sys.exit(1)

writer = dv.io.MonoCameraWriter(
    AEDAT_FILE,
    camera.getEventResolution()
)

# ---------------- Start Threads ----------------
threading.Thread(target=encoder_thread, daemon=True).start()

print("Recording started. Press Ctrl+C to stop.")

# ---------------- Main Event Loop ----------------
while running:
    events = camera.getNextEventBatch()

    if events is not None:
        writer.writeEvents(events)

# ---------------- Cleanup ----------------
print("Finalizing...")

csv_file.flush()
csv_file.close()
ser.close()
writer.close()

print("Done.")