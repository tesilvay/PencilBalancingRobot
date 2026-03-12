import serial
import time


class ServoController:

    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)

    def send(self, cmd):
        msg = cmd.strip() + "\r\n"
        self.ser.write(msg.encode("utf-8"))
        self.ser.flush()

    def set_mode(self, mode):
        self.send(f"MODE,{mode}")

    def experiment_cmd(self, a_deg, b_deg):
        self.send(f"CMD,{a_deg},{b_deg}")


# NEW: predefined cycle points
cycle_points = [
    (0, 0),
    (45, 45),
    (90, 90),
    (135, 135),
    (90, 45),
    (45, 90)
]


def run_cycle(ctrl):

    print("Starting cycle mode (Ctrl+C to stop)")

    try:
        while True:
            for a, b in cycle_points:

                ctrl.experiment_cmd(a, b)
                print(f"cycle → {a}, {b}")

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nCycle stopped")


def experiment_console(ctrl):

    print("\nExperiment mode")
    print("Enter two angles in degrees: a b")
    print("Example: 90 95")
    print("Type 'cycle' to run preset sequence")
    print("Type q to quit\n")

    ctrl.set_mode("EXP")

    while True:

        line = input("> ").strip()

        if line.lower() in ["q", "quit", "exit"]:
            break

        # NEW: cycle command
        if line.lower() == "cycle":
            run_cycle(ctrl)
            continue

        try:
            a, b = map(float, line.split())

            ctrl.experiment_cmd(a, b)

            print(f"sent angles: {a:.2f}, {b:.2f}")

        except Exception:
            print("Invalid input. Example: 90 95")


if __name__ == "__main__":

    ctrl = ServoController("/dev/ttyUSB0")
    
    experiment_console(ctrl)