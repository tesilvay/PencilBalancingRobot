import serial
import time
import numpy as np


class ServoController:
    """
    Low-level serial interface to Arduino.
    """

    def __init__(self, port="/dev/ttyUSB1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)

        self.send("MODE,EXP")

    def send(self, cmd):
        msg = cmd.strip() + "\r\n"
        self.ser.write(msg.encode("utf-8"))
        self.ser.flush()

    def send_angles(self, a, b):
        cmd_str = f"CMD,{a:.2f},{b:.2f}"
        self.send(cmd_str)

class MockServoController:

    def __init__(self):
        self.start = time.perf_counter()

    def send(self, cmd):
        t = time.perf_counter() - self.start
        #print(f"{t:0.4f}s | {cmd}")

    def send_angles(self, theta1, theta2):
        cmd = f"CMD,{theta1:.2f},{theta2:.2f}"
        self.send(cmd)

class MechanismAdapter:

    def __init__(self, mech):
        self.mech = mech

    def command_to_angles(self, command):

        x = command.x_des
        y = command.y_des

        target_mm = np.array([x, y]) * 1000.0
        
        # returns theta's in rad
        theta1, theta2 =self.mech.ik(target_mm)
        
        return np.rad2deg(theta1), np.rad2deg(theta2)
    
class ServoSystem:

    def __init__(self, mechanism, port="/dev/ttyUSB1", frequency=250):

        self.adapter = MechanismAdapter(mechanism)
        
        if port == None:
            self.controller = MockServoController()
        else:
            self.controller = ServoController(port)

        self.last_send = 0.0
        self.period = 1.0 / frequency

    def send(self, command):

        now = time.time()

        if now - self.last_send < self.period:
            return

        theta1, theta2 = self.adapter.command_to_angles(command)

        self.controller.send_angles(theta1, theta2)

        self.last_send = now