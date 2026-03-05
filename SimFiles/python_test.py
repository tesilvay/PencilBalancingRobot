import serial
import time

arduino = serial.Serial("COM3", 115200, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

while True:
    num = input("Enter a number: ").strip()

    
    arduino.write((num + "\r\n").encode("utf-8"))
    arduino.flush()

   
    time.sleep(0.02)

    resp = arduino.readline().decode("utf-8", errors="replace").strip()
    print("Arduino returned:", resp)