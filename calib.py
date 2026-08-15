import tkinter as tk
import serial
import time

arduino = serial.Serial('COM3', 9600)
time.sleep(2)


def send_angle(val):
    arduino.write(f"{val}\n".encode())

root = tk.Tk()
tk.Scale(root, from_=0, to=180, orient='horizontal', length=300, command=send_angle).pack(pady=20)
root.mainloop()