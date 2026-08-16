import tkinter as tk, serial

# Nhớ đổi 'COM3' thành cổng thực tế trên máy bạn
s = serial.Serial('COM7', 115200) 

root = tk.Tk()
tk.Scale(root, from_=500, to=2500, orient='horizontal', length=300, command=lambda v: s.write((v + '\n').encode())).pack()
root.mainloop()