import ast
import numpy as np
import math
import os
import pygame as pg
from pygame.locals import *
from datetime import datetime
import time
import serial
import argparse
parser = argparse.ArgumentParser(description="visualize_record_touchiq")
parser.add_argument("--port", type=str, default=None, help="port name; ex: /dev/ttyACM0 or like /dev/cu.usbmodem101")
parser.add_argument("--baud", type=int, default=115200, help="baud rate, defaults to 115200")
parser.add_argument("--max_distance", type=int, default=2000, help="max distance in mm for color scale")
parser.add_argument("--record_path", type=str, default=None, help="path to record csv data to. todo: docuemnt csv formatting")

args_cli = parser.parse_args()
SERIAL_PORT = args_cli.port
BAUD_RATE = args_cli.baud
MAX_DISTANCE_MM = args_cli.max_distance
RECORD_PATH = args_cli.record_path
if not SERIAL_PORT:
    raise AssertionError("you must provide a port val.; for example, --port /dev/ttyACM0")
else:
    assert(os.path.exists(str(SERIAL_PORT))), "port isn't showing up on file system"
if not RECORD_PATH:
   pass 
else:
    if not os.path.exists(RECORD_PATH):
        f = open(RECORD_PATH, "w")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    exit()
#while True:
#    try:
#        line = ser.readline().decode('utf-8').strip()
#        # Only process lines that contain data (look for digits)
#        print(line)
#        time = int(line.split(",")[0].strip("{time: "))
#        data = ast.literal_eval(line.split("data: ")[1].strip("}"))
#        print(time)
#        print(data)
#        ser.reset_input_buffer()
#    except KeyboardInterrupt:
#            exit()
 
def test():
    pg.init()
    screen = pg.display.set_mode((1423, 989), pg.SHOWN)
    image_path = "/home/mateo/code/correll_ws/tactile_tof/touchiq_mount_cad.png"
    image = pg.image.load(image_path)
    pg.display.set_caption("TouchIQ 3d Visualization")
    pg.mouse.set_visible(False)
    background = pg.Surface(screen.get_size())
    if pg.font:
        font = pg.font.Font(None, 64)
        text = font.render("Pummel The Chimp, And Win $$$", True, (10, 10, 10))
        textpos = text.get_rect(centerx=background.get_width() / 2, y=10)
        background.blit(text, textpos)
    screen.blit(background, (0, 0))
    pg.display.flip()
    going = True
    while going:
        line = ser.readline().decode('utf-8').strip()
        print(line)
        time = int(line.split(",")[0].strip("{time: "))
        data = ast.literal_eval(line.split("data: ")[1].strip("}"))
        print(time)
        print(data)
        ser.reset_input_buffer()
        data = np.array(data)
        data = data.reshape(8, 8)
        print(data)
        screen.blit(image, (0, 0))
        draw_circles(screen, data)
        pg.display.flip()
        pass

def draw_circles(screen, data):
    h_spacing = math.floor(1423/ 8)
    v_spacing = math.floor(989 / 8)
    curr_i = 0
    curr_j = 0
    for i_idx, i in enumerate(data):
        for j_idx, j in enumerate(data):
            curr_i = h_spacing * i_idx
            curr_j = v_spacing * j_idx
            print(f"{curr_i, curr_j=}")
            print(f"{h_spacing=}")
            print(f"{v_spacing=}")
            print(type(h_spacing))
            print(type(v_spacing))
            if data[i_idx][j_idx] == 19:
                color = (0, 200, 0)
            else:
                color = (200, 0, 0)
            pg.draw.circle(screen, color, (curr_i, curr_j), 20)
    
if __name__ == "__main__":
    test()
