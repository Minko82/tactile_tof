import os
import ast
import time
import serial
import argparse
from datetime import datetime
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np

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

width, height = 800, 600
pygame.init()
screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
pygame.event.set_grab(True) 
pygame.mouse.set_visible(False)
offscreen_surface = pygame.Surface((400, 400))
vector_surface = pygame.Surface((400, 400))

camera_pos = [0, 1, 5]  
camera_angle = [0, 0]  
MOUSE_SENSITIVITY = 0.2
MOVE_SPEED = 0.2
VERT_SPEED = 0.2 

glEnable(GL_DEPTH_TEST)

red_blue_map = {}
low = 5
high = 50

for i, val in enumerate(range(low, high)):
    print(val)
    t = i / (high - low)  
    r = int(255 * (1 - t))
    b = int(255 * t)
    red_blue_map[val] = (r, 0, b)



def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)

def generate_current_heatmap(surface, data_arr_2d):
    global red_blue_map
    side_length = 50
    loc_x = 0
    loc_y = 0
    for element in data_arr_2d.flat:
        pos = (loc_x * side_length, loc_y * side_length)
        print(red_blue_map[element])
        pygame.draw.rect(surface, (red_blue_map[element]), pygame.Rect(loc_x * side_length + 4, loc_y * side_length + 4, side_length - 4, side_length - 4))
        pygame.display.flip()
        if loc_x == 7:
            loc_x = 0
            loc_y += 1
        else:
            loc_x += 1

def draw_force_vector(surface, data_arr_2d):
    max_val = np.amax(data_arr_2d)
    vector_surface.fill(pygame.Color(0, 0, 0))
    draw_arrow(vector_surface, pygame.Vector2(100, 200), pygame.Vector2(100 + 200 * (max_val/100), 200), pygame.Color(255, 0, 0), 50, 50, 50)
    pygame.display.flip()

texID = glGenTextures(1)
def surfaceToTexture(pygame_surface):
    global texID
    rgb_surface = pygame.image.tostring(pygame_surface, 'RGB')
    glBindTexture(GL_TEXTURE_2D, texID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    surface_rect = pygame_surface.get_rect()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, surface_rect.width, surface_rect.height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_surface)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

def handle_keys():
    keys = pygame.key.get_pressed()
    dx, dz, dy = 0, 0, 0
    if keys[pygame.K_w]:
        dz = -MOVE_SPEED
    if keys[pygame.K_s]:
        dz = MOVE_SPEED
    if keys[pygame.K_a]:
        dx = -MOVE_SPEED
    if keys[pygame.K_d]:
        dx = MOVE_SPEED
    if keys[pygame.K_SPACE]:
        dy = VERT_SPEED
    if keys[pygame.K_LCTRL]:
        dy = -VERT_SPEED
    yaw_rad = math.radians(camera_angle[1])
    camera_pos[0] += dx * math.cos(yaw_rad) - dz * math.sin(yaw_rad)
    camera_pos[2] += dx * math.sin(yaw_rad) + dz * math.cos(yaw_rad)
    camera_pos[1] += dy

def handle_mouse():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    delta_x = mouse_x - width // 2
    delta_y = mouse_y - height // 2
    camera_angle[1] += delta_x * MOUSE_SENSITIVITY
    camera_angle[0] -= delta_y * MOUSE_SENSITIVITY
    camera_angle[0] = max(-90, min(90, camera_angle[0]))
    pygame.mouse.set_pos(width // 2, height // 2)

def check_collision(pos, ground_level=0):
    if pos[1] < ground_level:
        pos[1] = ground_level

def update():
    handle_mouse()
    handle_keys()
    check_collision(camera_pos)

def set_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(camera_angle[0], 1, 0, 0)
    glRotatef(camera_angle[1], 0, 1, 0)
    glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])

def draw_cube():
    vertices = [
    (1, -1, -1),  
    (1, 1, -1),   
    (-1, 1, -1),  
    (-1, -1, -1), 
    (1, -1, 1),   
    (1, 1, 1),    
    (-1, -1, 1),  
    (-1, 1, 1)    
    ]
    faces = [
        (0, 1, 2, 3),  
        (4, 5, 7, 6),  
        (0, 4, 5, 1),  
        (3, 2, 7, 6),  
        (1, 5, 7, 2),  
        (0, 3, 6, 4)   
    ]
    surfaceToTexture(offscreen_surface)
    glBindTexture(GL_TEXTURE_2D, texID)
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(1, 1, -1)
    glTexCoord2f(1, 0); glVertex3f(1, 1, 1)
    glTexCoord2f(1, 1); glVertex3f(-1, 1, 1)
    glTexCoord2f(0, 1); glVertex3f(-1, 1, -1)
    glEnd()
    surfaceToTexture(vector_surface)
    glBindTexture(GL_TEXTURE_2D, texID)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(1, -1, 1)
    glTexCoord2f(1, 0); glVertex3f(1, 1, 1)
    glTexCoord2f(1, 1); glVertex3f(-1, 1, 1)
    glTexCoord2f(0, 1); glVertex3f(-1, -1, 1)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    for face in faces:
        if face != (1, 5, 7, 2):
            for vertex in face:
                glVertex3fv(vertices[vertex])
    glEnd()

def draw_floor(size=20):
    glBegin(GL_LINES)
    glColor3f(0.5, 0.5, 0.5)
    for i in range(-size, size + 1):
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
    glEnd()
    glColor3f(1, 1, 1)

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        update()
        set_camera()
        line = ser.readline().decode('utf-8').strip()
        time = int(line.split(",")[0].strip("{time: "))
        data = ast.literal_eval(line.split("data: ")[1].strip("}"))
        ser.reset_input_buffer()
        data = np.array(data)
        data = data.reshape(8, 8)
        generate_current_heatmap(offscreen_surface, data)
        draw_force_vector(vector_surface, data)
        draw_floor()
        draw_cube()
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
