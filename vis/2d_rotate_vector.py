import ast
import pygame
import math
import numpy
width, height = 1200, 1200
pygame.init()
screen = pygame.display.set_mode((width, height))
base_surface = pygame.Surface((1200, 1200))
heatmap_surface = pygame.Surface((800, 800))
vector_surface = pygame.Surface((1200, 1200), pygame.SRCALPHA)
x = (base_surface.get_width() - heatmap_surface.get_width()) // 2
y = (base_surface.get_height() - heatmap_surface.get_height()) // 2
red_blue_map = {}
low = 0
high = 100


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
        outline_color: pygame.Color = pygame.Color("black"),
        outline_width: int = 4
    ):
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # --- HEAD ---
    def make_head(width, height):
        return [
            pygame.Vector2(0, height / 2),
            pygame.Vector2(width / 2, -height / 2),
            pygame.Vector2(-width / 2, -height / 2)
        ]
    head_verts_outline = make_head(head_width + outline_width, head_height + outline_width)
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts_outline)):
        head_verts_outline[i].rotate_ip(-angle)
        head_verts_outline[i] += translation
        head_verts_outline[i] += start
    pygame.draw.polygon(surface, outline_color, head_verts_outline)
    head_verts = make_head(head_width, head_height)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start
    pygame.draw.polygon(surface, color, head_verts)
    if arrow.length() >= head_height:
        def make_body(width):
            return [
                pygame.Vector2(-width / 2, body_length / 2),
                pygame.Vector2(width / 2, body_length / 2),
                pygame.Vector2(width / 2, -body_length / 2),
                pygame.Vector2(-width / 2, -body_length / 2)
            ]
        body_verts_outline = make_body(body_width + outline_width)
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts_outline)):
            body_verts_outline[i].rotate_ip(-angle)
            body_verts_outline[i] += translation
            body_verts_outline[i] += start
        pygame.draw.polygon(surface, outline_color, body_verts_outline)
        body_verts = make_body(body_width)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start
        pygame.draw.polygon(surface, color, body_verts)

def generate_current_heatmap(surface, data_arr_2d):
    global red_blue_map 
    heatmap_rectangle_centers = []
    side_length = 100
    loc_x = 0
    loc_y = 0
    for element in data_arr_2d.flat:
        rect_x = loc_x * side_length
        rect_y = loc_y * side_length
        pygame.draw.rect(surface, red_blue_map[element], pygame.Rect(rect_x + 4, rect_y + 4, side_length - 4, side_length - 4))
        center = pygame.Vector2(rect_x + side_length / 2, rect_y + side_length / 2)
        heatmap_rectangle_centers.append((center, element))
        if loc_x == 7:
            loc_x = 0
            loc_y += 1
        else:
            loc_x += 1

    return heatmap_rectangle_centers

def generate_vector_field(offset_x, offset_y, heatmap_rectangle_centers): 
    global vector_surface 
    mini_offset = 0
    max_length = 300 
    angle_rad = math.radians(45) 
    direction = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) 
    for idx, i in enumerate(heatmap_rectangle_centers): 
        center, element = i 
        length = max_length * math.cos(math.radians(element)) 
        screen_center = center + pygame.Vector2((offset_x + mini_offset), (offset_y)) 
        end = screen_center + direction * length 
        draw_arrow(vector_surface, screen_center, end, pygame.Color(red_blue_map[element]), body_width=5, head_width=10, head_height=12) 
        mini_offset += 0.2 
        mini_offset *= -1
    mini_offset = 0

import serial
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)


while True:
    line = ser.readline().decode('utf-8').strip()
    time = int(line.split(",")[0].strip("{time: "))
    data = ast.literal_eval(line.split("data: ")[1].strip("}"))
    ser.reset_input_buffer()
    data = numpy.array(data)
    data = data.reshape(8,8)
    base_surface.fill((255, 228, 196))
    x = (base_surface.get_width() - heatmap_surface.get_width()) // 2
    y = (base_surface.get_height() - heatmap_surface.get_height()) // 2
    screen.blit(base_surface, (0, 0))
    heatmap_surface.fill((0, 0, 0))
    heatmap_rectangle_centers = generate_current_heatmap(heatmap_surface, data)
    vector_surface.fill((0, 0, 0, 0))
    generate_vector_field(x, y, heatmap_rectangle_centers)
    screen.blit(heatmap_surface, (x, y))
    screen.blit(vector_surface, (0, 0))
    pygame.display.flip()
