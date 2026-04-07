import pygame
import numpy as np

#pygame.draw.rect(surface, color, pygame.Rect(30, 30, 60, 60))  
#pygame.display.flip() 
#pygame.time.wait(3000)  # Pause for 3 seconds
#pygame.quit()

blue = (127, 191, 63+192)
w_height = 400
w_width = 400
side_length = 50
pygame.init()  
surface = pygame.display.set_mode((w_width, w_height)) 
width_offset = 4
height_offset = 4
max_sensor_val = 400

sensors_test = [[25, 27, 30, 28, 30, 31, 29, 25],
 [27, 28, 28, 29, 30, 33, 28, 32],
 [24, 28, 25, 27, 29, 33, 30, 30],
 [26, 27, 29, 30, 31, 27, 31, 30],
 [23, 26, 27, 33, 32, 29, 32, 31],
 [26, 28, 29, 32, 31, 31, 31, 31],
 [25, 25, 28, 31, 29, 30, 31, 29],
 [26, 27, 32, 32, 29, 29, 30, 30]]
arr_2d = np.array(sensors_test)

red_blue_map = {}
for i, val in enumerate(range(20, 36)):
    t = i / 15  # 0 → 1
    r = int(255 * (1 - t))
    b = int(255 * t)
    red_blue_map[val] = (r, 0, b)
while True:
    loc_x = 0
    loc_y = 0
    for element in arr_2d.flat:
        pos = (loc_x * side_length, loc_y * side_length)
        pygame.draw.rect(surface, (red_blue_map[element]), pygame.Rect(loc_x * side_length + 4, loc_y * side_length + 4, side_length - 4, side_length - 4))
        pygame.display.flip()
        if loc_x == 7:
            loc_x = 0
            loc_y += 1
        else:
            loc_x += 1

#while True:
#    pygame.display.flip()
#
#    for i_idx, i in enumerate(sensors_test):
#        for j_idx, j in enumerate(sensors_test):
#            sensor_val = sensors_test[i_idx][j_idx]
#            for x in range(0, w_width, side_length):
#                for y in range(0, w_height, side_length):
#                    pygame.draw.rect(surface, (red_map[sensor_val]), pygame.Rect(x + 4, y + 4, side_length - 4, side_length - 4))
#    pygame.time.wait(20);
#
#while True:
#    side_length = 20
#    for x in range(0, w_width, side_length):
#        for y in range(0, w_height, side_length):
#            rect = pg.Rect(x, y, side_length, side_length)
#            pg.draw.rect(screen, (random.randint(0, 255),random.randint(0, 255), random.randint(0, 255)), rect, 2)
#            pg.display.flip()
#


#pg.init()
#screen = pg.display.set_mode((1423, 989), pg.SHOWN)
#pg.display.set_caption("Touch IQ test 3d heatmap vis")
# 
#
#grid_size = 64
#grid_width = 8
#grid_height = 8
#grid_margin = 8
#distance_from_left = 10
#distance_from_top = 10
#
#while True:
#    grid = []
#    pos = pg.mouse.get_pos()
#    for y in range(grid_size):
#        row = []
#        for x in range(grid_size):
#            gx = x * (grid_width + grid_margin) + distance_from_left
#            gy = y * (grid_height + grid_margin) + distance_from_top
#            distance = pg.math.Vector2(pos).distance_to((gx + grid_width/2, gy + grid_height/2))
#            max_len = grid_size * (grid_height + grid_margin)
#
#            f = max(0, 1 - distance/ max_len)
#            color = (127 * f, 191 * f, 63 + 192 * f)
#            row.append([gx, gy, color])
#        grid.append(row)
#
#    for row in grid:
#            for x, y, colour in row:
##                screen.blit()
#                pg.draw.rect(screen, colour, (x, y, grid_width, grid_height))
#                pg.display.flip()
#
