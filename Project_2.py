# Project_2.py
# This program uses Dijkstra's algorithm to find the most efficient path in a 400x250 sized workspace
# also generated, showing all nodes that were searched, and their indices.
# Author: Jon Kreinbrink
# Date: 2022/2/07

import time
import numpy as np
import heapq as hq
import cv2 as cv
import math

start_time = time.time()

NODEINDEX = 1 # Global index for counting nodes

#Action Sets MoveX, where X is the direction the action set moves the search algorithm

def MoveE(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0],CurrentNode[3][1]+1)
    return NewNode

def MoveS(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]-1,CurrentNode[3][1])
    return NewNode

def MoveW(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0],CurrentNode[3][1]-1)
    return NewNode   

def MoveN(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]+1,CurrentNode[3][1])
    return NewNode

def MoveSE(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1.4
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]-1,CurrentNode[3][1]+1)
    return NewNode

def MoveSW(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1.4
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]-1,CurrentNode[3][1]-1)
    return NewNode

def MoveNW(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + 1.4
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]+1,CurrentNode[3][1]-1)
    return NewNode   

def MoveNE(CurrentNode):
    global NODEINDEX
    NODEINDEX = NODEINDEX + 1
    weight = 1.4
    NewNode = [0,0,0,0]
    NewNode[0] = CurrentNode[0] + weight
    NewNode[1] = NODEINDEX
    NewNode[2] = CurrentNode[1]
    NewNode[3] = (CurrentNode[3][0]+1,CurrentNode[3][1]+1)
    return NewNode

#Freespace Calculation

# Initializes a numpy matrix full of np.inf values to represent the workspace
c2c_node =  np.full((250,400),np.inf)

# Initializes image file full of pixels with black color (0,0,0) RGB scale
image = np.zeros((250,400,3),np.uint8)

# Creating objects using half planes and semi-algebraic definitions, all object
# spaces have their cost set to -1
for x in range(0,400,1):
    for y in range(0,250,1):
        if x < 5 or x > 395 or y < 5 or y > 245:
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 95 and y <= 180 and (x>math.floor(105-((37/45)*(y-95)))) and x < math.floor(110-(.3125*(y-95))):
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 180 and y <= 185 and x>math.floor(105-((37/45)*(y-95))) and x < math.floor(85+(1*(y-180))):
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 185 and y <= 215 and x>math.ceil(31+(2.9666*(y-185))) and x < math.floor(85+(1*(y-180))):
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 59 and y <= 82 and x > math.ceil(200 -(1.732*(y-59))) and x < math.floor(200+(1.732*(y-59))):
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 82 and y <= 117 and x > 160 and x < 240:
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif y >= 117 and y <= 141 and x > math.ceil(160 +(1.732*(y-117))) and x < math.floor(240 -(1.732*(y-117))):
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]
        elif math.floor(((x-300)**2)+((y-185)**2)) < (45**2): 
            c2c_node[y][x] = -1
            image[y,x]= [255,0,0]