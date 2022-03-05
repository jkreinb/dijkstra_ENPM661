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
