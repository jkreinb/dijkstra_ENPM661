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

# Creates variables for setting goal/start locations
goal_set = False
start_set = False

# Goal setting loop
while goal_set == False:
    print("\nPlease enter a goal location on the workspace\n")
    print("-------------------------------------------------------\n")
    goal_x = int(input("Enter X position of goal node, must be an integer between 0-400\n"))
    goal_y = int(input("Enter Y position of goal node, must be an integer between 0-250\n"))
    if goal_x < 0 or goal_x > 400 or goal_y > 250 or goal_y < 0: # Checks if goal is outside of workspace
        print("\nGoal outside of workspace! Please try again\n")
        time.sleep(2)
        continue
    elif c2c_node[goal_y][goal_x] == np.inf:
        goal_set = True
    else: # If goal is not equal to np.inf it's an obstacle, prompts user again for new goal
        print("\nGoal location is ontop of an obstacle! Please enter new goal location\n")
        time.sleep(2)

while start_set == False:
    print("\nPlease enter a starting location on the workspace\n")
    print("-------------------------------------------------------\n")
    start_x = int(input("Enter X position of start node, must be an integer between 0-400\n"))
    start_y = int(input("Enter Y position of start node, must be an integer between 0-250\n"))
    if start_x < 0 or start_x > 400 or start_y > 250 or start_y < 0: # Checks if start is outside workspace
        print("\nStarting location is outside of workspace! Please try again\n")
        time.sleep(2)
        continue
    elif c2c_node[start_y][start_x] == np.inf:
        start_set = True
    else: # If start is not equal to np.inf it's an obstacle, prompts user again for new start
        print("\nStarting location is ontop of an obstacle! Please enter new starting location\n")
        time.sleep(2)

goal_node = (goal_y,goal_x)
start_node = (start_y,start_x)

print("\nGoal node is:\n",goal_node)
print("\nStart node is:\n",start_node)


# Creates window showing obstacle
flipped_image=cv.flip(image,0)
cv.imshow("actual_image",flipped_image)
cv.waitKey(10)

# Checks if an input node is inside of a list
def Check_List(new_node,list):
    for i in range(len(list)):
        if list[i][3] == new_node[3]:
            return True
        else:
            return False
       
# Updates pixels to white in image for searched nodes 
def UpdateSearched(node):
    flipped_image[250-node[3][0],node[3][1]]= [255,255,255]

# Updates pixels to green in image for goal nodes 
def UpdateGoal(node):
    flipped_image[250-node[3][0],node[3][1]]= [0,255,0]

# Calls cv.imshow to update image, scales image up 4X
def UpdateImage():
        resized = cv.resize(flipped_image,(1600,1000))
        cv.imshow("image",resized)
        cv.waitKey(1)

# Checks if node is not in obstacle space, has been searched, if lower cost found
# updates the cost respectively
def Check_Node(new_node,ClosedList,OpenList,c2c_node):
    if Check_List(new_node,ClosedList) == False and c2c_node[new_node[3]] != -1:
        if (Check_List(new_node,OpenList) == False or Check_List(new_node,OpenList) == None) and c2c_node[new_node[3]]==np.inf:
            c2c_node[new_node[3]] = new_node[0]
            UpdateSearched(new_node)
            hq.heappush(OpenList,new_node)
        else:
            if (new_node[0] < c2c_node[new_node[3]]):
                c2c_node[new_node[3]] = new_node[0]

# Finds a node_index within a list
def Find_Node(list,node_index):
    for i in range(len(list)):
        if list[i][1]==node_index:
            return list[i]

# Algorithm to generate goal path, takes parent nodes of goal location until
# backtracked to initial point, reverses stack and returns list of nodes to goal
def Backtrack(ClosedList):
    goal = ClosedList.pop()
    reverse = []
    reverse.append(goal)
    while reverse[-1][2] > 0:
        parent = Find_Node(ClosedList,reverse[-1][2])
        reverse.append(parent)
    route = []
    while reverse:
        UpdateGoal(reverse[-1])
        UpdateImage()
        route.append(reverse.pop())
    return route

# dijkstra algorithm that uses action sets to search for input goal location
def dijkstra(start_node,goal_node,c2c_node):
    goal_found = False
    #Open/Closed List creation
    OpenList = []
    ClosedList = []
    hq.heappush(OpenList,(0,1,0,start_node))  #Pushes starting node into OpenList
    iterator = 0
    while OpenList and goal_found == False:
        current_node = hq.heappop(OpenList)
        ClosedList.append(current_node)
        if current_node[3] == goal_node:
            goal_found = True
            print("Goal Found!")
            goal_route = Backtrack(ClosedList)
            return goal_route
        else:
            iterator = iterator + 1
            new_nodeE = MoveE(current_node)
            Check_Node(new_nodeE,ClosedList,OpenList,c2c_node)
            new_nodeS = MoveS(current_node)
            Check_Node(new_nodeS,ClosedList,OpenList,c2c_node)
            new_nodeW = MoveW(current_node)
            Check_Node(new_nodeW,ClosedList,OpenList,c2c_node)
            new_nodeN = MoveN(current_node)
            Check_Node(new_nodeN,ClosedList,OpenList,c2c_node)
            new_nodeSE = MoveSE(current_node)
            Check_Node(new_nodeSE,ClosedList,OpenList,c2c_node)
            new_nodeSW = MoveSW(current_node)
            Check_Node(new_nodeSW,ClosedList,OpenList,c2c_node)
            new_nodeNW = MoveNW(current_node)
            Check_Node(new_nodeNW,ClosedList,OpenList,c2c_node)
            new_nodeNE = MoveNE(current_node)
            Check_Node(new_nodeNE,ClosedList,OpenList,c2c_node)
            if iterator % 100 == 0: # Only updates image every 100 nodes for speed
                UpdateImage()
    if goal_found == False:
        print("No Goal Found!")
            
goal_route = dijkstra(start_node,goal_node,c2c_node)
print("Press '0' to exit")
cv.waitKey(0)
print("Runtime: ",time.time()-start_time, " seconds")