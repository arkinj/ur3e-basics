import numpy as np 
import pickle
import matplotlib.pyplot as plt

"""
Goal of this script is to calibrate the action-space into a physically realizable space
with the UR3e manipulator. The pickle file has stored data from diffusion policy runs, that can be 
used to map the space into a scaled space for demo. 

Actual space [0, 0]X[500,500] {[x_min,y_min]X[x_max,y_max]}. 100 units to 10cm, realizable
space will be [-25cm,0cm]X[25cm,50 cm]. Hopefully center at the robot elbow joint (?)
"""
def scale_action(action_np):
    action_scaled = action_np/(1000)
    action_scaled[:,0] = action_scaled[:,0]-0.25
    return action_scaled

def unscale_action(action_scaled):
    action_np = action_scaled
    action_np[:,0] = action_scaled[:,0]+0.25
    action_np = action_np * 1000
    return action_np

# maybe see mit_perception.transform_utils which extends this i think

def transform_action():
    #Load numpy array consisting of [X,Y] positions that the end-effector must reach 
    file = open('/home/realm/ur3e-basics/scripts/action_reference.pkl','rb')
    action_data = pickle.load(file)
    action_np = action_data['action']
    print(action_np)

    #Scale the action size
    # action_scaled = action_np/(1000)
    # action_scaled[:,0] = action_scaled[:,0]-0.25
    action_scaled = scale_action(action_np)
    return action_scaled, action_np

def visualize(action_scaled,action_np):
    #visualize scaled action space
    plt.scatter(action_scaled[:,0],action_scaled[:,1])
    plt.show()

    plt.figure()
    plt.scatter(action_np[:,0],action_np[:,1])
    plt.show()

if __name__ == '__main__':
    action_scaled, action_np = transform_action()
    breakpoint()