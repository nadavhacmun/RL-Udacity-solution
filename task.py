import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pose = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward.
        target_vector = self.target_pose - self.sim.pose[:3]
        target_unit_vector = target_vector / np.sqrt(sum(target_vector ** 2))
        
        velocity_vector = self.sim.v
        velocity_unit_vector = velocity_vector / np.sqrt(sum(velocity_vector ** 2))
        
        reward = np.dot(target_unit_vector, velocity_unit_vector)
        if self.sim.v[2] >= 9.81:
            reward *= 2
        
        #distance = np.sqrt(sum((self.target_pose - self.sim.pose[:3]) ** 2))
        #reward += 1/distance
        #if np.sqrt(sum(self.sim.v**2)) > 5:
        #    reward -= sigmoid(np.sqrt(sum(self.sim.v**2)))
        
        return reward"""
        """distance = np.sqrt(sum((self.target_pose - self.sim.pose[:3]) ** 2))
        reward = (1 / distance) * 5
        reward -= np.tanh(abs(self.sim.pose[0])) + np.tanh(abs(self.sim.pose[1]))
        
        return reward"""
        """reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pose)).sum()
        return reward"""
        """distance = np.sqrt(sum((self.target_pose - self.sim.pose[:3]) ** 2))
        reward = -distance
        return reward"""
        """distance = np.sqrt(sum((self.target_pose - self.sim.pose[:3]) ** 2))
        reward = np.tanh(distance-10)
        reward += np.tanh(self.sim.v[2])
        return reward"""
        """reward = self.sim.pose[2] + 1
        if self.sim.pose[2] > 10:
            reward -= self.sim.pose[2]
        reward -= np.tanh(abs(self.sim.pose[1]) + abs(self.sim.pose[0]))
        return reward"""
        """reward = abs(self.sim.pose[:3] - self.target_pose).sum()
        return -reward"""
        if self.sim.pose[2] < 10:
            reward = np.tanh(self.sim.pose[2])
        elif self.sim.pose[2] > 10 and self.sim.pose[2] < 20:
            reward = np.tanh(20 - self.sim.pose[2])
        else:
            reward = -0.2 
        reward -= np.tanh((abs(self.sim.pose[1]) + abs(self.sim.pose[0])))
        return reward
            
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
def sigmoid(x):
    return 1/(1+np.exp(-x))