Blockchain Enabled Multi-Agent Reinforcement Learning for Online Food Delivery with Location Privacy Preservation suggested the framework to delivery the food with different insitutations in the secure manner. 
The code has follwoing elements and components. 
![image](https://github.com/user-attachments/assets/052d09f6-eb04-4d1c-865e-b8c572443b49)


Blockchain:

Each agent encrypts its location using RSA encryption before adding the transaction to the blockchain.
The blockchain class handles transaction storage, block creation, and validation.
Location Privacy:

The agent’s location is encrypted before sending to the blockchain, ensuring privacy.
Multi-Agent System:

Agents are modeled as delivery vehicles with a grid-based environment, where they aim to minimize the distance to food locations.
Reinforcement Learning (QLearning):

Each agent learns the best action to take (move direction) based on its location and the location of the food using Q-learning.
Simulation Results:
The simulation prints the total rewards for each episode.
As agents interact with the environment, they learn the optimal routes, improve their policies, and validate their actions via the blockchain.
You can modify the grid size, the number of agents, or the number of episodes to simulate different scenarios.

Simulation code.
import numpy as np
import gym
from gym import spaces
import random
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import hashlib
import time

# Blockchain utility functions
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(previous_hash='1', proof=2)

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.transactions,
            'proof': proof,
            'previous_hash': previous_hash
        }
        self.transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, receiver, amount):
        self.transactions.append({
            'sender': sender,
            'receiver': receiver,
            'amount': amount
        })
        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    def validate_transaction(self, sender, location_encrypted):
        # Placeholder for blockchain validation logic
        print(f"Validating transaction: {sender} delivered at location {location_encrypted}")
        return True

# Location Privacy - Encrypting Location
def encrypt_location(location, public_key):
    cipher_rsa = PKCS1_OAEP.new(public_key)
    encrypted_location = cipher_rsa.encrypt(str(location).encode())
    return encrypted_location

def decrypt_location(encrypted_location, private_key):
    cipher_rsa = PKCS1_OAEP.new(private_key)
    decrypted_location = cipher_rsa.decrypt(encrypted_location).decode()
    return decrypted_location

# Multi-Agent Environment for Online Food Delivery
class FoodDeliveryEnv(gym.Env):
    def __init__(self, n_agents=3, grid_size=10):
        super(FoodDeliveryEnv, self).__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.agents_locations = np.random.randint(0, grid_size, size=(n_agents, 2))
        self.food_locations = np.random.randint(0, grid_size, size=(n_agents, 2))
        self.action_space = spaces.Discrete(4)  # 4 possible actions: Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(n_agents, 2), dtype=np.int32)
        self.blockchain = Blockchain()
        self.public_key, self.private_key = RSA.generate(2048).publickey(), RSA.generate(2048)

    def step(self, actions):
        rewards = []
        for agent_id in range(self.n_agents):
            action = actions[agent_id]
            x, y = self.agents_locations[agent_id]

            if action == 0:  # Up
                self.agents_locations[agent_id] = [max(0, x-1), y]
            elif action == 1:  # Down
                self.agents_locations[agent_id] = [min(self.grid_size-1, x+1), y]
            elif action == 2:  # Left
                self.agents_locations[agent_id] = [x, max(0, y-1)]
            elif action == 3:  # Right
                self.agents_locations[agent_id] = [x, min(self.grid_size-1, y+1)]

            # Reward calculation: closer to the food location
            food_x, food_y = self.food_locations[agent_id]
            distance = np.abs(self.agents_locations[agent_id] - np.array([food_x, food_y])).sum()
            reward = -distance  # Negative distance as a reward
            rewards.append(reward)

            # Encrypt the location and store transaction in blockchain
            encrypted_location = encrypt_location(self.agents_locations[agent_id].tolist(), self.public_key)
            self.blockchain.add_transaction(f"Agent {agent_id}", encrypted_location, reward)

        # Return observations and total reward
        return self.agents_locations, sum(rewards), False, {}

    def reset(self):
        self.agents_locations = np.random.randint(0, self.grid_size, size=(self.n_agents, 2))
        return self.agents_locations

# Q-learning for agents
class QLearningAgent:
    def __init__(self, n_actions):
        self.q_table = np.zeros((10, 10, n_actions))  # 10x10 grid for each agent
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Explore
        return np.argmax(self.q_table[state[0], state[1]])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.alpha * (reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action] - self.q_table[state[0], state[1], action])

# Simulating the environment
def run_simulation():
    env = FoodDeliveryEnv(n_agents=3)
    agents = [QLearningAgent(env.action_space.n) for _ in range(env.n_agents)]

    total_episodes = 2
    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = [agent.choose_action(state[agent_id]) for agent_id, agent in enumerate(agents)]
            next_state, reward, done, _ = env.step(actions)

            for agent_id, agent in enumerate(agents):
                agent.update_q_table(state[agent_id], actions[agent_id], reward, next_state[agent_id])

            total_reward += reward
            state = next_state

        print(f"Episode {episode+1}: Total Reward: {total_reward}")

if __name__ == '__main__':
    run_simulation()


