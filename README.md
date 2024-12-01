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

