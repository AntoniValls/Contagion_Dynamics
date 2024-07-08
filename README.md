# Contagion_Dynamics

Contagion dynamics models fitted in the #TwitterMigration movement of 2022-2023 after Elon Musk's takeover.

Two **simple contagion models**, SIR and SIRS applied on two approaches:
- Microscopic approach: node based. Checking the state of all nodes each timestep.
- Macroscopic approach: solving the ODEs. It supposes that all nodes have the same probability to be infected at each timestep. No notion of the network's topology.

One **complex contagion model**, SI threshold model on the microscopic approach.

For fitting the models to the data I am using the scipy.optimize.minimize() function. It works perfectly for the macroscopic approach, but there should be something better for the microscopic one.
