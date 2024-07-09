# Contagion_Dynamics

Contagion dynamics models fitted in the #TwitterMigration movement of 2022-2023 after Elon Musk's takeover.

Different approaches:
- **SIR and SIRS in Microscopic approach with simple contagion**: node based. Checking the state of all nodes each timestep and infecting by probability.
- **SIR and SIRS in Macroscopic Mean-Field approach with simple contagion**. Solving the ODEs. It supposes that all nodes have the same probability to be infected at each timestep. No notion of the network's topology.
- **SI in Microscopic approach with threshold contagion**. Checking the state of all nodes each timestep and infecting if the fraction of infected neighbours is surpasses a threshold.

For fitting the models to the data I am using the scipy.optimize.minimize() function. It works perfectly for the macroscopic approach, but there should be something better for the microscopic one. 

TODO: code AMEs for simple and threshold models.
