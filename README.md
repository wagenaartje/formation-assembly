First thing, we can backpropagate a network on manual control,
and then we can see if it evaluates correctly! If there is a
discrepancy... we can evolve afterwards. If not, super
tough problem.

--> extrapolate to larger groups!!!
--> make relative to group velocity
--> once working, use CMA-ES instead (or a variant, like Vlad supposes)

# Report tips
- Use neural networks visualization like in Croon's paper
- Mention number of parameters, number of dimensions, ranges etc.
- Compare to simple algorithm. Both average distance at end, plus time to certain radius!
- Do not plot convergence best fitness, but rather the average fitness of the best genome on 100 samples or so.
- Explain the sign issue: extremely similar inputs have to result in extremely different outputs (2 agents close to each other)
- "Asymmetric formation forming using identical agents: initial exploration"
- Sowieso assignment op novelty search doen. Proberen te combineren met CMA-ES: going in direction of novelty!

# EUR-FUCKING-KA
The problem seems to be that we do not vary the number of formations enough. We should do more localized training therefore instead.

So we cannot seem to bypass a very strong local minima, somehow. So can we just do the following: see if we can beat the 
simple algorithm using a pre-trained network? IT DOES!!! It goes to about 0.05, which is the best we can  do? not with sign...

# Create MWE example, and post. Ask why not working?

# Do fixed formation, for many agents.

# LAST PRIORITY:
- Investigate what causes simple algorithm to fail. 
-> identify issue, if possible, adapt input of neural network.
-> if issue is unsolvable, forget about it.

- also explain why standing still is not an option for the agents. which one is going to stand still?

# You can define a different simple solution: plot in mean frame, let each one go to a target. optimize so that
# maximum path length is shortest. it's a simple assignment problem. this is also how we could solve it.

# Breakthrough? I think if we properly define novelty search, we will get some breaktrhough
BREAKTHROUGH CONFIRMED