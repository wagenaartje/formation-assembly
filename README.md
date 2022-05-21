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

# EUR-FUCKING-KA
The problem seems to be that we do not vary the number of formations enough. We should do more localized training therefore instead.

So we cannot seem to bypass a very strong local minima, somehow. So can we just do the following: see if we can beat the 
simple algorithm using a pre-trained network? IT DOES!!! It goes to about 0.05, which is the best we can  do? not with sign...

# Create MWE example, and post. Ask why not working?

# Do fixed formation, for many agents.
