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

Resulting plan
- Train for far less steps, and evaluate fitness **improval** instead. 
- Do many evaluations per epoch. Every evaluation = one more formation changed

We can also look how we can combine inputs. so that we can keep number of evaluations low and
focus on long term behavior to beat the simple algorithm.

Notes:
- If no formation is given in inputs, they will all home to the same location so that relative distances become zero.