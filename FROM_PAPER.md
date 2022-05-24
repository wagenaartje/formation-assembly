# Some things seen in the papers
### Novelty search
- Making a behavioral map
- Turn the archive into a queue instead
- We have an _unconstrained_ domain
- Distance should also be taken w.r.t. current population (otherwise you don't maintain diversity)
- You should use k-means averaging. Otherwise óne good individual will block progress.
- Only enter in archive above some $\rho_{min}$
- Using fitness as behavior is too conflating. Better to increase the behavioral space. We could first split into x- and y-components. Otherwise, distance (or components) per vehicle?
- We could probably have solved our issue with incremental evolution using different fitness functions, but this seems "wrong"
- The paper makes clear that _Novelty search effectively finds approximate solutions, while objective optimization is good for tuning approximate solutions_. So we need to use a standard fitness function for the final part. This is the incrementing approach, but it is also possible to do a multiobjective approach where fitness and behaviour are the two metrics.
- Using fitness as the behaviour method is very conflating, and is actually very similar to FUSS [Hutter and Legg 2006]. 
- The paper mentions that including the current population for comparison gives better results. 

### Paper by Guido
- Copy his paper format
- He only evolves 1 formation!! We have made it general!!
- He says it is a nontrivial task for most human designers, but there are simple algorithms out there. The reason it is non-trivial is due to combinatorial optimization. 
- He adds relative distances. We should look more into this, because the system of equations is not enough it seems to me. They even say distance is not redundant, but this is not the case.
- Maybe we should implement some distance constraint if everything works. 
- Maybe we should use roulette selection for mutation: higher chance of small probabilities.

### Other paper
- They punish being far away from center. But we simply define the outputs w.r.t. the center so we don't have drifting issues.
- they use Runge-Kutta integration instead of simple euler. 
- they _square_ the fitness
- in section 8, they mention this part about that the fitness per epoch is not defined w.r.t. the same starting positions, so there is some uncertainty. 
- they do thousands of simulations on the final best individuals
- just because the agents are identical, does not mean they are independent. if you keep 2 at the same position, solution might not converge. so not very robust I would say.