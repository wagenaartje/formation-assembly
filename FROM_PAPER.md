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
- The paper makes clear that _Novelty search effectively finds approximate solutions, while objective optimization is good for tuning approximate solutions_. So we need to use a standard fitness function for the final part. 