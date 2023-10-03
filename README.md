# Python-PSO

This is a simple implementation of particle swarm optimization, a method for approximating the maximum/minimum of an n-dimensional, continuous function over a finite domain. This implementation does not rely on any differentiation of the function being optimized.

### A quick warning:
This does not guarantee that the global optimum is found, nor does it guarantee an exact solution. It is an apporoximation of the optimum over the specified domain.

### Another warning:
Increasing the depth when calling `minimze()` or `maximize()` makes the optimizer recursively search a smaller and smaller area, letting the user hone in on a more precise answer. However, there is a chance that the optimizer will find an incorrect minimum, and a depth greater than 1 will no longer provide a more accurate answer. Increase the depth only if you are satisfied with the general value of a depth 1 optimization.
