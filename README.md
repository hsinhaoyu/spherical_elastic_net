# An experiment in adapting the elastic net algorithm to the spherical surface

The elastic net is a numerical algorithm for approximating the solution to the traveling salesman problem (Durbin & Willshaw, 1987). It's an unsupervised learning algorithm that shares some of the features of Kohonen's self-organizing map. In neuroscience, it's widely used to model the development of topographic maps in the visual cortex (Drubin & Mitchison, 1990; Goodhill, 2007).

The standard elastic net algorithm is formulated such that the "prototypes" to be matched by the elastic net are distributed in Euclidean spaces. However, for applications involving the mapping between the cortical sheet and the retina, the spherical geometry of the retina should be taken into account. This is an implementation of the elastic net (using Tensorflow) for the spherical surface. It can be used as a research tool, or as an demonstration of using gradient descent in non-Euclidean spaces.

A video of the program in action is at: https://www.youtube.com/watch?v=uoZcTG_i2jQ&feature=youtu.be

# References
- Durbin R, Mitchison G (1990) A dimension reduction framework for understanding cortical maps. _Nature_ 343 644-647.
- Durbin R, Willshaw D (1987) An analogue approach to the travelling salesman problem using an elastic net method. _Nature_ 326 689-691.
- Goodhill GJ (2007) Contributions of theoretical modeling to the understanding of neural map development. _Neuron_ 56, 301-311.
