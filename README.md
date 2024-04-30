# DeepLearningBasics
Home work assignment for deep learning class
This git repo contains 3 home work assignment for deep learning course.

## Simple fully connencted

## NLP sentiment prediction

## VAE
#### The assignment
Disentangled representations are defined as ones where a change in a single unit of the representation corresponds to a change in single factor of variation of the data while being invariant to others.
However, a large number of datasets contain inherently discrete generative factors which can be difficult to capture with these methods. In our dataset, distinct objects or entities would most naturally be represented by discrete variables, while their position or scale
might be represented by continuous variables.
#### The solution
This work is based off of "Learning Disentangled Joint Continuous and Discrete Representations". We decided to use the model, JointVAE, which seemed to fit our assignment.
In contrast to many supervised methods, the above paper focused on learning the latent representation of discrete features, in an unsupervised manner.
The framework they proposed is based on Variational Autoencoders (VAE), and therfore comes with the advatages of VAE which includes: stable training, large sample diversity and a principled inference network; while having the flexibility to model a combination of
continuous and discrete generative factors
