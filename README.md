# MAML
This code uses pytorch framework
Subset of MNIST to implement MAML

# MAML Algorithm
Given a set of tasks T = {T1, T2, …, TN}, where each task Ti has a training set Di, MAML aims to find a set of parameters θ that can be quickly adapted to new tasks.

1. Initialization: Initialize the model parameters θ randomly or with pre-trained weights.
2. Inner loop: For each task Ti, compute the adapted parameters θi by taking a few gradient steps on the loss function L(Di, θ) using the training data Di.
3. Outer loop: Update the initial parameters θ by taking the gradient descent step on the meta-objective J(T, θ) over all tasks. This objective measures the performance of the adapted parameters θi on the validation set for each task. Different meta-objectives can be used, such as minimizing the average loss or maximizing the accuracy across tasks.
4. Repeat steps 2 and 3 for a few iterations to refine the initial parameters
