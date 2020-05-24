# arc-solver
This system was developed for the Abstraction and Reasoning Challenge (ARC) Kaggle competition.

## Overview
ARC is a novel benchmark for Artificial General Intelligence (AGI) systems [1]. It features 800 tasks in its
public data set where half are training and the other half are evaluation. Each task has a training and testing set of input and output pairs. The AGI system should attempt to learn from the training pairs and
submit a program that solves the testing pairs. The training set allows the AGI system to adjust
its ability to learn while the evaluation set independently measures these adjustments. Therefore, the
AGI system is learning how to learn. This general approach to learning underpins the purpose of ARC, which
is to steer attention away from creating task specific systems to general systems that are capable of learning many tasks within a given domain.

## Approach
The general approach to solving the ARC benchmark is currently thought to be a smart search over a space of tokens that make up a domain specific language (DSL). Throughout the competition this approach was met with opposition as some stated that the search space was too large or that the problem of designing a complete DSL is too difficult. I think this approach is commensurate with the composition like nature of ARC, but whether it is the approach the winners used is currently unknown.

## Implementation
I discovered this competition on March 15th, 2020, so I was a few months behind from the start. This did not discourage me in trying to place, but rather pushed me to work harder. Unfortunately, my efforts did not produce a system that I am ultimately proud of, but they did produce ideas that possess potential to be reused. My system is made up of the following components:
- Priors
  - Library of functions responsible for axiomatic knowledge. See Core Knowledge priors, III.1.2. [1].
- General tokens
  - Accept other tokens as input and compose them to solve the problem. The initial search space is derived by the cartesian product of the argument space, where each element in this set is a partially bound function. I call this a function space.
- Objective tokens
  - Accept objects as input and perform an action, such as shift until axis are equal. A subset of the arguments are used in creating a function space similar to general tokens.
- Interpreter
  - Provides a means of arbitrary orderings of tokens (i.e. programs). Also implements a program cache where the longest sub sequence will be replaced by a cached value.
- Evaluation Function
  - Measures the difference between the input and output image.
- Optimizer
  - Issues constraints on the initial function space with values from the current input/output pair. Searches the resulting space for a token that reduces the difference between input/output pair and continues until the difference is 0 or space is exhausted.
- System Cache
  - Since the system is value based, all function calls can be cached from all parts of the system. This greatly reduces overhead.

## Run instructions
- clone the ARC project https://github.com/fchollet/ARC
- Example execution:
  ```bash
  python3 main.py -a <arc path> -dp data/training -tp 1cae*.json
  ```

## References
[1] F. Chollet. On the Measure of Intelligence. arXiv:1911.01547, 2019.
