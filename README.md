# 15418-Project-Miracle-Gro
Miracle-Gro Project for Fast Parallel Training and Inference of Random Forest Models

# Milestone Report
<a href="15418-project-milestone-report.pdf">Milestone Report PDF</a>

# URL
[Miracle-Gro Project Page](https://dinodeep.github.io/15418-Project-Miracle-Gro/)

Raw URL: [https://dinodeep.github.io/15418-Project-Miracle-Gro/](https://dinodeep.github.io/15418-Project-Miracle-Gro/)

# Summary
We implemented a version of the Random Forest Machine Learning Model that is implemented in  Python's `sklearn`'s library in C++ and parallelized it using the 8-core GHC machines. We parallelized the sequential algorithm using OpenMP's task construct to create parallel work both across decision trees in the forest as well as within a given decision tree.

# Background

The machine learning application that we have parallelized is the Random Forest machine learning algorithm. This algorithm is an ensemble-based supervised algorithm that trains multiple independent decision tree models on bootstrapped subsets of the original training dataset. The algorithm is trained on a dataset that includes dataset entries that have features as well as a label, hence supervised learning. The goal is to be able to accurately predict the label for a new data point. 

This random forest classifier is effectively a list of independent binary trees that have data and decisions stored in their child nodes. The random forest data structure primarily consists of two main functions in its API which are described below

- `fit(data, labels)` -> None;

    data is a $N \times M$ matrix where there are $N$ samples in the dataset and there are $M$ features per sample. Furthermore, labels are $N \times 1$ matrix which is simply the label for each sample. This function modifies the data structure to allocate decision trees and train them by finding nodes of best split. Some of the computationally expensive portions of this function are finding the best split to split a non-leaf node at which requires iterating throughout the dataset multiple times for multiple different potential splits. Furthermore, there are opportunities for parallelism which will be described below.

- `predict(data)` -> predictions;

  data is an $N \times M$ matrix which we are trying to perform prediction with using a trained random forest classifier. The output predictions are an $N \times 1$ matrix containing the output predictions per sample. The expensive portions of predictions require iterating through the tree and finding which leaves in the tree the samples map to which is not as parallelizable because each move left or right in the tree on the path to a node is dependent on the prior decision along the path.

For demonstration, we ran our random forest algorithm on [Firat University's Internet Firewall Dataset](https://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data) which contains 11 different features and 65532 samples, each classifying to one of 4 classes. The features and classes are listed below.

- Features: source port, destination port, NAT source port, NAT destination port, action, bytes, bytes sent, bytes received, packets, elapsed time (sec), pkts\_sent, pkts\_received
- Classes: allow, take action on, drop, or reset-both

We've limited the dataset size from 65532 to 1024 samples in order to maintain reasonable training times using the current implementation of our algorithm. This would allow us to experiment with new parallelization methods as well in a more efficient manner.

During prediction, the predicted samples are passed through each of the trees and their results are combined to get an expected output for the given input. The high-level algorithm for generating the random forest classifier can be found below. 

Furthermore, we have allowed for additional hyperparameters in training on randomized tree algorithm by allowing the depth of the tree to be greater than 1 which allows us to explore more areas to parallelize throughout the algorithm. 

![Random Forest Algorithm](rf-algo.png) 

Source: [University of Wisconsin-Madison](https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/ensembles/RandomForests.pdf)

![Find Best Feature to Split on for a Given Node](psuedo-best-split.png) 

There are a number of avenues of improvement that can exploit the potential for parallelism in this code. For example, each tree in the random forest (individual decision tree) can be generated in parallel. Furthermore, within each tree, the various branches of the decision tree can be generated in parallel as well because they are independent regions of code. Additionally, when we are calculating the best split for a given node, that requires iterating over all possible features in our data and considering some set of splits for that feature. Then, we compute the gini-impurity score over all possible splits. Thus, to find the best split, calculating the gini-impurity score for all possible splits is independent of each other. As a result, the algorithm for training the random forest model has model parallelism that can be exploited to a high degree. When providing a prediction, the result is dependent on the predictions from each of the individual decision trees and therefore there are dependencies in the `predict` part of the algorithm. One important aspect to note is that this algorithm is recursive due to the recursive training algorithm of decision trees.

With respect to the various traits of synchronization and communication, the primary dependencies in the training algorithm consist of computing the best split before determining the data that is to be sent to the nodes that are deeper in the tree. However, as mentioned before, due to the independent possible splits and the independent leaf nodes, we can perform the calculation of these objects in parallel. Due to the recursive and conditional nature of training the decision trees, there is little data-parallelism, and instead, model-parallelism is more prevalent to this algorithm. As a result, this algorithm is not necessarily amenable to SIMD execution, and furthermore, there is not as much locality in this algorithm. However, the focus on using shared-memory parallelism allows to us ensure that communication is not as expensive as other parallelism methods such as SIMD and message passing parallelism.

# The Challenge

This problem can be challenging to parallelize because there is sequential dependencies when training individual decision tree models for the overall random forest classifier. Furthermore, because there are a lot of different spaces for parallelism, there is likely a balance on what sections of the code to parallelize with the limited resources that we have access to when running our code on parallel machines. 

The primary dependencies in the code are mostly related to the need to implement the prior nodes in the decision tree algorithms for later nodes in the list. Another important difficulty is that we need to determine the best splits for each node when performing the algorithm which reduces the amount of time. Furthermore, when creating a single decision tree, there is significant computation required to determine the best split for the tree.

One property of the system that makes this workload challenging is that there is a lot of shared memory that is being used between different trees and various nodes within trees, and as a result, this can make it difficult to make sure that there arent' any issues with have race conditions on shared memory. As a result, we must make sure to be extra careful to avoid these kinds of issues.

# Resources

We will use the gates machines to train our models. We will probably not use any type of starter code. Our plan is to first write an implementation of the random forest algorithm in C++. In order to write this implementation, we will reference implementations in Python from packages such as scikit-learn. The reference can be seen here: [sklearn Documentation](https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/ensemble/\_forest.py#L1081). 

# Goals and Deliverables

## What We Plan to Achieve
The following are the goals that we definitely want to complete during the timeline of this project as well as what we define as a successful project. 
- Write a working implementation of the Random Forest Algorithm in C++.
- Profile the performance of the sequential algorithm to determine where most of the time is spent in the program to determine what components of the Random Forest training and prediction algorithsm to most heavily focus on efficient parallelization.
- Parallelize this algorithm using OpenMP and perform experiments to analyze speedup and limitations to speedup in the original implementation. 


## What We Hope to Achieve
We hope to achieve the following given that our prior portions of the project work out well and allow us the time to pursue other goals
- Implement the parallel decision tree algorithm in other parallel programming models and potentially using hybrid programming models to improve performance such as mixing MPI across trees and OpenMP within trees to perform more efficient training and prediction.

## Poster Session Demo
We plan to demo our algorithm at the poster session where we will train our Random Forest algorithm on various datasets using our sequential implementation and then show the performance improvements using the our parallel version of the algorithm. We will show the comparable accuracies of the sequential and parallel algorithms which should show similar performance; however, we will show plots displaying how the parallel algorithm has strong speedup. Furthermore, we will generate plots on how speedup for training and inference time change with the number of processors for our workloads and show that our algorithms significantly improve the speedup.

## System Capability and Desired Performance
While this is not exactly a systems project, we hope to see that this algorithm will train models with similar accuracy compared to those the models trained sequentially; and furthermore, we hope to reach near-linear speedup for the performance of our parallel algorithm in comparison to the sequential algorithm.


# Platform Choice
Our platform for developing our algorithm is the Bridges-2 machines and the Gates cluster machines. This makes sense because they allow us to perform shared memory parallelism that allows us to parallelize the random forest algorithm without having to worry about the difficulties of message-passing memory models which is much more complex for recursive algorithms like decision trees which the random forest is made up of. Furthermore, we will be using the C++ programming language to implement our models because the C++ programming language is efficient, and it allows us to write implementations in OpenMP, allowing us to take advantage of the parallel programming mdoels that we have been using in class. Furthermore, we will implement a sequential version of the random forest algorithm in C++ so that we are not unfairly comparing the performance of Python's `sklearn` implementation (which has overhead from Python) with a purely C++ based model.

# Schedule

- [DONE]: Week of 4/2-4/8: Start on implementation of Random Forest in C++. Reference source code from \texttt{sklearn}. 
- [DONE]: Week of 4/9-4/15: Finish writing implementation of Random Forest. Start on profiling of the sequential implementation. Perform experiments to show what parts of the code are the slowest and where there is the most room for improvement with parallelism. 
- [In Progress]: By Midway 4/19: Finish sequential implementation and profiling. (Deep)
  - [Done]: By 4/17: finish writing the sequential implementation and finding dataset for performing profiling (Deep)
  - [In Progress]: By 4/19: complete profiling of the sequential profiling and determine the slow portions of training the random forest model and define specific conditions for experimentation with parallelized version.
- [Not Started]: Week of 4/16-4/22: Start optimizing sequential implementation using OpenMP. (Meher)
  - [In Progress]: By 4/20: complete an initial parallelization of the random forest training process by training trees in parallel
  - [Not Started]: By 4/22: improve the parallelization by performing fine-grained parallelized training by parallelizing a finer task in the training process that is expensive (determined by profiling the sequential implementation) and by parallelizing the prediction of the random forest.
- [Not Started]:  Week of 4/23-4/29: Perform further experiments to compare the new parallel implementation to the original sequential implementation. (Deep)
  - Not Started: By 4/26: Complete experiments comparing the speedups of training the random forest model's sequential implementation versus the parallel implementation on various dataset, and then, compare the results using different parallelized components.
  - Not Started: By 4/29: Accumulate the results into a writeup and compare the results by running the experiments with a higher core count on the PSC machines.
- [Not Started] Week of 4/30-5/4: Put together final deliverables and prepare for final demo. (Meher)
  - Not Started: Begin generating plots describing the speedup of the parallel version over the sequential version for both training and prediction of the random forest model using various datasets. Furthermore, find demo that can be run during the poster session that describes the performance improvements and trade-offs considered in training the parallel random forest in comparison to the sequential one.
  - Not Started: Produce final poster and confirm the demo that will be run during the poster session.
