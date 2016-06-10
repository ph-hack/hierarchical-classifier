# hierarchical-classifier

## Metric function idea
Create a dynamic metric function that learns weights for the features to manimize distances between samples of the same class and maximize distances between samples of different classes:
- create a class for this dynamic metric
- this would take any metric function as base
- and keep the record of the weights for each feature
- it needs a weight update (learning) algorithm (gradient descend perhaps)
