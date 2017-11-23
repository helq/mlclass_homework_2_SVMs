# Optimization/Training Algorithm #

I used the python library "scikit-learn" [@scikit-learn]. The library offers an simple
interface for SVM training and inference, unfortunately the types of kernels are
restricted to a couple, including polynomial and gaussian (RBF), but no string subsequence
kernel (SSK). I implemented the procedure to calculate the SSK, and used it with
"scikit-learn".

$\nu$-SVM was the version of SVM that I selected to solve the problem. It is easier to
think about a parameter $\nu$ restricted to $[0,1)$ than a parameter $C$ with only a
positivity restriction.

To find the best hyperparameters for the model, i.e., gamma and degree (for
RBF and polynomial kernels, respectively), a grid search was done with K-cross validation.
The results of the search and the precise parameters used in it can be found in the next
section.

To have a more stable estimation of the testing errors, I employed K-fold crossvalidation
with size of 5 for both problems.

<!-- vim:set filetype=markdown.pandoc : -->
