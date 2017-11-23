# Training and Analysis #

## First Problem ##

### Polynomial Kernel ##

The testing accuracy for the model selected ($\nu = 0.62$ and $degree = 3$ with
normalization) is of 81.00%, a better value than 79.58% that was obtained with less
datapoints (validation datapoints), therefore it seems accurate to say that the real
accuracy value for the model is about 81.00%, but remember that this value is highly
expeculative because, as we know from subsection~\ref{first-problem} (Preprocessing /
First Problem), because with a test set of only 300 datapoints the estimation could be off
up to 7.8%. Therefore the real value lies between $[73.2\% , 88.8\%]$, but I am hopeful
that the actual accuracy will fall very close to 81.00% with an error of about 4% (not
7.8%).

### Gaussian Kernel ##

The testing accuracy for the model selected ($\nu = 0.52$ and $\gamma = 1.30 \times 10^{-5}$)
is of 80.67%, marginally better than the original model trained with less datapoints.
Given the size of the test set (300), the actual accuracy for the model lies between $[
72.87\%, 88.47\%]$, but given that the accuracy measure didn't change much with more data
I adventure to say that the error between the real accuracy and the one from the testing
data isn't more than 4%.

## Second Problem ##

The testing accuracy for the model ($\nu = 0.2$ and $\lambda = 0.55$) is $90.79\%$, which
is lower to the result gotten in the k-fold crossvalidation. The real accuracy value falls
between $[87.44\%, 94.14\%]$ with a confidence of 95%.

It is very interesting though, how well the binary classifier works with some
preprocessing. The preprocessing procedure reduces greatly the amount of work to compute
the kernels but it also gives better results than pure kernels in strings. This may be
because the lexemizer used groups many words by their semantic meanings, and that added
information helps the process of differenciating between two types of news greatly.

<!-- vim:set filetype=markdown.pandoc : -->
