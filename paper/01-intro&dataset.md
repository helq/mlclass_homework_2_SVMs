<!--\listoftodos-->

# Introduction and Dataset #

<!--
   -\inlinetodo{Explain what is this paper about, the problems presented in the homework, and
   -the data}
   -->

In this report, I explore some of the things I tried to solve two problems given in the
course Machine Learning (IELE4014). The two problems presented for homework 2 are on
binary classification using Support Vector Machines (SVMs).

In the first problem, we are asked to find the classificator with the smallest error of
classification using two different kernels: polynomial, and gaussian (RBF). The input data
consists of 15 features with integer values between 0 and 20 (inclusive). 2000 datapoints
are given for training. Additionaly, we are given 2000 datapoints with no labels, and our
goal is to label each datapoint using the best classifiers we could find using polynomial
and gaussian kernels.

In the second problem, we are asked to find a classifier for short sequences of text
(the maximum length of the strings of text is 470 characters long, and the mean length is
around 150 characters). The main idea is to use the string subsequence kernel to classify
the data. A total of 9920 datapoints are given for the task.

For both problems, any preprocessing of the data is allowed.

The code implementing the training procedures, postprocessing and graphic analysis can be
found in <https://github.com/helq/mlclass_homework_2_SVMs>. I also implemented in
Cython[^cython] the fast-SSK procedure presented in @lodhi2002text.

[^cython]: Cython is python with C-like characteristics and the code gets compiled not
  interpreted.

<!-- vim:set filetype=markdown.pandoc : -->
