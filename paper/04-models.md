# Models #

## First problem ##

For the first problem, we should find the best parameters for a binary classificator
$\nu$-SVM with two different kernels: polynomial and gaussian.

I searched the spaces of (hyper-)parameters in grid fashion. In one axis, $\nu$ took the
values $\nu \in \{$ 0.02,  0.04,  0.06,  0.08,  0.1 ,  0.12,  0.14,  0.16,  0.18, 0.2 ,
0.22,  0.24,  0.26,  0.28,  0.3 ,  0.32,  0.34,  0.36, 0.38,  0.4 ,  0.42,  0.44,  0.46,
0.48,  0.5 ,  0.52,  0.54, 0.56,  0.58,  0.6 ,  0.62,  0.64,  0.66,  0.68,  0.7 ,  0.72,
0.74,  0.76,  0.78 $\}$. For the
polynomial kernel, the $degree$ paramater took the values $\{1, 2, 3, 4, 5, 6, 7\}$. And,
for the gaussian kernel, the $\gamma$ parameter took different range values depending on
the preprocessing the data passed through.

### Grid search and model selected for polynomial kernel ###

Remember from section \ref{preprocessing} (preprocessing), I tested 6 different
preprocessing strategies, below I present the analysis of each one of their results (using
the grid search):

\begin{itemize}
  \item[No-preprocessing:] Figure~\ref{poly-no-preprocessing_accuracy} shows the mean
  validation accuracy with variable values of $\nu$ and $degree$. The maximum validation
  accuracy is around the $\nu$ values of $[0.4,0.6]$, and $degree \in [1,4]$. In fact the
  maximum value is on $\nu = 0.48$ and $degree = 2$ with a (mean) accuracy of
  $79.52\%$\footnote{This could be considered the baseline for this problem, now my
  purpose is to find a better model!}.

  Figure~\ref{poly-no-preprocessing_accuracy_std} shows the standard deviation (from the
  5-fold crossvalidation) for each $\nu$ and $degree$ in the grid search. The standard
  deviation is low ($<5\%$) for all values were the validation accuracy is high, i.e., for
  values with low mean validation accuracy ($\nu \in [0.02,0.22]$ and $degree \in [1,2]$)
  their standard deviation is very high, we shouldn't be too confident with those
  values.\footnote{Note: the standard deviation plots for all preprocessing procedures
  (except autoencoder) are very similar, thus I only show you one plot and not all of
  them}

  In Figure~\ref{poly-no-preprocessing_test-accuracy_errorbar}, we can see the mean
  validation accuracy with errorbars indicating the standard deviation of each value that
  $\nu$ takes. Notice how it is actually possible that the best value $\nu$ for the final
  model falls between $45\%$ and $80\%$.

  We can see in Figure~\ref{poly-no-preprocessing_support_vectors} the number of support
  vectors that the final model\footnote{remember that for the final model I used all
  training and validation datapoints, but not the test datapoints} has.  And as it is
  expected from theory, the number of support vectors increases as $\nu$ increases.
  Interestingly, the number of support vectors grows too with the $degree$ of the
  polynomial, though for big $degree$ values the number of support vectors keeps constant.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: No preprocessing \label{poly-no-preprocessing_accuracy}](imgs/poly-no-preprocessing_accuracy.pdf)

![Standard deviation of validation accuracy (using 5-fold crossvalidation) in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$  \label{poly-no-preprocessing_accuracy_std}](imgs/poly-no-preprocessing_accuracy_std.pdf)

![Validation accuracy for values of $\nu \in [0.02,0,8]$ and $degree = 2$ \label{poly-no-preprocessing_test-accuracy_errorbar}](imgs/poly-no-preprocessing_test-accuracy_errorbar.pdf)

![Number of support vectors for values of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$ \label{poly-no-preprocessing_support_vectors}](imgs/poly-no-preprocessing_support_vectors.pdf)

\begin{itemize}
  \item[Scaling:] As it can be seen in Figure~\ref{poly-scaling_accuracy}, the accuracy
  depends greatly on the $degree$ of the polynomial kernel! Accuracy is higher for odd
  $degree$ values. The highest mean accuracy is about $78.35\%$ with parameters $\nu = 0.64$
  and $degree = 1$, i.e., the best model using scaling is linear and it's not better than
  no preprocessing.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: Scaling \label{poly-scaling_accuracy}](imgs/poly-scaling_accuracy.pdf)

\begin{itemize}
  \item[Robust scaling:] As with scaling, the accuracy of the model heavily depends on
  $degree$, see Figure~\ref{poly-robust-scaling_accuracy}. The highest mean accuracy is
  about $78.41\%$ with parameters $\nu = 0.64$ and $degree = 1$, same as with regular
  scaling. The idea of robust scaling is to ignore datapoints that are far from the
  central cluster of data, i.e., outliers. No significative change can be seen between
  scaling and robust scaling.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: Robust scaling \label{poly-robust-scaling_accuracy}](imgs/poly-robust-scaling_accuracy.pdf)

\begin{itemize}
  \item[Normalizing:] As it can be seen in Figure~\ref{poly-normalization_accuracy}, the
  accuracy behaivor with normalization is very similar to the behaivor of no normalization
  at all. The highest mean accuracy is about $79.58\%$ with parameters $\nu = 0.62$ and
  $degree = 3$. Sadly, the best model using normalization isn't much better than the
  baseline, but the surface is smoother and the error doesn't grow much for big values of
  $\nu$ or $degree$
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: Normalization \label{poly-normalization_accuracy}](imgs/poly-normalization_accuracy.pdf)

\begin{itemize}
  \item[Kernel PCA:] Figure~\ref{poly-kernelPCA_gamma2.2_poly2_accuracy} shows the mean
  validation accuracy. The highest mean accuracy is about $79.53\%$ with parameters $\nu =
  0.48$ and $degree = 1$. Sadly, the best model using Kernel PCA isn't much better than
  the baseline. Interestingly, the behaivor of the accuracy for Kernel PCA depends, as
  scaling does, on the degree of the polynomial.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: Kernel PCA \label{poly-kernelPCA_gamma2.2_poly2_accuracy}](imgs/poly-kernelPCA_gamma2.2_poly2_accuracy.pdf)

\begin{itemize}
  \item[Autoencoder:] This was my last try in the search of a good preprocessing procedure,
  and it failed terribly. Figure~\ref{poly-autoencoder_accuracy} shows the mean validation
  accuracy. The highest mean accuracy is about $56.00\%$ with parameters $\nu = 0.78$ and
  $degree = 4$, i.e., no good model could be found when compressing the data from 15
  features to 10 using an autoencoder. Figure~\ref{poly-autoencoder_accuracy_std} shows the
  standard deviation for each 5-fold crossvalidation on the grid. All values for the
  standard deviation are low, which means that no matter which values of $\nu$ and
  $degree$ we use, we will always get very bad classification results.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$. Preprocessing step: Autoencoder \label{poly-autoencoder_accuracy}](imgs/poly-autoencoder_accuracy.pdf)

![Standard deviation of validation accuracy (using 5-fold crossvalidation) in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $degree \in [1,7]$  \label{poly-autoencoder_accuracy_std}](imgs/poly-autoencoder_accuracy_std.pdf)

The model selected with polynomial kernel is $\nu = 0.62$ and $degree = 3$ with a
preprocessing step of normalization.

### Grid search and model selected for gaussian kernel ###

Below I present the analysis of the different preprocessing strategies in the search of
the best classification:

\begin{itemize}
  \item[No-preprocessing:] Figure~\ref{rbf-no-preprocessing_accuracy} shows the mean
  validation accuracy with variable values of $\nu \in [0.02,0,8]$ and $\gamma \in
  \{$2.22e-06, 4e-06, 7.2e-06, 1.3e-05, 2.33e-05, 4.2e-05, 7.56e-05, 0.000136, 0.000245,
  0.000441, 0.000793, 0.00143, 0.00257, 0.00463, 0.00833, 0.015, 0.027, 0.0486, 0.0874,
  0.157$\}$. The maximum validation accuracy is around $81.12\%$ with values $\nu = 0.52$ and
  $\gamma = 1.30 \times 10^{-5}$. This validation accuracy is no much bigger than the
  baseline but it improves by more than $1\%$ the mean accuracy, something that couldn't
  be done with the polynomial kernel.

  Figure~\ref{rbf-no-preprocessing_accuracy_std} shows the standard deviation (from the
  5-fold crossvalidation) for each $\nu$ and $\gamma$ in the grid search. The standard
  deviation is low ($<5\%$) for all values were the validation accuracy is high, i.e., for
  values with low mean validation accuracy ($\nu \in [0.02,0.30]$ and $\gamma < 2.45 \times 10^{-4}$)
  their standard deviation is very high, we shouldn't be too confident with those
  values.

  We can see in Figure~\ref{rbf-no-preprocessing_support_vectors} the number of support
  vectors that the final model has. The paramater $\gamma$ influences greatly (more than
  what $degree$ influences the polynomial kernel) the number of support vectors, in fact,
  as it can be seen in Figure~\ref{rbf-no-preprocessing_accuracy_training}, for big values
  of $\gamma$ the model overfits. Notice the highest value on the validation set does not
  fall inside the overfitting zone, which is tranquilizing.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $\gamma \in [2.22\times{}10^{-6}, 1.57\times{}10^{-1}]$. Preprocessing step: No preprocessing \label{rbf-no-preprocessing_accuracy}](imgs/rbf-no-preprocessing_accuracy.pdf)

![Training accuracy in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $\gamma \in [2.22\times{}10^{-6}, 1.57\times{}10^{-1}]$. Preprocessing step: No preprocessing \label{rbf-no-preprocessing_accuracy_training}](imgs/rbf-no-preprocessing_accuracy_training.pdf)

![Standard deviation of validation accuracy (using 5-fold crossvalidation) in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $\gamma \in [2.22\times{}10^{-6}, 1.57\times{}10^{-1}]$  \label{rbf-no-preprocessing_accuracy_std}](imgs/rbf-no-preprocessing_accuracy_std.pdf)

![Number of support vectors for values of $\nu \in [0.02,0,8]$ and $\gamma \in [2.22\times{}10^{-6}, 1.57\times{}10^{-1}]$ \label{rbf-no-preprocessing_support_vectors}](imgs/rbf-no-preprocessing_support_vectors.pdf)

\begin{itemize}
  \item[Scaling:] Figure~\ref{rbf-scaling_accuracy} shows the mean validation accuracy,
  with $\gamma \in \{$ 4e-05, 7.2e-05, 0.00013, 0.000233, 0.00042, 0.000756, 0.00136,
  0.00245, 0.00441, 0.00793, 0.0143, 0.0257, 0.0463, 0.0833, 0.15, 0.27, 0.486, 0.874,
  1.57, 2.83 $\}$.
  The highest mean accuracy is about $80.65\%$ with parameters $\nu = 0.52$ and $\gamma = 0.00793$.
  The best model using scaling isn't much better than the baseline.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\gamma \in [4\times{}10^{-5}, 2.83$]. Preprocessing step: Scaling \label{rbf-scaling_accuracy}](imgs/rbf-scaling_accuracy.pdf)

\begin{itemize}
  \item[Robust scaling:] Figure~\ref{rbf-robust-scaling_accuracy} shows the mean validation accuracy,
  with $\gamma \in \{$ 2e-05, 3.6e-05, 6.48e-05, 0.000117, 0.00021, 0.000378, 0.00068,
  0.00122, 0.0022, 0.00397, 0.00714, 0.0129, 0.0231, 0.0416, 0.075, 0.135, 0.243, 0.437,
  0.787, 1.42 $\}$.
  The highest mean accuracy is about $80.71\%$ with parameters $\nu = 0.52$ and $\gamma = 0.00397$.
  The best model using scaling isn't much better than the baseline. Again as with
  polynomial kernels, robust scaling doesn't change the result of the validation accuracy.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\gamma \in [2\times{}10^{-5}, 1.42]$. Preprocessing step: Robust Scaling \label{rbf-robust-scaling_accuracy}](imgs/rbf-robust-scaling_accuracy.pdf)

\begin{itemize}
  \item[Normalization:] Figure~\ref{rbf-normalization_accuracy} shows the mean validation accuracy,
  with $\gamma \in \{$ 2.44e-05, 5.38e-05, 0.000118, 0.00026, 0.000573, 0.00126, 0.00277,
  0.0061, 0.0134, 0.0295, 0.0649, 0.143, 0.314, 0.691, 1.52, 3.35, 7.36, 16.2, 35.6, 78.4 $\}$.
  The highest mean accuracy is about $79.76\%$ with parameters $\nu = 0.56$ and $\gamma = 0.314$.
  The best model using scaling isn't much better than the baseline. Contrary to the
  polynomial kernel case, normalizing the data before feeding it to training results in
  not the best results.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\gamma \in [2.45\times{}10^{-5}, 78.4]$. Preprocessing step: Normalizing \label{rbf-normalization_accuracy}](imgs/rbf-normalization_accuracy.pdf)

\begin{itemize}
  \item[Kernel PCA:] Figure~\ref{rbf-kernelPCA_gamma2.2_poly2_accuracy} shows the mean validation accuracy,
  with $\gamma \in \{$ 1.11e-05, 2.44e-05, 5.38e-05, 0.000118, 0.00026, 0.000573, 0.00126,
  0.00277, 0.0061, 0.0134, 0.0295, 0.0649, 0.143, 0.314, 0.691, 1.52, 3.35, 7.36, 16.2,
  35.6 $\}$.
  The validation accuracy is always the same because it was not possible to learn from the
  data, this was a surprise to me, I still don't understand why it didn't learn anything.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\gamma \in [1.11\times{}10^{-5}, 35.6]$. Preprocessing step: Kernel PCA \label{rbf-kernelPCA_gamma2.2_poly2_accuracy}](imgs/rbf-kernelPCA_gamma2.2_poly2_accuracy.pdf)

\begin{itemize}
  \item[Autoencoder:] Figure~\ref{rbf-autoencoder_accuracy} shows the mean validation accuracy,
  with $\gamma \in \{$ 1.11e-05, 2.44e-05, 5.38e-05, 0.000118, 0.00026, 0.000573, 0.00126,
  0.00277, 0.0061, 0.0134, 0.0295, 0.0649, 0.143, 0.314, 0.691, 1.52, 3.35, 7.36, 16.2,
  35.6 $\}$.
  The highest mean accuracy is about $54.82\%$ with parameters $\nu = 0.76$ and $\gamma = 0.000118$.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\gamma \in [1.11\times{}10^{-5}, 35.6]$. Preprocessing step: Autoencoder \label{rbf-autoencoder_accuracy}](imgs/rbf-autoencoder_accuracy.pdf)

The model selected with polynomial kernel is $\nu = 0.52$ and $\gamma = 1.30 \times
10^{-5}$ with no preprocessing.

## Second Problem ##

As explained in subsection~\ref{second-problem} (Preprocessing / Second Problem), I
selected two preprocessing procedures. For each one of them I search in a grid fashion to
find the best parameters, $\nu$ and $\lambda$ for the problem[^preprocessing_footnote].

[^preprocessing_footnote]: Once I had computed all Gramm matrices for each $\lambda = \{$
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 $\}$, I just computed the resulting
  models of traninig SVMs on several different $\nu$ values.

The results of training the SVMs by preprocessing strategy:

\begin{itemize}
  \item[No preprocessing:] Figure~\ref{no-preprocessing_accuracy} shows the mean
  validation accuracy in the grid search. The highest accuracy value in the search
  corresponds to $89.73\%$ with the parameters $\nu = 0.26$ and $\lambda = 0.6$.

  The standard deviation for each 5-fold crossvalidation is very low, less than $0.5\%$, as
  it can be seen in Figure~\ref{no-preprocessing_accuracy_std}. As with the first problem,
  low values of $\nu$ and $\lambda$ makes for a not so good resulting model.

%  The number of support vectors increases as the value of $\lambda$ does, see
%  Figure~\ref{no-preprocessing_support_vectors}.

  \item[Tokenizing:] Figure~\ref{tokenized_leximized_accuracy} shows the mean
  validation accuracy in the grid search. The highest accuracy value in the search
  corresponds to $91.20\%$ with the parameters $\nu = 0.2$ and $\lambda = 0.55$. The
  behaivor of lists of tokens (one per word) is radically different to the behaivor of
  lists of characters, but their computation times are the same once the kernels have been
  computed.

  The standard deviation for each 5-fold crossvalidation is very-very low, less than
  $0.25\%$! see Figure~\ref{tokenized_leximized_accuracy_std}. The standard deviation with
  this preprocessing beats applying no preprocessing at all, this preprocessing procedure
  seems to be superior.
\end{itemize}

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\lambda \in [0.1, 1.0]$. Preprocessing step: No preprocessing \label{no-preprocessing_accuracy}](imgs/no-preprocessing_accuracy.pdf)

![Standard deviation of validation accuracy (using 5-fold crossvalidation) in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $\lambda \in [0.1, 1.0]$. \label{no-preprocessing_accuracy_std}](imgs/no-preprocessing_accuracy_std.pdf)

<!--
   -![Number of support vectors for values of $\nu \in [0.02,0,8]$ and $\lambda \in [0.1, 1.0]$. \label{no-preprocessing_support_vectors}](imgs/no-preprocessing_support_vectors.pdf)
   -->

![Validation accuracy in a grid search in the two dimensional spaces of $\nu \in [0.02,0,8]$ and $\lambda \in [0.1, 1.0]$. Preprocessing step: Tokenized and Leximized \label{tokenized_leximized_accuracy}](imgs/tokenized_leximized_accuracy.pdf)

![Standard deviation of validation accuracy (using 5-fold crossvalidation) in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and $\lambda \in [0.1, 1.0]$. \label{tokenized_leximized_accuracy_std}](imgs/tokenized_leximized_accuracy_std.pdf)

The model selected corresponds to the one obtained applying $\nu = 0.2$ and
$\lambda = 0.55$ with Tokenizing and Lemmatizing as preprocessing.

<!-- vim:set filetype=markdown.pandoc : -->
