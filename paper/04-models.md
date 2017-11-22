# Models #

\subsection{First problem}

For the first problem, we should find the best parameters for a binary classificator
$\nu$-SVM with two different kernels: polynomial and gaussian.

I searched the spaces of (hyper-)parameters in grid fashion. In one axis, $\nu$ took the
values $\nu \in \{.2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, 8\}$. For the
polynomial kernel, the $degree$ paramater took the values $\{1, 2, 3, 4, 5, 6, 7\}$. And,
for the gaussian kernel, the $\gamma$ parameter took different range values depending on
the preprocessing the data passed through.

\subsubsection{Grid search and model selected for polynomial kernel}

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

\subsubsection{Grid search and model selected for gaussian kernel}

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

\subsection{Second Problem}

\inlinetodo{something to add here}

<!-- vim:set filetype=markdown.pandoc : -->
