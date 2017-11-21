# Models #

\subsection{First problem}

For the first problem, we should find the best parameters for a binary classificator
$\nu$-SVM with two different kernels: polynomial and gaussian.

I searched the spaces of (hyper-)parameters in grid fashion. In one axis, $\nu$ took the
values $\nu \in \{.2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, 8\}$. For the
polynomial kernel, the degree paramater took the values $\{1, 2, 3, 4, 5, 6, 7\}$. And,
for the gaussian kernel, the $\gamma$ parameter took different range values depending on
the preprocessing the data passed through.

\subsubsection{Grid search and model selected for polynomial kernel}

Remember from section \ref{preprocessing} (preprocessing), I tested 6 different
preprocessing strategies, below I present the analysis of each one of their results (using
the grid search):

\begin{itemize}
  \item[No-preprocessing:] The figure~\ref{poly-no-preprocessing_accuracy} shows the mean testing
  accuracy with variable values of $\nu$ and degree. The maximum testing accuracy is around
  the $\nu$ values of $[0.4,0.6]$, and degree $\in [1,4]$. In fact the maximum value is on
  $nu = 0.48$ and degree $= 2$ with a (mean) accuracy of $79.52\%$\footnote{This could be
  considered the baseline for this problem, now my purpose is to find a better model!}
\end{itemize}

![Testing accuracy in a grid search in the two dimensional space of $\nu \in [0.02,0,8]$ and degree $\in [1,7]$ \label{poly-no-preprocessing_accuracy}](imgs/poly-no-preprocessing_accuracy.pdf)

\inlinetodo{Each of the models required by the problem, problem one and problem two}

\inlinetodo{Results of the k-crossvalidation process, and models selected, lots of plots!!}

\inlinetodo{trying robust normalizing (ignoring "outliers") does not work better}

\missingfigure{how the value gamma affects the score (classification k-fold error) and the
number of support vectors}

\missingfigure{how nu affects the classification error and the number of sv}

\missingfigure{a graphic of classification error in the grid space}

\missingfigure{a graphic of support vectors in the grid space}

<!-- vim:set filetype=markdown.pandoc : -->
