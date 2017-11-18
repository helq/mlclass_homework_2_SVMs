# Preprocessing #

\subsection{first part}

\inlinetodo{Explain about how you left a computer processing for two days all kernel
operations for all strings, it was very expensive, but seemed to be worthy.}

\inlinetodo{Remember to try to apply the string kernel to lists and not only to str}

\inlinetodo{subsection, Training, testing and validation size datasets (look at first
paper)}

\inlinetodo{with a test size of 300 we get a boundary error of %7.8, awful but, it's
something (copy explanation from previous paper)}

<!--
   ->>> from math import log, sqrt
   ->>> err = 0.012
   ->>> 1/(2*err**2) * log(2/.05)
   -12808.60921567339 # size of test file if we wanted to the error to not change more than 1.2%
   ->>> N = 300
   ->>> sqrt( log(2/.05)/(2*N) )
   -0.07841002756996855 # error range :S
   -->

\inlinetodo{trying robust normalizing (ignoring "outliers") does not work better}

\missingfigure{how the value gamma affects the score (classification k-fold error) and the
number of support vectors}

\missingfigure{how nu affects the classification error and the number of sv}

\inlinetodo{explain why the grid search was not very useful, and how many characteristics
where tested}

\missingfigure{a graphic of classification error in the grid space}

\missingfigure{a graphic of support vectors in the grid space}

\subsection{second part}

<!--
   ->>> from math import log, sqrt
   ->>> N = 1640
   ->>> sqrt( log(2/.05)/(2*N) )
   -0.03353592655879196
   -->

<!-- vim:set filetype=markdown.pandoc : -->
