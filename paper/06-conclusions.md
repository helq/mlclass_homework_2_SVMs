# Conclusions #

The two problems were easy to solve but required big amounts of computing time to find the
right models, the right (hype-)parameters. This is specially true for the second problem,
which required storing in memory almost a gigabyte, and took 14 hours to compute a single
gramm matrix to feed into an SVM optimizer.

SVMs are very powerful, but they require the hard task of finding the right kernel for a
task, the more complex a kernel is the more information it has about a specific problem,
but the harder it gets to compute, and therefore the computation time gets higher.

At first SVMs seemed mystical, those things that the industry used to use heavily, but are
they hard to use? Not really, many packages for machine learning come with them included,
and they are quite fast to calculate for small amounts of data (few thousend). Their major
drawback is that computing using big amounts of data seems to be a little hard, or at
least at little annoying.

<!-- vim:set filetype=markdown.pandoc : -->
