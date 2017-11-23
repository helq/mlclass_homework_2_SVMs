# Appendix #

The second problem required the use of a kind of "obscure" kernel for strings, namely
String Subsequences Kernel (SSK) and the only implementation available was that of
shogun [@shogun_2017]. Shogun has one huge drawback, it is rather
un-pythonic, its API hasn't been designed to integrate cleanly with python idioms or ways
of passing around objects. Besides, it seems like shogun doesn't have best support for
installation in any platform, the best way to install it is by compiling it from source,
which I managed to do, but didn't feel as worthed to anyone else who wanted to use the
code I've written.

I, therefore, implemented the fast-SSK subroutine presented in @lodhi2002text. The naive
recursive version shown in the paper is rather slow, a very efficient version of the
algorithm can be written by transforming all recursions into loops, and managing caching
efficiently. In the end, after some optimizations I arrived to a similar perfromant
version of the algorithm developed in shogun, but I want to clarify two things: first, I
took a clever idea from the shogun implementation, and second, the algorithm is written in
Cython not python, the python version is hundreds of times slower than the Cython version,
this is given to all the stuff python does on runtime, checking the types and other
things.

The code to the implementation SSK can be found in <https://github.com/helq/python-ssk>
with CC0 license (Public Domain Dedication).

<!-- vim:set filetype=markdown.pandoc : -->
