# Preprocessing #

## First Problem ##

The first dataset has a total of 2000 datapoints with 15 features (each feature is an
integer between 0 and 20 (inclusive)). The dataset is balanced, with 1049 datapoints
marked as 1 and 951 points as -1.

The training, test and validation datasets are divided into 68%, 15%, and 17% dataset
sets of the original, respectively. The training and validation datasets were used
together to train the final model, while the testing dataset was never used for anything
else more than measuring the accuracy of the final model.

With a test dataset of 300 points, we can ensure with a precision of 95% that the real
error measure, or better, than the real accuracy of the model is 7.8%, i.e., if the final
model has an error of 28.5% in the test dataset, we know with a precision of 95% that the
real error of the model lies between 20.7% and 36.3%.\footnote{
  we know this boundaries thanks to Chernoff equation. Below I copy an
  explanation of how to arrive at the values presented here with the use of Chernoff.

  Remember that the additive form of the Chernoff bounds are given by the equation:

  $$ P\left[\frac{1}{n} \sum_{j=1}^n X_j - p \geq \epsilon \right] \leq e^{-2 \epsilon^2 n} $$

  Given that we only have a dataset, we train only over a train set and not multiple, $n=1$.
  $X_j$ is the empirical error we get from the test set. And considering a confidence
  of $1-\delta$ ($\delta = e^{-2 \epsilon^2 n}$), we can rewrite the equation above as:

  $$ N \geq \frac{1}{2\epsilon^2} ln\left(\frac{2}{\delta}\right) $$

  where $N$ is the size of the test set and $\delta$ a value of our choosing.

  If we assume $\delta = 5\%$ (a confidence of 95\%), we know by solving the equation above
  that there is at most a ~3.64\% difference between the experimental classification error
  and the real error of the model when the model is tested using the test set, i.e., I know
  that if I get a 19\% classification error on the test set for a model, then this model has
  a real classification error that lies between 16\% and 22\%.
}

<!--
   ->>> from math import log, sqrt
   ->>> err = 0.012
   ->>> 1/(2*err**2) * log(2/.05)
   -12808.60921567339 # size of test file if we wanted to the error to not change more than 1.2%
   ->>> N = 300
   ->>> sqrt( log(2/.05)/(2*N) )
   -0.07841002756996855 # error range :S
   -->

I tried several different preprocessing strategies. All of the results by using each on of
them can be seen in \ref{training}. The preprocessing steps that I tried for this problem
were:

1. No preprocessing, the raw data was used directly in the models.
2. Scaling and Centering of the data.
3. Robust scaling and centering ("evading outliers", just like scaling and centering but
   the data that is used to determine the mean and the scaling are just those datapoints
   that lie between the first and the third quartiles, i.e., 50% of the data that is in
   the middle of the rest).
4. Normalizing (scaling each datapoint to the unit length).
5. Kernel PCA, using polynomial kernel with degree 2 and gamma of 2.2[^kpca]
6. Autoencoder, an arbitrary neural net that has (hopefully) the ability to reduce the
   dimensionality of the data from 15 features to 10.[^autoencoder]

[^kpca]: this parameters were selected by doing a grid search. With KPCA is not possible
  to determine the feature size output, like in regular PCA, therefore a search is
  necessary to find the parameters that gave the smallest feature space as output. The
  parameters which produced the smallest set of features after KPCA was applied were
  $degree = 2$ and $\gamma = 2.2$ with a polynomial kernel (here the polynomial kernel is
  defined by $kernel(x,y) = (\gamma(x*y))^d$).

[^autoencoder]: The space to search for an autoencoder is humongous, there are many
  different architectures to choose from, different layer dapths and types of activation
  functions. I just tested with a couple of architectures and setted to use a specific
  architecture arbitrarily. The architecture starts with a two layer shrinking phase, and
  grows to the original size of the input in one layer. All features are integers between
  0 and 20, thus the input and output of the network are one hot vectors.

## Second Problem ##

This dataset consists of 32603 datapoints. Each datapoint consists of a sequences of
characters forming a text in English. Each datapoint is labeled with one of the following
labels: "sports", "business", "entertainment", "us", "world", "health", or "sci&tech". My
task was to find a binary classificator to differenciate between labels "us", and "health"
and "entertainment", thus the final size of the dataset was of 9920 points, 4783 of which
were labeled as "us".

The training, test and validation datasets were divided into the sizes 6624, 1640 and
1656, respectively. These sizes correspond approximately to 4/6, 1/6 and 1/6 of the
original dataset size. Having similar sizes for test and validation makes it approximately
similar to talk about the precision and accuracy of the error obtained in the final models
and the ones found in k-fold crossvalidation.

With a size of 1640 datapoints, we can calculate using Chernoff that the maximum error we
can expect to see is about 3.35%[^chernoffeq].

[^chernoffeq]: $\epsilon \geq \sqrt{ \frac{\log(2/.05)}{2 N} }$ where $N = 1640$.

<!--
   ->>> from math import log, sqrt
   ->>> N = 1640
   ->>> sqrt( log(2/.05)/(2*N) )
   -0.03353592655879196
   -->

Two preprocessing procedures were applied to the dataset before passing it to learning:

1. No-preprocessing, the SSK routine was the only "intermideate" step in the learning procedure
2. Tokenization and Lemmatization of each sentence, instead of feeding strings to the
   SSK-SVM routine, each sentence was broken into tokens, the tokens were then
   "normalized"[^normtokens], SSK was performed on this list of normalized tokens. One
   list of tokens per sentence.

[^normtokens]: For this, I used the procedure `lemmatize` from the library NLTK
  [@BirdKleinLoper09] which takes a word and returns the root of the word (if found). This
  function uses the WordNet [@wordnet] dataset which contains thousands of words with
  their semantic meaning.

SVMs work on the power of the kernel trick, to make it efficient most implementations of
optimization libraries for SVMs precompute the Gramm matrix of the data used to optimize.
This is usually the first step on the optimization process. Unfortunately, creating a
gramm matrix using SSK is quite expensive, it is so expensive, that it takes often longer
to calculate the matrix than the optimization process.

I precomputed the gramm matrix for a range of lambda parameters. In this way I only passed
to the optimization subroutine the gramm matrix and no computation needed to be done. This
made the computation times way lower. Though, each computation of a Gramm Matrix (of size
$9920 \times 9920$)

<!-- vim:set filetype=markdown.pandoc : -->
