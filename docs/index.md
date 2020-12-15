# Linear Models with Python

## by Julian Faraway


- [Published by CRC Press](https://www.routledge.com/Linear-Models-with-Python/Faraway/p/book/9781138483958)
- [Python scripts](pyscripts/) of the code from each chapter
- [Python package](https://pypi.org/project/faraway/)

## Preface

This is a book about linear models in Statistics. A linear model describes a quantitative
response in terms of a linear combination of predictors. You can use a linear model
to make predictions or explain the relationship between the response and the predictors.
Linear models are very flexible and widely used in applications in physical
science, engineering, social science and business. Linear models are part of the core
of Statistics and understanding them well is crucial to a broader competence in the
practice of statistics.

This is not an introductory textbook. You will need some basic prior knowledge of
statistics as might be obtained in one or two courses at the university level. You
will need to be familiar with essential ideas such as hypothesis testing, confidence
intervals, likelihood and parameter estimation. You will also need to be competent in
the mathematical methods of calculus and linear algebra. This is not a particularly
theoretical book as I have preferred intuition over rigorous proof. Nevertheless,
successful statistics requires an appreciation of the principles and it is my hope
that the reader will absorb these through the many examples I present.

This book is written in three languages: English, Mathematics and
Python. I aim to combine these three seamlessly to allow coherent
exposition of the practice of linear modeling. This requires the
reader to become somewhat fluent in Python. This is not a book about
learning Python but like any foreign language, one becomes proficient
by practicing it rather than by memorizing the dictionary. The reader
is advised to look elsewhere for a basic introduction to Python but
should not hesitate to dive into this book and pick it up as you go. I
shall try to help. See the appendix to get started.

This book has an ancestor entitled [Linear Models with R](https://julianfaraway.github.io/faraway/LMR/). Clearly
the book you hold now is about Python and not `` but it is not an exact translation. Although
I was able to accomplish almost all of the `R` book in this Python book, I found reason for
variation:

- Python and `R` are similar (at least in the way they are used for Statistics) but they
  make different things easy and difficult. Hence it is natural to flow along the Python path
  for easier ways to accomplish the same tasks.

- Python is multi-talented but `R` was designed to do Statistics. `R` has a very large library
  of packages for statistical methods while Python has comparitively few. This has restricted
  the choice of methods I have presented in this book. One might expect the statistical functionality
  of Python to grow over time.

If your sole objective is to do Statistics, `R` is more attractive. Yet there are several reasons
why you might prefer Python. You may already know Python and use it for other tasks. Indeed it would
be unusual for someone to solely do Statistics. The data in this text is already clean and ready to use.
In practice, this is rarely the case and flexible software for obtaining and manipulating data is essential.
You may already be using Python for this purpose.

Python also has a place at the heart of Machine Learning (ML) but this is a
book about Statistics rather than ML. But the aims of these two
disciplines overlap considerably to the extent that any data analyst
should become familiar with the ideas and methods of both. The
datasets in this text are small by ML standards. I hope that a reader
coming to this book from an ML background would learn new statistical
perspectives on learning from data.

This book would not have been possible without several key open source
Python packages.  I thank the authors and maintainers of these
packages for their outstanding work.
