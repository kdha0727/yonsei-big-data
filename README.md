---

# CSI4104 Big-Data class assignments, in Yonsei University.

Written by **Dongha Kim**.

---

# Assignment #1: MNIST

Implement/train/test image classifier on MNIST dataset

* Dataset: http://yann.lecun.com/exdb/mnist/

* Model: MLP (not CNN)

* Framework: Pytorch (or what you want)

* Include the comment/explanation of the model and code, and the model performance (loss and accuracy).

* Include the figure, where x-axis is the proportion (basically 25%, 50%, 75%, 100% but you can choice more) of used training data and y-axis is the model performance.

<a href="./Assignment%20%231%20MNIST%20code.py">
  <img src="http://ForTheBadge.com/images/badges/made-with-python.svg?style=for-the-badge&logo=Python" alt="Python Code" height="28" />
</a>
<a href="./Assignment%20%231%20MNIST%notebook.ipynb">
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange.svg?style=for-the-badge&logo=Jupyter" alt="Jupyter Code" height="28" />
</a>

---

# Assignment #2: Book Exercise

* See [Book Info](http://infolab.stanford.edu/~ullman/mmds/book.pdf).

---

# Assignment #3: Min-hashing

Implemented min-hashing.

1) Download Quora duplicate question dataset.

* This data consists of pairs of duplicate questions. Maybe you have log-in and follow a sort of procedure to get the dataset from the URL.

2) Make a corpus/shingling vocabulary/shingling-based matrix

* Randomly pick 10 pairs of duplicate questions, then list up the 20 questions as a question corpus (N=20).

* Then make document vectors based on the shingles.

3) Implement Min-hashing

* Make the signature matrix M. 

<a href="./Assignment%20%233%20Min-hashing%20code.py">
  <img src="http://ForTheBadge.com/images/badges/made-with-python.svg?style=for-the-badge&logo=Python" alt="Python Code" height="28" />
</a>
<a href="./Assignment%20%233%20Min-hashing%20notebook.ipynb">
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange.svg?style=for-the-badge&logo=Jupyter" alt="Jupyter Code" height="28" />
</a>

---

# Assignment #4: Sentiment analysis

Implemented text classifiers using RNN and LSTM which are able to achieve the sentiment analysis task.

The main purpose is to experience the quantitative (the effects of hyper-parameters) and qualitative (case study) analysis in text data. 

1) In-report explanation for code

2) Add table or chart to compare the model performance varying the hyper-parameters (size of hidden representation, size of layers)

3) Add your comments for two error cases: wrong prediction for positive review and wrong prediction for negative review.   

<a href="./Assignment%20%234%20Sentiment-analysis%20code.py">
  <img src="http://ForTheBadge.com/images/badges/made-with-python.svg?style=for-the-badge&logo=Python" alt="Python Code" height="28" />
</a>
<a href="./Assignment%20%234%20Sentiment-analysis%20notebook.ipynb">
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange.svg?style=for-the-badge&logo=Jupyter" alt="Jupyter Code" height="28" />
</a>

---
