---
layout: post
title: "Learning Theano"
date: 2016-04-23
---

I've downloaded and installed the Theano machine learning library, and now I'm trying out a few examples. Here I am attempting to fit a line to a set of linear datapoints.

{% highlight python %}
import matplotlib.pyplot as plt
from theano import tensor as T
import theano
import numpy as np

def makeline(data):
    return 5 + (2 * data + np.random.randn(*data.shape) * 0.33)

train_x = np.linspace(0, 1, 100)
train_y = makeline(train_x)

test_x = np.linspace(1, 2, 100)
test_y = makeline(test_x)
{% endhighlight %}

Here I call all the imports, and set up the training and testing datasets with NumPy.

{% highlight python %}
def model(X, M, B):
    return (X * M) + B

X = T.scalar()
Y = T.scalar()

W0 = theano.shared(np.asarray(0, dtype=theano.config.floatX))
W1 = theano.shared(np.asarray(0, dtype=theano.config.floatX))

y = model(X, W0, W1)
{% endhighlight %}

Here we're setting up the model. Theano is interesting in that we build mathematical expressions first, and then evaluate them at a later time. Here I have defined X and Y as scalar variables. Then, I defined W0 and W1 to be two shared variables (both being initialized to zero). A shared variable must be initialized, and as such it can be evaluated immediately (W0.eval() would do the trick). On the other hand, since X and Y are variables we would first have to pass a value in for them to take before we could evaluate them. The same goes for any expression containing them. As you can see, the model in this case is a simple line: $$y=mx+b$$. W0 is the slope, and W1 is the y-intercept of this line.

{% highlight python %}
mse = T.mean(T.sqr(y - Y))

gradientW0 = T.grad(cost=mse, wrt=W0)
gradientW1 = T.grad(cost=mse, wrt=W1)

updates = [
  (W0, W0 - gradientW0 * 0.01),
  (W1, W1 - gradientW1 * 0.01)
]

train_model = theano.function(inputs=[X, Y], outputs=mse, updates=updates)
{% endhighlight %}

This is where the magic begins. First, I define the mean squared error function (mse). Then I define the gradients of W0 and W1. How? Scorcery! (Being able to do it in a single line of code each certainly *feels* like sorcery). T.grad(cost=mse, wrt=W0) returns the derivative of the mean squared error function with respect to W0. All you have to do is pass T.grad a function and a shared variable, and it figures the rest out for you.

updates is a list of tuples of length two. When we pass it into theano.function, we're telling it to update W0 to be W0 - gradientW0 * 0.01 whenever the function is called (and likewise for W1). With this, in a few simple lines of code, I have defined stochastic gradient descent!

{% highlight python %}
for i in range(100):
    for x, y in zip(train_x, train_y):
        train_model(x, y)

total_x = train_x + test_x
output = [model(X, M, B).eval({X: x}) for x in total_x]
plt.scatter(train_x, train_y, color='blue')
plt.scatter(test_x, test_y, color='green')
plt.scatter(total_x, output, color='red')
{% endhighlight %}

Here I train the model on each point of the training set 100 times over, and then print out the result. This is a animated gif showing the process:

<img src="https://github.com/lhannest/learningTheano/blob/master/linear_regression/images/animation.gif?raw=true" alt="hi" class="inline"/>
