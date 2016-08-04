---
layout: post
title: "Math Test"
date: 2016-04-23
---

When $a \ne 0$, there are two solutions to \(ax^2 + bx + c = 0\) and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

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
