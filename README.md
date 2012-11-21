miptree
=======

Maximum Inner Product Tree

Suppose you have a large collection of vectors and a query vector, and you want to find the top few
vectors in the collection which have the largest inner products with the query.  You can think of this
as a nearest neighbour search where the distance measure is the inner product, the catch being that
inner product isn't a proper distance metric as it doesn't satisfy the triangle inequality.  The MIP
Tree is a clever variant of the Ball Tree designed to solve this problem.  In practice it can be
several orders of magnitude faster than a linear scan over the entire collection.

This is a simple implementation based on P. Ram & A. Gray, [Maximum Inner-Product Search
using Tree Data-Structures](http://arxiv.org/pdf/1202.6101v1).


applications
------------

The MIP Tree solves the problem of generating recommendations from the user and item factors output
by most recent collaborative filtering algorithms.
