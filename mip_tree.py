"""
Use a ball tree to find vectors maximising inner product
with a query efficiently.

Based on P. Ram & A. Gray, Maximum Inner-Product Search
using Tree Data-Structures, 2012.
"""

import random
import numpy as np
from operator import itemgetter

def d2(S,x):
    D = S - np.tile(x,(len(S),1))
    return sum((D*D).T)

class Node(object):
    
    def __init__(self,S,indices):
        self.S = indices
        self.mu = np.mean(S[indices],axis=0)
        self.R = max(d2(S[indices],self.mu))**0.5
        self.left = None
        self.right = None

    def dump(self,char,level):
        for x in self.S:
            print '{0}{1}:{2}'.format(' '*level,char,x)
        if self.left:
            self.left.dump('L',level+1)
        if self.right:
            self.right.dump('R',level+1)

    def get_ordering(self,indices):
        if self.left and self.right:
            self.left.get_ordering(indices)
            self.right.get_ordering(indices)
        else:
            indices.extend(self.S)

    def set_ordering(self,index_map):
        self.S = [index_map[ix] for ix in self.S]
        if self.left and self.right:
            self.left.set_ordering(index_map)
            self.right.set_ordering(index_map)

    def scan(self,vecs,q):
        """compute dot products for all the vectors
           indexed by this node"""
        d = [(ix,q.dot(vecs[ix])) for ix in self.S]
        return d

class Query(object):

    def __init__(self,q,k):
        self.q = q
        self.norm = sum(q**2)**0.5
        self.l = -np.inf
        self.k = k
        self.nn = []
        self.scanned = 0

    def dot(self,v):
        return self.q.dot(v)

    def update(self,d):
        self.scanned += len(d)
        self.nn.extend(d)
        self.nn.sort(key=itemgetter(1),reverse=True)
        self.nn = self.nn[:self.k]
        self.l = self.nn[-1][1]             
    
class MIPTree(object):
    """Maximum Inner Product Ball Tree"""

    def __init__(self,S,N0=20,reorder=False):
        """Construct a tree from a dataset

        S        numpy array of data vectors
        N0       the maximum size of each leaf node
        reorder  if True reorder the rows of S to optimise match time,
                 the reordered row indices will be stored in the indices field

        Reordering ensures that data is arranged serially in each leaf node
        which should reduce memory access time - you are only likely to see
        a benefit for very large data arrays.
        """
        self.N0 = N0
        self.root = Node(S,np.arange(len(S)))
        self.build(S,self.root)
        if reorder:
            self.indices = self.get_ordering()
            # now map these to dense range
            index_map = dict((ix,i) for i,ix in enumerate(self.indices))
            self.set_ordering(index_map)
            # indices in each leaf node are arranged serially, reorder the data to match
            S.data = S[self.indices].data
        self.S = S

    def match(self,query,k):
        """Search the tree for vectors maximising inner product with query

        query  the query vector
        k      number of best matches to return

        Returns the matches found as a list of pairs (index into S, inner product),
        and the number of vectors for which the inner product had to be computed.
        """
        q = Query(query,k)
        self.search(q,self.root)
        return q.nn,q.scanned

    def dump(self):
        """Print a primitive visualization of the tree"""
        self.root.dump('',0)

    def make_split(self,S):
        x = random.choice(S)
        A = S[np.argmax(d2(S,x))]
        d2A = d2(S,A)
        B = S[np.argmax(d2A)]
        dd = d2A - d2(S,B)
        return A,B,dd

    def build(self,S,node):
        if len(node.S) > self.N0:
            A,B,dd = self.make_split(S[node.S])
            node.left = Node(S,node.S[dd<=0])
            node.right = Node(S,node.S[dd>0])
            self.build(S,node.left)
            self.build(S,node.right)

    def mip(self,q,node):
        return q.dot(node.mu) + q.norm*node.R

    def search(self,q,node):
        if q.l < self.mip(q,node):
            if node.left is None and node.right is None:
                d = node.scan(self.S,q)
                q.update(d)
            else:
                left = self.mip(q,node.left)
                right = self.mip(q,node.right)
                if q.l < left or q.l < right:
                    if left < right:
                        self.search(q,node.right)
                        if q.l < left:
                            self.search(q,node.left)
                    else:
                        self.search(q,node.left)
                        if q.l < right:
                            self.search(q,node.right)

    def get_ordering(self):
        indices = []
        self.root.get_ordering(indices)
        return indices

    def set_ordering(self,index_map):
        self.root.set_ordering(index_map)

def main():

    import numpy as np
    from operator import itemgetter

    d = 5
    n = 10000
    X = np.random.randint(1,1000,(n,d))

    print 'building tree...'
    N0 = 20
    t = MIPTree(X,N0=N0,reorder=True)

    print 'running trials...'
    ntrials = 5000
    pr = []
    for i in xrange(ntrials):
        if i and i%1000 == 0:
            print i
        q = np.random.randint(1,1000,(1,d))[0]

        nn,scanned = t.match(q,3)
        pr.append(float(scanned)/len(X))

        if i%100 == 0: # check some cases for correctness
            tree = X[[j for j,_ in nn]]
            brute_force = sorted([(x,x.dot(q)) for x in X],key=itemgetter(1),reverse=True)[:3]
            for tt,bb in zip(tree,[x for x,_ in brute_force]):
                if not ((tt== bb).all() or tt.dot(q) == bb.dot(q)):
                    raise RuntimeError(' '.join(map(str,(tt,bb,tt.dot(q),bb.dot(q)))))

    print 'scanned',np.mean(pr)

if __name__ == '__main__':

    main()
