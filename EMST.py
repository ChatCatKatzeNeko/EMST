
'''
Author: Siyun WANG (wangsiyun6.15elsa@gmail.com)
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class EMST():
    '''
    create a Euclidean minimum spanning tree with a given dataset
    
    Class parametres
    ----------------
    X: numpy array or pandas dataframe of shape [nb_datapoints, nb_features]
    weight: array-like of shape [nb_datapoints,], optional, weight of each datapoint
    
    Class method(s)
    -------------
    fit(self): create the EMST using the Prim's algorithm
    
    Class attributes
    ----------------
    mst_pair_weight: dictionary. keys: tuple of vertices in the form of (parent,child); values: edge length
    mst_node_neighb: dictionary. keys: vertices; values: all its neighbors
    '''
    def __init__(self, X, weight=None):
        '''
        X: data array
        weight: positive float array-like of shape [nbData,], optional, default to None
                if given, it represents the weight of each data point; otherwise all points are equal
        '''
        
        self.data = X
        self.nbVertices = X.shape[0]
        self.weight = weight
    
    #=============
    # preparation
    #=============
    def getAdjMatrix(self, useMomentum=False):
        '''
        return the adjacency matrix from the data array
        useMomentum: bool, optional, default to False, only used when self.weight is not None.
                     if True, edge value is calculated in an angular momentum flavor, else the inverse of gravity
                     is given.
        '''
        self.M = np.zeros((self.nbVertices, self.nbVertices))
        
        # equi-weight
        # Euclidean distance is calculated for each pair of points
        if self.weight is None:
            for i in range(self.nbVertices):
                for j in range(i+1,self.nbVertices):
                    self.M[i,j] = np.linalg.norm(self.data[i]-self.data[j])
                    
        # weighted (!! NOT YET TESTED !!)
        # Here the weight of each point is taken into account by 
        # involving it into the calculation of the adjacency matrix
        else:
            #  1. Angular momentum
            #  we mimic the angular momentum of a pair of orbiting objects in a 2-body problem (here we are only
            #  interested in the angular momentum of the lighter body, in order to keep the adjacency matrix 
            #  symmetric):
            #         L = r*m*v 
            #  where
            #         r = r(M,m): radius of the orbiting object (the lighter) to the barycentre
            #         m: mass of the orbiting object
            #         v = v(M,m)
            #           ~ sqrt((M + m)/r): velocity of the orbiting object ("~" stands for "propotional to")
            #  simplifying the formula gives
            #         L ~ sqrt(r(M + m)) * m 
            if useMomentum:
                for i in range(self.nbVertices):
                    for j in range(i+1,self.nbVertices):
                        heavy = max(self.weight[i], self.weight[j])
                        light = min(self.weight[i], self.weight[j])
                        sumOfMass = heavy + light
                        r = np.linalg.norm(self.data[i]-self.data[j]) * heavy / sumOfMass
                        self.M[i,j] = (r * sumOfMass)**.5 * light
                    
            #  2. Inverse of gravity
            #  another way to take distance and weight into account is to use the inverse of gravity:
            #         G_inv ~ R^2 / (M*m)
            #  where 
            #         R = distance between the two object
            #         M,m: weight of the objects
            else:
                for i in range(self.nbVertices):
                    for j in range(i+1,self.nbVertices):
                        prodOfMass = self.weight[i] * self.weight[j]
                        R = np.linalg.norm(self.data[i]-self.data[j])
                        self.M[i,j] = R**2 / prodOfMass

        self.M += self.M.T
    
    
    #=================
    # helper function
    #=================
    def _minKey(self, key, inMSTSet):
        '''
        find the vertex with minmum distance value, from the complement of the mstSet
        key: array of positive real numbers, key values used to pick minimum weight edge
        inMSTSet: array of booleans, whether an element is in the mstSet
        '''
        m = np.inf
        for v in range(self.nbVertices):
            if (key[v] < m) & (~inMSTSet[v]):
                m = key[v]
                mInd = v
        return mInd
    
    
    #==============================
    # "main" function of the class
    #==============================
    def fit(self,useMomentum=False):
        self.getAdjMatrix(useMomentum)
        self.getEMST()
        
    def getEMST(self):
        '''
        construct the MST using Prim's algorithm
        '''
        # key values used to pick the vertex from the complement of mstSet to mstSet
        # key[node]: distance to the father node
        key = np.ones(self.nbVertices) * np.inf
        key[0] = 0
        
        # elements in the mstSet: mstSet[node] = father node
        mstSet = np.ones(self.nbVertices) * np.nan
        mstSet[0] = -1
        
        # boolean array indicating whether an element is in mstSet
        inMSTSet = np.zeros(self.nbVertices, dtype=bool)
        
        for _ in range(self.nbVertices):
            u = self._minKey(key, inMSTSet)
            
            # mark the vertex as a node of the MST
            inMSTSet[u] = True 
            
            for v in range(self.nbVertices):
                # update the key if
                #     u and v are adjacent vertices (self.M[u,v] > 0)
                #     v not in mstSet (~inMSTSet[v])
                #     distance between u and v is smaller than key[v] (key[v] > self.M[u,v])
                if (self.M[u,v] > 0) & (~inMSTSet[v]) & (key[v] > self.M[u,v]):
                    key[v] = self.M[u,v]
                    mstSet[v] = u
                 
        self.mst_pair_weight = self._makeMSTDictionary(mstSet)
     
    #=========================
    # formating, finalisation
    #=========================
    def _makeMSTDictionary(self, parent):
        '''
        store the MST in a dictionnary whose keys are pairs of neighbors and 
        whose values are the Euclidian length of the corresponding edge
        
        parent: list of integers, all parent nodes
        '''
        mst = {}
        for i in range(1, self.nbVertices):
            mst[(int(parent[i]), i)] = self.M[i, int(parent[i])]
        return mst
    
    def _transformTree(self, tree=None):
        '''
        create an other dictionary of the EMST whose keys are each node and values are lists of neighbors
        '''
        if tree is None:
            mst_pair_weight = self.mst_pair_weight
        else:
            mst_pair_weight = tree
            
        adjTree = np.zeros_like(self.M)
        for key in mst_pair_weight.keys():
            adjTree[key] = mst_pair_weight[key]
        
        adjTree += adjTree.T 
        
        self.mst_node_neighb = {}
        for i in range(self.M.shape[0]):
            self.mst_node_neighb[i] = np.where(adjTree[i])[0]
           
 

def plot2DTree(X, pairWeight):
    '''
    plot 2D trees
    '''
    plt.figure(figsize=(10,10))
    for tup in pairWeight.keys():
        plt.plot([X[tup[0],0],X[tup[1],0]], [X[tup[0],1],X[tup[1],1]], 'co-')
    plt.show()

def plot3DTree(X, pairWeight):
    '''
    plot 2D trees
    '''
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for tup in pairWeight.keys():
        ax.plot([X[tup[0],0],X[tup[1],0]], [X[tup[0],1],X[tup[1],1]], [X[tup[0],2],X[tup[1],2]], 'co-')
    plt.show()
