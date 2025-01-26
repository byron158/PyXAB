# -*- coding: utf-8 -*-
"""Implementation of SequOOL (Bartlett, 2019)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT

import math
import random
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
from PyXAB.partition.BinaryPartition import BinaryPartition, BinaryPartitionWithLimits
import pdb


class SequOOL_node(P_node):
    """
    Implementation of the SequOOL node
    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the SequOOL node
        Parameters
        ----------
        depth: int
            fepth of the node
        index: int
            index of the node
        parent: 
            parent node of the current node
        domain: list(list)
            domain that this node represents
        """
        super(SequOOL_node, self).__init__(depth, index, parent, domain)

        self.rewards = []
        self.mean_reward = 0
        self.opened = False

    def update_reward(self, reward):
        """
        The function to update the reward list of the node
        
        Parameters
        ----------
        reward: float
            the reward for evaluating the node
        
        Returns
        -------
        
        """
        self.rewards.append(reward)

    def get_reward(self):
        """
        The function to get the reward of the node

        Returns
        -------
        
        """
        return self.rewards[0]

    def open(self):
        """
        The function to open a node
        
        Returns
        -------
        
        """
        self.opened = True

    def not_opened(self):
        """
        The function to get the status of the node (opened or not)

        Returns
        -------
        
        """
        return False if self.opened else True

class SequOOL(Algorithm):
    """
    The implementation of the SequOOL algorithm (Barlett, 2019)
    """

    def __init__(self, n=1000, domain=None, partition=BinaryPartition, rng_seed=None):
        """
        The initialization of the SequOOL algorithm
        
        Parameters
        ----------
        n: int
            The totdal number of rounds (budget)
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm
        """
        super(SequOOL, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SequOOL_node, rng_seed=rng_seed)
        self.iteration = 0

        self.h_max = math.floor(n / self.harmonic_series_sum(n))
        self.curr_depth = 0
        self.loc = 0
        self.open_loc = 0
        self.chosen = []

    @staticmethod
    def harmonic_series_sum(n):
        """
        A static method for computing the summation of harmonic series
        
        Parameters
        ----------
        n: int
            The number of terms in the summation
        
        Returns
        -------
        res: float
            The sum of the series
        """
        res = 0
        for i in range(1, n + 1):
            res += 1 / i
        return res

    def pull(self, t):
        """
        The pull function of SequOOL that returns a point in every round
        
        Parameters
        ----------
        time: int
            time stamp parameter
        
        Returns
        -------
        point: list
            the point to be evaluated
        """
        node_list = self.partition.get_node_list()
        self.iteration = t

        if self.curr_depth <= self.h_max:
            if self.curr_depth == 0:
                node = node_list[0][0]
                if node.get_children() is None:
                    self.partition.make_children(
                        node, newlayer=(self.curr_depth >= self.partition.get_depth())
                    )
                if self.loc < len(node.get_children()):
                    if self.loc == len(node.get_children()) - 1:
                        self.loc = 0
                        self.curr_depth += 1
                        self.budget = math.floor(self.h_max / self.curr_depth)
                        self.chosen.append(node.get_children()[-1])
                        self.curr_node = node.get_children()[-1]
                        return node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.chosen.append(node.get_children()[self.loc - 1])
                        self.curr_node = node.get_children()[self.loc - 1]
                        return node.get_children()[self.loc - 1].get_cpoint()
            else:
                max_value = -np.inf
                max_node = None
                num = 0
                for i in range(len(node_list[self.curr_depth])):
                    node = node_list[self.curr_depth][i]
                    if node.not_opened():
                        num += 1
                        if node.get_reward() >= max_value:
                            max_value = node.get_reward()
                            max_node = node

                if max_node.get_children() is None:
                    self.partition.make_children(
                        max_node,
                        newlayer=(self.curr_depth >= self.partition.get_depth()),
                    )
                if self.loc < len(max_node.get_children()):
                    if self.loc == len(max_node.get_children()) - 1:
                        max_node.open()
                        self.loc = 0
                        self.budget -= 1
                        if self.budget == 0 or num == 1:
                            self.curr_depth += 1
                            self.budget = math.floor(self.h_max / self.curr_depth)
                        self.curr_node = max_node.get_children()[-1]
                        self.chosen.append(max_node.get_children()[-1])
                        return max_node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.curr_node = max_node.get_children()[self.loc - 1]
                        self.chosen.append(max_node.get_children()[self.loc - 1])
                        return max_node.get_children()[self.loc - 1].get_cpoint()
        else:
            self.curr_node = node_list[0][0]
            return node_list[0][0].get_cpoint()

    def receive_reward(self, t, reward):
        """
        The receive_reward function of SequOOL to obtain the reward and update Statistics
        
        Parameters
        ----------
        t: int
            The time stamp parameter
        reward: float
            The reward of the evaluation
        
        Returns
        -------
        
        """
        self.curr_node.update_reward(reward)

    def get_last_point(self):
        """
        The function to get the last point in SequOOL
        
        Returns
        -------
        point: list
            The output of the SequOOL algorithm at last
        """
        max_node = None
        max_value = -np.inf

        for node in self.chosen:
            if node.get_reward() >= max_value:
                max_node = node
                max_value = node.get_reward()

        return max_node.get_cpoint()

class SequOOLWithLimits(SequOOL):
    """
    The implementation of the SequOOL algorithm (Barlett, 2019), but with additional constraints on the minimum size of the domain and minimum precision
    """
    
    def __init__(self, n=1000, domain=None, partition=BinaryPartitionWithLimits, rng_seed=None, min_precision = None):
        """
        The initialization of the SequOOL algorithm
        
        Parameters
        ----------
        n: int
            The totdal number of rounds (budget)
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm
        """
        super(SequOOLWithLimits, self).__init__(n, domain, partition)
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SequOOL_node, rng_seed=rng_seed, min_precision=min_precision)
        self.iteration = 0

        self.h_max = math.floor(n / self.harmonic_series_sum(n))
        self.curr_depth = 0
        self.loc = 0
        self.open_loc = 0
        self.chosen = []

    def pull(self, t):
        """
        The pull function of SequOOL that returns a point in every round
        
        Parameters
        ----------
        time: int
            time stamp parameter
        
        Returns
        -------
        point: list
            the point to be evaluated
        """
        node_list = self.partition.get_node_list()
        self.iteration = t
        #### WARNING: This is where the code deviates significantly from the original
        # The original code used an if statement here, but we use a while loop 
        while self.curr_depth <= self.h_max: #check if the current depth is less than the maximum depth
            if self.curr_depth == 0: #check if the current depth is 0, which is the base of the binary tree
                node = node_list[0][0] #gets the root node, base of binary tree
                
                if node.get_children() is None: #checks if root node has children
                    self.partition.make_children( #makes children for the root node if it does not have any
                        node, newlayer=(self.curr_depth >= self.partition.get_depth())
                    ) #note that make_children triggers unsplittable = True if node can not be split further due to min size

                if not node.is_splittable(): #checks if node is unsplittable
                    self.curr_node = node #sets the current node to the unsplittable node
                    return node.get_cpoint() #returns the center point of the unsplittable node
        
                if self.loc < len(node.get_children()): #if the current location of sequool is less than the node's children (exploring the root node's children)
                    if self.loc == len(node.get_children()) - 1: #if the current location is the last child of the node
                        self.loc = 0 #reset the location tracker
                        self.curr_depth += 1 #increment the current depth (move to a deeper layer of the tree) (this moves the state machine to the next else block)
                        self.budget = math.floor(self.h_max / self.curr_depth) #update the budget
                        self.chosen.append(node.get_children()[-1]) #append the last child to the chosen list (for tracking chosen points)
                        self.curr_node = node.get_children()[-1] #set the current node to the last child
                        return node.get_children()[-1].get_cpoint() #return the center point of the last child
                    else:
                        self.loc += 1 #increment the location tracker (keep exploring the root node's children)
                        self.chosen.append(node.get_children()[self.loc - 1]) #add the loc's node to the chosen list
                        self.curr_node = node.get_children()[self.loc - 1] #set the current node to the loc's node
                        return node.get_children()[self.loc - 1].get_cpoint() ##return the center point of the loc's node
            else:
                max_value = -np.inf
                max_node = None
                num = 0

                for i in range(len(node_list[self.curr_depth])): #iterates through the nodes in the current depth
                    node = node_list[self.curr_depth][i] #gets the node at the current depth and index i
                    if node.not_opened() and node.is_splittable(): #Checks if node is unopened/unexplored and if it is splittable
                        num += 1 #increment the number of nodes that are both unopened and is splittable
                        if node.get_reward() >= max_value: #pick the node from list that has the highest reward; promotes exploring the maximum reward first
                            max_value = node.get_reward() #update the max value
                            max_node = node #set max node to the node with the highest reward

                #If no splittable or open nodes found, move up one depth level
                if max_node is None:
                    self.curr_depth -= 1  #Backtrack to the previous level
                    if self.curr_depth == 0: #if reached depth 0, reched root node
                        self.curr_node = node_list[0][0]  #Go back to the root node
                        return node_list[0][0].get_cpoint() #Return the center point of the root node
                    self.budget = math.floor(self.h_max / self.curr_depth) #if curr_depth is valid, reset budget to new depth
                    continue  #Continue searching at a shallower level

                if max_node.get_children() is None: #if the max node does not have children
                    self.partition.make_children( #make children for the max node
                        max_node,
                        newlayer=(self.curr_depth >= self.partition.get_depth()),
                    )#note that make_children triggers unsplittable = True if node can not be split further due to min size

                if not max_node.is_splittable(): #skip this node if it was not splittable, because it will not have children
                    continue  

                if self.loc < len(max_node.get_children()): #if the current location is less than the max node's children
                    if self.loc == len(max_node.get_children()) - 1: #if the current location is the last child of the max node
                        max_node.open() #open the max node, indicating that we explored all of its children
                        self.loc = 0 #reset the location tracker
                        self.budget -= 1 #decrement the budget
                        if self.budget == 0 or num == 1: #if the budget is 0 or there is only one node to explore
                            self.curr_depth += 1 #increment the current depth
                            self.budget = math.floor(self.h_max / self.curr_depth) #update the budget
                        self.curr_node = max_node.get_children()[-1] #set the current node to the last child of the max node
                        self.chosen.append(max_node.get_children()[-1]) #add the last child to the chosen list
                        return max_node.get_children()[-1].get_cpoint() #return the center point of the last child
                    else:
                        self.loc += 1 #increment the location tracker
                        self.curr_node = max_node.get_children()[self.loc - 1] #set the current node to the loc's child
                        self.chosen.append(max_node.get_children()[self.loc - 1]) #add the loc's child to the chosen list
                        return max_node.get_children()[self.loc - 1].get_cpoint()#return the center point of the loc's child
        else:
            self.curr_node = node_list[0][0]
            return node_list[0][0].get_cpoint()

