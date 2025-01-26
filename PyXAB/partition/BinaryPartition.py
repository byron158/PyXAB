# -*- coding: utf-8 -*-
"""Implementation of Binary Partition
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

from PyXAB.partition.Node import P_node
from PyXAB.partition.Partition import Partition
import numpy as np
import copy
import pdb


class BinaryPartition(Partition):
    """
    Implementation of Binary Partition
    """

    def __init__(self, domain=None, node=P_node, rng_seed=None):
        """
        Initialization of the Binary Partition

        Parameters
        ----------
        domain: list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.
        """
        if domain is None:
            raise ValueError("domain is not provided to the Binary Partition")
        self.rng_seed = rng_seed
        super(BinaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        rng = np.random.default_rng(self.rng_seed) #if rng_seed is not provided, generator will use the default seed

        parent_domain = parent.get_domain()
        dim = rng.integers(0, len(parent_domain))
        selected_dim = parent_domain[dim]

        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest

class BinaryPartitionWithLimits(BinaryPartition):
    """
    Implementation of Binary Partition with limits on the minimum size of the partition and the minimum precision
    """

    def __init__(self, domain=None, node=P_node, rng_seed = None, min_precision=None):
        """
        Initialization of the Binary Partition

        Parameters
        ----------
        domain: list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.
        """
        if domain is None:
            raise ValueError("domain is not provided to the Binary Partition")
        self.rng_seed = rng_seed
        super(BinaryPartitionWithLimits, self).__init__(domain=domain, node=node)
        self.min_precision = min_precision
    

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        if not parent.is_splittable():
            return

        parent_domain = parent.get_domain()

        splittable_dims = []
        for i, dim_range in enumerate(parent_domain): #check all dimensions to see if they are splittable, it they aren't they are removed from the parameter selection list for this node
            if self.min_precision is not None and (dim_range[1] - dim_range[0]) < self.min_precision: #this checks if the range of the domain is above the minimum precision
                continue  # Skip dimensions that are smaller than min_precision
            else:
                splittable_dims.append(i) #add dimension to the list of splittable dimensions

        if not splittable_dims:
            #No dimensions are splittable; skip creating children
            parent.mark_unsplittable()
            print(f"No dimensions are splittable; skip creating children. Node point: {parent.get_cpoint()}")
            return

        rng = np.random.default_rng(self.rng_seed) #if rng_seed is not provided, generator will use the default seed
        dim = rng.choice(splittable_dims)
        selected_dim = parent_domain[dim]

        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest