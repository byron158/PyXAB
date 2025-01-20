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

    def __init__(self, domain=None, node=P_node, min_size=None):
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
        super(BinaryPartition, self).__init__(domain=domain, node=node)
        self.min_size = min_size
    

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

        parent_domain = parent.get_domain()

        if not parent.is_splittable():
            return

        splittable_dims = []
        for i, dim_range in enumerate(parent_domain): #check all dimensions to see if they are splittable, it they aren't they are removed from the parameter selection list for this node
            if self.min_size is None:
                splittable_dims.append(i)  # No size constraint, all dimensions are splittable
            else:
                mid_point = (dim_range[0] + dim_range[1]) / 2 
                if (mid_point - dim_range[0]) >= self.min_size and (dim_range[1] - mid_point) >= self.min_size:
                    splittable_dims.append(i)

        if not splittable_dims:
            #No dimensions are aplittable; skip creating children
            parent.mark_unsplittable()
            print(f"No dimensions are splittable; skip creating children. Node point: {parent.get_cpoint()}")
            return

        dim = np.random.choice(splittable_dims)
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
