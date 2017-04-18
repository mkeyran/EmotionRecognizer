#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:07:06 2017

@author: keyran
"""

import mappings
import numpy as np
import unittest

class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.data = np.load("test_data/milestones.npy")
        self.labels = np.load("test_data/labels.npy")
        self.normalize = mappings.NormalizeMapping()
        
    def test_normalize(self):
        new_data, new_labels = self.normalize.training_mapping(self.data, self.labels)
        self.assertTrue (new_data.shape == self.data.shape)
        self.assertTrue (new_data.max()<=0.5)
        self.assertTrue (new_data.min()>=-0.5)


class TestMirror(unittest.TestCase):
    def setUp(self):
        self.data = np.load("test_data/milestones.npy")
        self.labels = np.load("test_data/labels.npy")
        self.mirror = mappings.ImageMirrorMapping()
        self.normalize = mappings.NormalizeMapping()
        self.data, _ = self.normalize.training_mapping(self.data, self.labels)
        
    def test_mirror(self):
        new_data, new_labels = self.mirror.training_mapping(self.data, self.labels)
      #  import pdb; pdb.set_trace();
        self.assertTrue (new_data.shape[0] == self.data.shape[0]*2)
        self.assertTrue (all(new_labels[0:len(self.labels)] == self.labels))
        
if __name__ == '__main__':
    unittest.main()