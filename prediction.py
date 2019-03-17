#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
prediction.py: Link Prediction Network Model
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class prediction(nn.Module):
	def __init__(self, input_dim, output_dim):
		""" Init Predicter.

        @param 
        """
		super(prediction, self).__init__()
	    self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predict == nn.Sequential(
            # linear layer
            # sigmoid?
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )


    def forward(self, patent,p_citation ):
        """ Forward pass of predicter

        @param patent: tensor of integers
        @param p_citation: tensor of integers

        @returns prediction (int): prediction between 0 and 1 
        
        """
        p = self.features(patent)
        c = self.features(p_citation)
        x = torch.cat((p, c), 1)
        prediction = self.predict(x)

        return prediction



