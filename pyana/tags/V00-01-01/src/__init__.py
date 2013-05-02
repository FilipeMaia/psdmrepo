#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 
"""
This package collects code which deals with various aspects of the analysis tasks.
"""

__all__ = ['event', 'histo', 'input', 'calib']


# Set of constants used by user modules to 
# control what framework does
Normal = 0
Skip = 1
Stop = 2
Terminate = 3
