.. ParCorAna documentation master file, created by
   sphinx-quickstart on Thu Mar 19 11:32:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

############
 ParCorAna
############

This package provides a framework for doing cross 
correlation analysis on LCLS detector data. It is written in Python
and uses MPI to parallelize across pixels in the detector data.
It is designed around running the :ref:`g2` calculation, however
it can be adapted to parallelize any correlation that operates independently 
on each pixel in the detector.

Contents:

.. toctree::
   :maxdepth: 2

   overview
   tutorial
   framework
   testing
   userg2
   architecture

############
 Background
############

.. toctree::
   :maxdepth: 2

   g2

#################
 API Reference
#################

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

