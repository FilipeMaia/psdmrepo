/*
 * This file exists exclusively for documentation and will be
 * processed by doxygen only. Do not include it anywhere else.
 */
	
/**
   @defgroup TimeTool TimeTool package
   
   @section Introduction
   
   Modules for analyzing recorded data from a timetool camera setup.
   The timetool camera measures the time difference between laser
   and FEL in one of two methods: (1) spatial encoding, where the
   X-rays change the reflectivity of a material and the laser probes
   that change by the incident angle of its wavefront; or (2) spectral
   encoding, where the X-rays change the transmission of a material
   and the chirped laser probes it by a change in the spectral components
   of the transmitted laser.
   
   @section TimeTool/src/Analyze.cpp
   A module that analyzes the camera image by
   projecting a region of interest onto an axis and dividing by a reference
   projection acquired without the FEL.  The resulting projection is
   processed by a digital filter which yields a peak at the location of
   the change in reflectivity/transmission.  The resulting parameters
   are written into the event.
   
   @section TimeTool/src/Check.cpp
   A module that retrieves results from the 
   event for either the above module or from data recorded online.
   
   @section TimeTool/src/Setup.cpp
   A module that calculates the reference
   autocorrelation function from events without FEL for use in the
   digital filter construction.
   
   @section TimeTool/data/timetool_setup.py
   A python script to calculate the digital filter weights.

   @section RelatedPackages Related Packages
   - psalg - computational routines used by the Analyze module

   @section Example Example Usage
   Spectral timetool camera data is available in xpptut13 run 178 and 179,
   though no signal is observable.
   An example configuration file can be used to process this data
   psana -c TimeTool/data/xpptut.cfg /reg/d/psdm/xpp/xpptut13/e308-r0179-s0*.xtc
*/
