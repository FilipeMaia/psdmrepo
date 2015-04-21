
.. _tutorial:

################
 Tutorial
################

**************************
 Get/Build the Software
**************************

To get started with the software, make a newrelease directory and add the ParCorAna
package to it (Presently the package is not part of the release). For example, do
these commands::

  newrel ana-current parCorRel
  cd parCorRel
  sit_setup
  # now get a kerboses ticket to get code from the repository
  kinit   
  addpkg ParCorAna HEAD
  scons

**************************
 Make a Config File
**************************

One first has to make a configuration file. This is a python file
that defines two dictionaries::

  system_params
  user_params

system_params defines a number of keys that the framework uses. user_params 
is only for the user module, the framework simple passes user_params through to the user module.

The simplest way to do this is to copy the default config file from the ParCorAna.
From the release directory parCorRel where you have added the ParCorAna package, do::

  cp data/ParCorAna/default_params.py myconfig.py

Below we go over the most important parts of the config file. Full details are in the
comments of the config file you have copied. 

The config file starts by importing modules used later. In particular::

  import psana
  import ParCorAna

DataSource
=============

These paramters::

  system_params['dataset']   = 'exp=xpptut3:run=1437'
  system_params['src']       = 'DetInfo(XppGon.0:Cspad.0)'
  system_params['psanaType'] = psana.CsPad.DataV2

tell the frameowrk the dataset, source, and psana data type the framework needs to distribute to the workers.
The default values are for xpp tutorial data with full cspad. Note that because the config file import's psana,
it can use the psana type CsPad.DataV2. We will use these values to run the tutorial.

Mask File
===========
::

  system_params['maskNdarrayCoords'] = 'maskfile.npy' # not created yet

You need to provide the framework with a mask file for the detector data. This is a 
numpy array with the same dimensions as ndarray the psana calibration system uses to 
represent the detector. This is not neccessarily a 2D image. More on this below. 

Times, Delays
========================
::

  system_params['times'] = 50000
  system_params['delays'] = ParCorAna.makeDelayList(start=1,
                                                  stop=25000, 
                                                  num=100, 
                                                  spacing='log',  # can also be 'lin'
                                                  logbase=np.e)

For the tutorial, We will leave these alone. They specify a collection of 50,000 
events, and a set of 100 logarithmically spaced delays from 1 to 25,000.

User Code
========================
::

  import ParCorAna.UserG2 as UserG2
  system_params['userClass'] = UserG2.G2atEnd

The user_class is where users hook in their worker code. We will be using the example 
class in the ParCorAna package - G2atEnd does a simplified version of the G2 
calculation used in XCS.

User Color File
=======================
This is a parameter that the UserG2 needs - a color file that labels the detector pixels
and determines which pixels are averaged together for the delay curve. It bins the pixels
into groups. More on this in the next section::

  user_params['colorNdarrayCoords'] = 'colorfile.npy' # not created yet


***************************
 Create a Mask/Color File
***************************
The system requires a mask file that identifies the pixels to process. 
Reducing the number of pixels processed can be the key to fast feedback during an experiment.

The ParCorAna package provides a tool to make mask and color files in the numpy ndarray
format required. To read the tools help do::

  parCorAnaMaskColorTool -h

(Another tool to look at is roicon, also part of the analysis release). The command::

  parCorAnaMaskColorTool --start -d 'exp=xpptut13:run=1437' -t psana.CsPad.DataV2 -s 'DetInfo(XppGon.0:Cspad.0)' -n 300 -c 6

Will produce a mask and color file suitable for this tutorial::

  xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy  
  xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy 

Note that our input will be ndarr files, not image files. The mask file is only  0 or 1. It is 1
for pixels that we **INCLUDE**. The color file uses 6 colors (since we gave the -c 6 option to the tool. 
As an example, these colors bin pixels based on intensity. In practice users will want to bin pixels
based on other criteria.

Once you have modified these files, or produced similarly formatted files you are ready for the 
next step.

Add to Config
==================

Now in myconfig.py, set the mask and color file::

  system_params['maskNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy'
  user_params['colorNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy'

Note that the last parameter is to the user_params - the framework knows nothing about the coloring.

********************
Check Config File
********************

Once you have modified the config file, it is a good idea to check that it runs as python code, i.e, that
all the imports work and the syntax is correct::

  python myconfig.py

The config file does a pretty-print of the two dictionaries defined.

***********************************
Run Software 
***********************************

Now you are ready to run the software. To test using a few cores on your local machine, do::

  mpiexec -n 4 parCorAnaDriver -c myconfig.py -n 100

This should run without error. 

***********************************
Results
***********************************
You can get a listing of what is in the output file by doing::

  h5ls -r g2calc_xpptut13-r1437.h5

The h5 file contains two groups at the root level::

  /system
  /user

In /system, one finds::

  /system/system_params    Dataset 
  /system/user_params      Dataset
  /system/color_ndarrayCoords Dataset
  /system/mask_ndarrayCoords Dataset 

The first two are the output of the Python module pprint on the system_params and
user_params dictionaries after evaluating the config file.

The latter two are the mask and color ndarrays specified in the system_params.

In /user one finds whatever the user viewer code decides to write. The example 
UserG2 module writes, for example::

  /user/G2_results_at_539  Group
  /user/G2_results_at_539/G2 Group
  /user/G2_results_at_539/G2/delay_000001 Dataset {32, 185, 388}
  /user/G2_results_at_539/G2/delay_000002 Dataset {32, 185, 388}
  ...
  /user/G2_results_at_539/IF Group
  /user/G2_results_at_539/IF/delay_000001 Dataset {32, 185, 388}
  /user/G2_results_at_539/IF/delay_000002 Dataset {32, 185, 388}
  ...
  /user/G2_results_at_539/IP Group
  /user/G2_results_at_539/IP/delay_000001 Dataset {32, 185, 388}
  /user/G2_results_at_539/IP/delay_000002 Dataset {32, 185, 388}

