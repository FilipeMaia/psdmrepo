/*
 * This file exists exclusively for documentation and will be
 * processed by doxygen only. Do not include it anywhere else.
 */

/**
@defgroup Translator Hdf5 Translator package
 
@section Introduction

The Translator Package implements the psana module H5Output. This translates Psana events into an hdf5 files. We shall call this system psana-translate.  psana-translate replaces o2o-translate.  o2o-translate is part of the package O2OTranslator which has been responsible for translating LCLS xtc files to hdf5 to date. The main reason for developing the new translator is to take advantage of DDL code generation for translating the many xtc data types into hdf5 datasets (implemented in the psddldata and psddl_hdf2psana packages). Developing the new translator as a psana module has also made it easy to support features such as event selection and translation of user data. In addition, as a psana module, this translator now shares all of psana's code for parsing xtc files.

Documentation can be found in the default_psana.cfg file in the data sub-directory of this package, or the data/Translator subdirectory of working SConsTools based release directory, as well as at the confluence page:

<a href="https://confluence.slac.stanford.edu/display/PSDM/The+XTC-to-HDF5+Translator">The XTC to HDF5 Translator</a>

@section RelatedPackages Related Packages
- O2OTranslator - original hdf5 translator
- psddldata - DDL definitions
- psddl_hdf2psana - extra DDL definitions for hdf5 schema's and generated code
- psddl - package that carries out code generation from DDL

*/
