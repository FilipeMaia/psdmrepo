/*
 * This file exists exclusively for documentation and will be
 * processed by doxygen only. Do not include it anywhere else.
 */

/**
 * @defgroup Translator Translator package
 *  
 * @brief Package defining module for hdf5 translation.
 *  
 * Implements the module Translator.H5Output which is used to translate 
 * Psana events into an hdf5 files. This package is meant to 
 * replace the present hdf5 translation package O2OTranslator
 * 
 * The main reason for developing the new translator is to use 
 * DDL (the psddldata package) to generate code (in psddl_hdf2psana)
 * that stores the many xtc data types into hdf5 datasets.
 *
 * Developing the new translator as a psana module has also made 
 * it easy to support features such as event selection and translation
 * of user data.  In addition, as a psana module, this translator now shares 
 * all of psana's code for parsing xtc files.
 *
 * @see O2OTranslator psddldata psddl psddl_hdf2psana
 */
