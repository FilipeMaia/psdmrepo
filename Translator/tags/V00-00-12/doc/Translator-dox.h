/*
 * This file exists exclusively for documentation and will be
 * processed by doxygen only. Do not include it anywhere else.
 */

/**
@defgroup Translator Hdf5 Translator package
 
@section Introduction

The Translator Package implements the psana module H5Output. This translates Psana events into an hdf5 files. We shall call this system psana-translate (and it can be run via a command line wrapper by that name).  psana-translate is meant to replace o2o-translate.  o2o-translate is part of the package O2OTranslator which has been responsible for translating LCLS xtc files to hdf5 to date. The main reason for developing the new translator is to take advantage of DDL code generation for translating the many xtc data types into hdf5 datasets (implemented between the psddldata, and psddl_hdf2psana packages). Developing the new translator as a psana module has also made it easy to support features such as event selection and translation of user data. In addition, as a psana module, this translator now shares all of psana's code for parsing xtc files.

Documentation on O2OTranslator, which discusses some history with regards to selecting hdf5 for a scientific data format for general use can be found

- <a href="https://confluence.slac.stanford.edu/display/PCDS/Translator">o2o-translate</a>
- <a href="https://confluence.slac.stanford.edu/display/PCDS/User+Interface+to+Translator">using o2o-translate</a>

This documentation also contains important links to the <a href="https://confluence.slac.stanford.edu/display/PCDS/Interface+Controller">Interface Controller</a> which is the simplest way to run translation to hdf5.

@section UsersGuide User's Guide

This user's guide covers the following topics:

- @ref InputOutput 
- @ref Running
- @ref NewFeatures
- @ref PsanaCfg
- @ref TranslationDamage
- @ref DroppedFeatures
- @ref Speed
- @ref DroppedFeatures
- @ref TechnicalDifferences



@section InputOutput Input and Output Data Formats

A discussion of the input and output formats for the translator can be found here:
- <a href="https://confluence.slac.stanford.edu/display/PCDS/Event+data+format">Event Data Format</a> input format
- <a href="https://confluence.slac.stanford.edu/pages/createpage.action?spaceKey=PSDM&title=EPICS+data+format">Epics Data Format</a> input format
- <a href="https://confluence.slac.stanford.edu/display/PCDS/Scientific+data+format">Scientific Data Format</a> This is the output format.

@section Running Running the Translator

There are three ways to run the translator
- Through the <a href="https://confluence.slac.stanford.edu/display/PCDS/Interface+Controller">Interface Controller</a> (assuming the interface controller has been configured to run psana-translate as opposed to o2o-translate)
- As a psana module, via the psana command line or by writing a psana configuration.
- Through the command line wrapper psana-translate to the Translator.H5Output psana module

One runs psana-translate as
@verbatim
   psana-translate [psana arguments] --output_file=h5outfile [optional H5Output options] [--save_cfg=filename]
@endverbatim

For example
@verbatim
   psana-translate -v -v -n 5 /reg/d/psdm/mec/mec01/xtc/e01-r001-s0*-c0*.xtc --output_file=output.h5
@endverbatim

would use psana options to get debugging output and only read the first 5 events from the mec01 run1 files.  
Run psana-translate for more help on using the wrapper - for details on all of the translation options, see the section @ref PsanaCfg.

@section NewFeatures New Features - Filtering and Writing NDArrays (and strings)

With psana-translate, you can

- filter out whole events from translation
- filter out certain data, by data type, or by data source
- write ndarray's that other modules add to the event store
- write std::string's that other modules add to the event store

@subsection Filtering events

Since psana-translate runs as a psana module, it is possible to filter translated events through psana options and other modules. psana options allow you to start at a certain event, and process a certain number of events.  Moreover a user module that is loaded before the Translator module can tell psana that it should not pass this event on to any other modules, hence the H5Output will never see the event and it will not get translated.

psana-translate also provides a C++ interface to filtering that will record the event times of the filtered messages, as well as a user log message as to why the event was filtered.  See the function doNotTranslate and the example class TestDoNotTranslate.  Using this function will cause a group 'filtered' to be created in each CalibCycle where events are filtered.  The filtered group will include the datasets 'time' (with the even id's) and 'data' with the log messages.

@subsection Filtering types

The psana.cfg file accepts a number of parameters that will filter out sets of psana types.  For example setting
@code
EBeam = exclude
@endcode
would cause any of the types Psana::Bld::BldDataEBeamV0, Psana::Bld::BldDataEBeamV1, Psana::Bld::BldDataEBeamV2, Psana::Bld::BldDataEBeamV3 or Psana::Bld::BldDataEBeamV4 to be excluded from translation.  See the section @ref PsanaCfg for more details.

@subsection Src Filtering

Specific src's can be filtered by providing a list such as 
@code
src_filter = exclude NoDetector.0:Evr.2  CxiDs1.0:Cspad.0  CxiSc2.0:Cspad2x2.1  EBeam  FEEGasDetEnergy  CxiDg2_Pim
@endcode
again, see @ref PsanaCfg for more details.

@subsection Writing NDArrays and Strings
ndarrays (up to dimension 4 of the standard integral and float types) and std::string's that are written into the event store will be written to the hdf5 by default.  These events can be filtered as well.  See the section @ref PsanaCfg for details.

@section PsanaCfg Psana Configuration File and all Options

When running the translator as a psana module, if is often convenient to create a psana.cfg file.  The Translator package include
the file default_psana.cfg which is a psana configuration file that describes all the options possible, with extensive documentation
as to what they mean.  Below we include this file for reference:

@verbatim
######################################################################
[psana]

# MODULES: any modules that produce data to be translated need be loaded 
# **BEFORE** Translator.H5Output (such as calibrated data or NDArray's)
# event data added by modules listed after Translator.H5Output is not translated.
modules = Translator.H5Output

files = **TODO: SPECIFY INPUT FILES OR DATA SOURCE HERE**

######################################################################
[Translator.H5Output]

# TODO: enter the full h5 output file name, including the output directory
output_file = output_directory/h5output.h5

# # # # # # # # # # # # # # # # # # # # #
# EPICS FILTERING
# The Translator can store epics pv's in one of two ways, or not at all.
# set store_epics below, to one of the following:
#
# updates_only   stores an epic pv when it has changed. The pv is stored 
#                in the current calib cycle.  For mutli calib cycle experiments, 
#                users may have to look back through several calib cycle's to 
#                find the latest value of a pv.
#
# calib_repeat   each calib cycle will include the latest value of all the epics 
#                pv's.  This can make it easier to find pv's for a calib cycle. 
#                For experiments with many short calib cycles, it can degrade 
#                performance of translation and performance when working with the 
#                resulting hdf5 file.
#
# no             epics pv's will not be stored. You may also want to set Epics=exclude
#                (see below) if you do not want the epics configuration data stored

# The default is 'updates_only'

store_epics = updates_only

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE FILTERING 
#
# One can include or exclude a class of Psana types with the following 
# options. Only the strings include or exclude are valid for these 
# type filtering options. 
# 
# Note - Epics in the list below refers only to the epicsConfig data
# which is the alias list, not the epics pv's (see store_epics above for those)

AcqTdc = include               # Psana::Acqiris::TdcConfigV1, Psana::Acqiris::TdcDataV1
AcqWaveform = include          # Psana::Acqiris::ConfigV1, Psana::Acqiris::DataDescV1
Alias = include                # Psana::Alias::ConfigV1
Andor = include                # Psana::Andor::ConfigV1, Psana::Andor::FrameV1
Control = include              # Psana::ControlData::ConfigV1, Psana::ControlData::ConfigV2, Psana::ControlData::ConfigV3
Cspad = include                # Psana::CsPad::ConfigV1, Psana::CsPad::ConfigV2, Psana::CsPad::ConfigV3, Psana::CsPad::ConfigV4, Psana::CsPad::ConfigV5, Psana::CsPad::DataV1, Psana::CsPad::DataV2
Cspad2x2 = include             # Psana::CsPad2x2::ConfigV1, Psana::CsPad2x2::ConfigV2, Psana::CsPad2x2::ElementV1
DiodeFex = include             # Psana::Lusi::DiodeFexConfigV1, Psana::Lusi::DiodeFexConfigV2, Psana::Lusi::DiodeFexV1
EBeam = include                # Psana::Bld::BldDataEBeamV0, Psana::Bld::BldDataEBeamV1, Psana::Bld::BldDataEBeamV2, Psana::Bld::BldDataEBeamV3, Psana::Bld::BldDataEBeamV4
Encoder = include              # Psana::Encoder::ConfigV1, Psana::Encoder::ConfigV2, Psana::Encoder::DataV1, Psana::Encoder::DataV2
Epics = include                # Psana::Epics::ConfigV1
EpixSampler = include          # Psana::EpixSampler::ConfigV1, Psana::EpixSampler::ElementV1
Evr = include                  # Psana::EvrData::ConfigV1, Psana::EvrData::ConfigV2, Psana::EvrData::ConfigV3, Psana::EvrData::ConfigV4, Psana::EvrData::ConfigV5, Psana::EvrData::ConfigV6, Psana::EvrData::ConfigV7, Psana::EvrData::DataV3
EvrIO = include                # Psana::EvrData::IOConfigV1
Evs = include                  # Psana::EvrData::SrcConfigV1
FEEGasDetEnergy = include      # Psana::Bld::BldDataFEEGasDetEnergy
Fccd = include                 # Psana::FCCD::FccdConfigV1, Psana::FCCD::FccdConfigV2
Fli = include                  # Psana::Fli::ConfigV1, Psana::Fli::FrameV1
Frame = include                # Psana::Camera::FrameV1
FrameFccd = include            # Psana::Camera::FrameFccdConfigV1
FrameFex = include             # Psana::Camera::FrameFexConfigV1
GMD = include                  # Psana::Bld::BldDataGMDV0, Psana::Bld::BldDataGMDV1
Gsc16ai = include              # Psana::Gsc16ai::ConfigV1, Psana::Gsc16ai::DataV1
Imp = include                  # Psana::Imp::ConfigV1, Psana::Imp::ElementV1
Ipimb = include                # Psana::Ipimb::ConfigV1, Psana::Ipimb::ConfigV2, Psana::Ipimb::DataV1, Psana::Ipimb::DataV2
IpmFex = include               # Psana::Lusi::IpmFexConfigV1, Psana::Lusi::IpmFexConfigV2, Psana::Lusi::IpmFexV1
L3T = include                  # Psana::L3T::ConfigV1, Psana::L3T::DataV1
OceanOptics = include          # Psana::OceanOptics::ConfigV1, Psana::OceanOptics::DataV1
Opal1k = include               # Psana::Opal1k::ConfigV1
Orca = include                 # Psana::Orca::ConfigV1
PhaseCavity = include          # Psana::Bld::BldDataPhaseCavity
PimImage = include             # Psana::Lusi::PimImageConfigV1
Princeton = include            # Psana::Princeton::ConfigV1, Psana::Princeton::ConfigV2, Psana::Princeton::ConfigV3, Psana::Princeton::ConfigV4, Psana::Princeton::ConfigV5, Psana::Princeton::FrameV1, Psana::Princeton::FrameV2
PrincetonInfo = include        # Psana::Princeton::InfoV1
Quartz = include               # Psana::Quartz::ConfigV1
Rayonix = include              # Psana::Rayonix::ConfigV1, Psana::Rayonix::ConfigV2
SharedAcqADC = include         # Psana::Bld::BldDataAcqADCV1
SharedIpimb = include          # Psana::Bld::BldDataIpimbV0, Psana::Bld::BldDataIpimbV1
SharedPim = include            # Psana::Bld::BldDataPimV1
Spectrometer = include         # Psana::Bld::BldDataSpectrometerV0
TM6740 = include               # Psana::Pulnix::TM6740ConfigV1, Psana::Pulnix::TM6740ConfigV2
Timepix = include              # Psana::Timepix::ConfigV1, Psana::Timepix::ConfigV2, Psana::Timepix::ConfigV3, Psana::Timepix::DataV1, Psana::Timepix::DataV2
TwoDGaussian = include         # Psana::Camera::TwoDGaussianV1
UsdUsb = include               # Psana::UsdUsb::ConfigV1, Psana::UsdUsb::DataV1
pnCCD = include                # Psana::PNCCD::ConfigV1, Psana::PNCCD::ConfigV2, Psana::PNCCD::FramesV1

# user types to translate from the event store
ndarray_types = include        # ndarray<int8_t,1>, ndarray<int8_t,2>, ndarray<int8_t,3>, ndarray<int8_t,4>, ndarray<int16_t,1>, ndarray<int16_t,2>, ndarray<int16_t,3>, ndarray<int16_t,4>, ndarray<int32_t,1>, ndarray<int32_t,2>, ndarray<int32_t,3>, ndarray<int32_t,4>, ndarray<int64_t,1>, ndarray<int64_t,2>, ndarray<int64_t,3>, ndarray<int64_t,4>, ndarray<uint8_t,1>, ndarray<uint8_t,2>, ndarray<uint8_t,3>, ndarray<uint8_t,4>, ndarray<uint16_t,1>, ndarray<uint16_t,2>, ndarray<uint16_t,3>, ndarray<uint16_t,4>, ndarray<uint32_t,1>, ndarray<uint32_t,2>, ndarray<uint32_t,3>, ndarray<uint32_t,4>, ndarray<uint64_t,1>, ndarray<uint64_t,2>, ndarray<uint64_t,3>, ndarray<uint64_t,4>, ndarray<float,1>, ndarray<float,2>, ndarray<float,3>, ndarray<float,4>, ndarray<double,1>, ndarray<double,2>, ndarray<double,3>, ndarray<double,4>
std_string = include           # std::string



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SOURCE FILTERING
#
# The default for the src_filter option is "include all"
# If you want to include a subset of the sources, do
#
# src_filter include srcname1 srcname2  
#
#  or if you want to exclude a subset of sources, do
#
# src_filter exclude srcname1 srcname2
#
# The srcname's in xtc input can be printed using the EventKeys
# module.  For example
#
# psana -n 5 -m EventKeys exp=cxi1235:run=33 
#
# Will print the EventKeys in the first 5 events.  If the output includes
#
#   EventKey(type=Psana::EvrData::DataV3, src=DetInfo(NoDetector.0:Evr.2))
#   EventKey(type=Psana::CsPad::DataV2, src=DetInfo(CxiDs1.0:Cspad.0))
#   EventKey(type=Psana::CsPad2x2::ElementV1, src=DetInfo(CxiSc2.0:Cspad2x2.1))
#   EventKey(type=Psana::Bld::BldDataEBeamV3, src=BldInfo(EBeam))
#   EventKey(type=Psana::Bld::BldDataFEEGasDetEnergy, src=BldInfo(FEEGasDetEnergy))
#   EventKey(type=Psana::Camera::FrameV1, src=BldInfo(CxiDg2_Pim))
#
# Then one can filter on these six srcname's:
#
#  NoDetector.0:Evr.2  CxiDs1.0:Cspad.0  CxiSc2.0:Cspad2x2.1  EBeam  FEEGasDetEnergy  CxiDg2_Pim
#

src_filter = include all

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CALIBRATION FILTERING
#
# Psana calibration modules can produce a calibrated version of CsPad data
# (The data types CsPad::DataV1 or CsPad::DataV2). The module output will be 
# data of the same type and src as the uncalibrated data, with an additional key, 
# such as 'calibrated'.
#
# The Translator defaults to skipping the translation of the uncalibrated
# data when a calibrated version of that data is present.  Below you 
# can control the calibration key and whether or not to include the
# uncalibrated data.

calibration_key = calibrated
include_uncalibrated_data = false

# Note: this only affects calibrated data of the same type and src as the
# uncalibrated data.  When the calibration module produces a NDArray, both
# the NDArray and the uncalibrated data are translated.  If you do not wish
# to translate the uncalibrated data, use appropriate type or src_filter options.
# Likewise if you do not want to translate certain NDArray's, see the 
# ndarray_key_filter options below.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NDARRAY AND STD::STRING KEY FILTERING
#
# A number of NDArray's and any std::string found in the event store are translated into 
# the hdf5 file.  NDarray's up to 4 dimensions of 10 basic types (8, 16, 32 and 64 bit 
# signed and unsigned int, float and double) are translated, but see the comment after the 
# ndarray_types option in the type filtering section for the most up to date list.
#
# These NDArray's and std::string's can be filtered by specifying the eventKey key that was
# used to put the data in the event.  While a srcname and key uniquely distinguish data in the
# event store, the Translator filter's NDArray's and std::string's using only the
# key string. The default is to include all ndarray's and std::string's found:

ndarray_key_filter = include all
std_string_key_filter = include all

# an example of including only one ndarray (with keystring being 'finalanswer') would be
#
# ndarray_key_filter include finalanswer
#
# and several ndarrays or strings can be included or excluded
#
# ndarray_key_filter = exclude arrayA arrayB
# std_string_key_filter = include message1 eventDescription

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# COMPRESSION 
#
# The following options control compression for most all datasets.
# Shuffling improves compression for certain datasets. Valid values for
# deflate (gzip compression level) are 0-9. Setting deflate = -1 turns off
# compression.

shuffle = true
deflate = 1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TECHNICAL, ADVANCED CONFIGURATION
# 
# ---------------------------------------
# CHUNKING
# The commented options below give the default chunking options.
# Objects per chunk are selected from the target chunk size (16 MB) and 
# adjusted based on min/max objects per chunk, and the max bytes per chunk.
# It is important that the chunkCache (created on a per dataset basis) be 
# large enough to hold at least one chunk, ideally all chunks we need to have
# open at one time when writing to the dataset (usually one, unless we repair
# split events):
 
# chunkSizeTargetInBytes = 1703936 (16MB)
# chunkSizeTargetObjects = 0 (0 means select objects per chunk from chunkSizeInBytes)
# maxChunkSizeInBytes = 10649600  (100MB)
# minObjectsPerChunk = 50              
# maxObjectsPerChunk = 2048
# chunkCacheSizeTargetInChunks = 3
# maxChunkCacheSizeInBytes = 10649600  (100MB)

# ---------------------------------------
# REFINED DATASET CONTROL
#
# There are six classes of datasets for which individual options for shuffle,
# deflate, chunkSizeTargetInBytes and chunkSizeTargetObjects can be specified:
#
# regular (most everything, all psana types)
# epics (all the epics pv's)
# damage (accompanies all regular data from event store)
# ndarrays (new data from other modules)
# string's (new data from other modules)
# eventId (the time dataset that also accompanies all regular data, epics pvs, ndarrays and strings)
#
# The options for regular datasets have been discussed above. The other five datasets 
# get their default values for shuffle, deflate, chunkSizeInBytes and chunkSizeInObjects
# from the regular dataset options except in the cases below:
 
# damageShuffle = false
# stringShuffle = false
# epicsPvShuffle = false
# stringDeflate = -1
# eventIdChunkSizeTargetInBytes = 16384
# epicsPvChunkSizeTargetInBytes = 16384

# The rest of the shuffle, deflate and chunk size options for the other five datasets are:
#
# eventIdShuffle = true
# eventIdDeflate = 1
# damageDeflate = 1
# epicsPvDeflate = 1
# ndarrayShuffle = true
# ndarrayDeflate = 1
# eventIdChunkSizeTargetObjects = 0
# damageChunkSizeTargetInBytes = 1703936
# damageChunkSizeTargetObjects = 0
# stringChunkSizeTargetInBytes = 1703936
# stringChunkSizeTargetObjects = 0
# ndarrayChunkSizeTargetInBytes = 1703936
# ndarrayChunkSizeTargetObjects = 0
# epicsPvChunkSizeTargetObjects = 0

# ---------------------------------------
# SPLIT EVENTS
# When the Translator encounters a split event, it checks a cache to see
# if it has already seen it.  If it has, it fills in any blanks that it can.
# To prevent this cache from growing to large, set the maximum number of
# split events to look back through here (default is 3000):

max_saved_split_events = 3000

# ---------------------------------------
# HDF5 GROUP NAMES
# The typenames for beam line data defaults to being written as (for example) 
# Bld::BldDataEBeamV0. Setting short_bld_name to true causes it to be 
# written as BldDataEBeamV0. If set to true, names are written differently
# then with o2o-translate and the change may break code that reads h5 files 
# (such as psana)

short_bld_name = false

# ---------------------------------------
# HDF5 FILE PROPERTIES
#
# split large files, presently we only support NoSplit. Future options may be: Family and SplitScan
# for future splitting, splitSize defaults to 10 GB
split = NoSplit
splitSize = 10737418240
@endverbatim

@subsection TranslationDamage Translation and Damage

psana has a specific damage policy that tells it what damaged data is acceptable for psana modules and what data is not -- see PSXtcInput::XtcInputModuleBase::eventDamagePolicy and PSXtcInput::XtcInputModuleBase::configDamagePolicy. The default behavior is
- configStore - only undamaged data is stored in the configStore
- EventStore - undamaged data, and EBeam data with user damage is stored in the event, all other damage is not stored.

This deviates slightly from what o2o-translate would do.  o2o-translate would also store out of order damaged data.  There is a psana option that can be added to the [psana] section of the .cfg file to recover this behavior.  Below we document some special options that control what damaged data psana stores:

- store-out-of-order-damage  - defaults to false, set to true if you want to translate out of order damaged data
- store-user-ebeam-damage  - defaults to true, set to false if you do not want to translate EBeam data what has user damage
- store-damaged-config - defaults to false, set to true if you want to store damaged config data

@section Compare Difference's with O2OTranslator

@subsection DroppedFeatures Feature's Dropped from o2o-translate

- hdf file creation parameters
-- Only NoSplit is implemented - no family or split drivers.

In general a number of o2o-translate options are no longer supported.  In particular:
-  -G (long names like CalibCycle:0000 instead of CalibCycle) is always on.

Signficant Translation differences:
- PNCCD::FullFrame data is no longer translated. FullFrame is a copy of Frames with a more convenient interface. User's interested in having FullFrame written into their hdf5 files rather than the original Frames data should make a feature request.

@subsection Speed

psana-translate runs about 10% slower than o2o-translate does. 

Performance testing was done during November/December of 2013.  Both o2o-translate and psana-translate worked through a 92 GB xtc file using compression=1 on the rhat6 machine psdev105.  They read and wrote the data from /u1. They both used the non-parallel compression library.  o2o-translate produced a 68GB file in 65 minutes and psana-translate produced a 65GB file in 70 minutes.  (Speeds of about 22MB/sec).  Production runs will use the parallel compression library and are expected to run at faster speeds (about 50MB/sec).

@subsection TechnicalDifferences Technical Difference's with o2o-translate

- File attributes runNumber, runType and experiment not stored, instead expNum, experiment, instrument and jobName are stored (from the psana Env object)
- The attribute :schema:timestamp-format is always "full", there is no option for "short"
- The output file must be explicitly specificed in the psana cfg file. It is not inferred from the input.
- The File attribute origin is now psana-translator as opposed to translator
- The end sec and nanoseconds are not written into the Configure group at the end of the job as there is no EventId in the Event at the end.
- integer size changes - a number of fields have changed size, a few examples are below.  In one quirky case, this caused translation to be different.  The reason was that the data was uninitialized, and the new 32 bit value was different than the old 16 bit value. Beam line data produced from 2014 onward will not include unitialized data in the translation, users will not have to worry about.  Unitialized data is very rare in pre 2014 data and, due to its location, not likely to be used in analysis.

A few Examples of field size changes:
-    EvrData::ConfigV7/seq_config - sync_source - enum was uint16, now uint32
-    EvrData::ConfigV7/seq_config - beam_source - enum was uint16, now uint32
-    Ipimb::DataV2 - source_id was uint16, now uint8
-    Ipimb::DataV2 - conn_id was uint16 now uint8
-    Ipimb::DataV2 - module was uint16, now uint8

Some types have their field names rearranged. For example with ControlData::ConfigV2 one has:
@verbatim
ControlData::ConfigV2:
  o2o: uses_duration uses_events duration events npvControls npvMonitors npvLabels
psana: events uses_duration uses_events duration npvControls npvMonitors npvLabels

EvrData::ConfigV7:
  o2o: code isReadout isCommand isLatch reportDelay reportWidth releaseCode maskTrigger maskSet maskClear desc readoutGroup
psana: code isReadout isCommand isLatch reportDelay reportWidth maskTrigger maskSet maskClear desc readoutGroup releaseCode
@endverbatim

- Epics Ctrl datasets (in the configure group as opposed to the calib group) are not chunked.  They are stored as fixed size datasets depending on the number of pv's. 
- Only one epics pv is stored per name (of course, one epics pv may have any number of elements within it). This is fine as the epic pv name is supposed to uniquely identify the pv.  However in xtc files, you can see several epics pv's with the same pvname, but different pvid's. This scenario should only arise when the same pv is coming from different sources, and replicates the same data.  Psana only stores one epics pv per name (the last one it sees in a datagram). This is the one that the translator will pick up and store.
- All Epics pv's are stored in the source folder EpicsArch.0:NoDevice.0.  With o2o-translate, some could be split off into other folders (such as AmoVMI.0:Opal1000.0). As epics pv names uniquely identify the data, the source information should not be needed.
- Typenames that started with Bld::Bld can be shortened to start with just Bld, but they default to stay as Bld::Bld (set short_bld_names = false in the psana.cfg to shorten these names, but this may break existing code that reads .h5 files).
- Some DAQ config objects include space for a maximum number of entries.  o2o-translate would only write entries for those used.  The psana translator does not.  For example:
-- The Acqiris::ConfigV1 vert dataset now always prints the max of 20 channels, even if the user will only be using 3. 
-- Note, in this case the Acqiris data will still only include the 3 channels being used. o2o-translate was making an adjustment to the config data being written.
- psana-translate will write an emtpy output_lookup_table for Opal1k::ConfigV1 output_lookup_table, even if output_lookup_table() is enabled.  o2o-translate would not.
- psana-translate does not produce the _dim_fix_flag_20103 datasets that o2o-translate did.
- Bld::BldDataGMDV  the field fSpare1 has been dropped from this type.
- With psana-translate, if all the xtc's coming from a particular source are damaged, you will not see a 'data' dataset in the hdf5 file. You will see the time, _damage and _mask datasets that tell the damage and events where the omitted data lives. o2o-translate may have created a 'data' dataset filled with blanks.
- OutOfOrder Damage - by default, o2o-translate translated out of order damage, however psana-translate does not.  psana can be told to include this kind of damaged data by setting store-out-of-order-damage=true in the [psana] section of your .cfg file.




@section RelatedPackages Related Packages
- O2OTranslator - original hdf5 translator
- psddldata - DDL definitions
- psddl_hdf2psana - extra DDL definitions for hdf5 schema's and generated code
- psddl - package that carries out code generation from DDL

*/
