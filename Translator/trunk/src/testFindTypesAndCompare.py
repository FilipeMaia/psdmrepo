import sys
import os
import glob
import subprocess as sb
import collections
import random
import tempfile

def printAllTypes():  
  '''This function will produce a list of all the Psana types.
  It runs from the release directory and assumes that the psddldata
  package is installed.  It assumes xml files for the DDL.
  '''
  xmlFiles = glob.glob("data/psddldata/*.xml")  
  typeNames=set()
  for xml in xmlFiles:
    reader = XmlReader(xmlfiles=[xml],inc_dir=['data'])
    model = reader.read()
    for package in model.packages():
      namespace = package.name
      for packageType in package.types():
        if packageType.type_id:   # probably a real xtc ps type
          typename = packageType.name
          version = packageType.version
          xtcTypeId = packageType.type_id
          fullName = "Psana::%s::%s" % (namespace, typename)
          typeNames.add(fullName)
  typeNames=list(typeNames)
  typeNames.sort()
  print '\n'.join(typeNames)

# produced by printAllTypes
masterAllTypes = ['Psana::Acqiris::ConfigV1',
                  'Psana::Acqiris::DataDescV1',
                  'Psana::Acqiris::TdcConfigV1',
                  'Psana::Acqiris::TdcDataV1',
                  'Psana::Alias::ConfigV1',
                  'Psana::Andor::ConfigV1',
                  'Psana::Andor::FrameV1',
                  'Psana::Bld::BldDataAcqADCV1',
                  'Psana::Bld::BldDataEBeamV0',
                  'Psana::Bld::BldDataEBeamV1',
                  'Psana::Bld::BldDataEBeamV2',
                  'Psana::Bld::BldDataEBeamV3',
                  'Psana::Bld::BldDataEBeamV4',
                  'Psana::Bld::BldDataFEEGasDetEnergy',
                  'Psana::Bld::BldDataGMDV0',
                  'Psana::Bld::BldDataGMDV1',
                  'Psana::Bld::BldDataIpimbV0',
                  'Psana::Bld::BldDataIpimbV1',
                  'Psana::Bld::BldDataPhaseCavity',
                  'Psana::Bld::BldDataPimV1',
                  'Psana::Bld::BldDataSpectrometerV0',
                  'Psana::Camera::FrameFccdConfigV1',
                  'Psana::Camera::FrameFexConfigV1',
                  'Psana::Camera::FrameV1',
                  'Psana::Camera::TwoDGaussianV1',
                  'Psana::ControlData::ConfigV1',
                  'Psana::ControlData::ConfigV2',
                  'Psana::ControlData::ConfigV3',
                  'Psana::CsPad2x2::ConfigV1',
                  'Psana::CsPad2x2::ConfigV2',
                  'Psana::CsPad2x2::ElementV1',
                  'Psana::CsPad::ConfigV1',
                  'Psana::CsPad::ConfigV2',
                  'Psana::CsPad::ConfigV3',
                  'Psana::CsPad::ConfigV4',
                  'Psana::CsPad::ConfigV5',
                  'Psana::CsPad::DataV1',
                  'Psana::CsPad::DataV2',
                  'Psana::Encoder::ConfigV1',
                  'Psana::Encoder::ConfigV2',
                  'Psana::Encoder::DataV1',
                  'Psana::Encoder::DataV2',
                  'Psana::Epics::ConfigV1',
                  'Psana::EvrData::ConfigV1',
                  'Psana::EvrData::ConfigV2',
                  'Psana::EvrData::ConfigV3',
                  'Psana::EvrData::ConfigV4',
                  'Psana::EvrData::ConfigV5',
                  'Psana::EvrData::ConfigV6',
                  'Psana::EvrData::ConfigV7',
                  'Psana::EvrData::DataV3',
                  'Psana::EvrData::IOConfigV1',
                  'Psana::FCCD::FccdConfigV1',
                  'Psana::FCCD::FccdConfigV2',
                  'Psana::Fli::ConfigV1',
                  'Psana::Fli::FrameV1',
                  'Psana::Gsc16ai::ConfigV1',
                  'Psana::Gsc16ai::DataV1',
                  'Psana::Imp::ConfigV1',
                  'Psana::Imp::ElementV1',
                  'Psana::Ipimb::ConfigV1',
                  'Psana::Ipimb::ConfigV2',
                  'Psana::Ipimb::DataV1',
                  'Psana::Ipimb::DataV2',
                  'Psana::L3T::ConfigV1',
                  'Psana::L3T::DataV1',
                  'Psana::Lusi::DiodeFexConfigV1',
                  'Psana::Lusi::DiodeFexConfigV2',
                  'Psana::Lusi::DiodeFexV1',
                  'Psana::Lusi::IpmFexConfigV1',
                  'Psana::Lusi::IpmFexConfigV2',
                  'Psana::Lusi::IpmFexV1',
                  'Psana::Lusi::PimImageConfigV1',
                  'Psana::OceanOptics::ConfigV1',
                  'Psana::OceanOptics::DataV1',
                  'Psana::Opal1k::ConfigV1',
                  'Psana::Orca::ConfigV1',
                  'Psana::PNCCD::ConfigV1',
                  'Psana::PNCCD::ConfigV2',
                  'Psana::PNCCD::FramesV1',
                  'Psana::PNCCD::FullFrameV1',
                  'Psana::Princeton::ConfigV1',
                  'Psana::Princeton::ConfigV2',
                  'Psana::Princeton::ConfigV3',
                  'Psana::Princeton::ConfigV4',
                  'Psana::Princeton::ConfigV5',
                  'Psana::Princeton::FrameV1',
                  'Psana::Princeton::FrameV2',
                  'Psana::Princeton::InfoV1',
                  'Psana::Pulnix::TM6740ConfigV1',
                  'Psana::Pulnix::TM6740ConfigV2',
                  'Psana::Quartz::ConfigV1',
                  'Psana::Rayonix::ConfigV1',
                  'Psana::Timepix::ConfigV1',
                  'Psana::Timepix::ConfigV2',
                  'Psana::Timepix::ConfigV3',
                  'Psana::Timepix::DataV1',
                  'Psana::Timepix::DataV2',
                  'Psana::UsdUsb::ConfigV1',
                  'Psana::UsdUsb::DataV1']

def getAllTypesList():
  global masterAllTypes;
  return masterAllTypes;

# To test on the 100 types in the masterAllTypes list, we need to find
# data that covers all the types.  Below we define masterType2Files which
# will map a type to a set of files that contain the type.  This is built with
# the functions findNewData() and buildXtcFileList() by exploring what is on
# disk at the time it is run.  

masterType2Files = { 'Psana::Acqiris::ConfigV1': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc']),
                     'Psana::Acqiris::DataDescV1': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc']),
                     'Psana::Acqiris::TdcConfigV1': set(['/reg/d/psdm/sxr/sxr16410/xtc/e75-r0105-s00-c00.xtc']),
                     'Psana::Acqiris::TdcDataV1': set(['/reg/d/psdm/sxr/sxr16410/xtc/e75-r0105-s00-c00.xtc']),
                     'Psana::Alias::ConfigV1': set(['/reg/d/psdm/dia/diaxpp13/xtc/e311-r0031-s00-c00.xtc']),
                     'Psana::Andor::ConfigV1': set(['/reg/d/psdm/sxr/sxr63212/xtc/e208-r0113-s00-c00.xtc']),
                     'Psana::Andor::FrameV1': set(['/reg/d/psdm/sxr/sxr63212/xtc/e208-r0092-s00-c00.xtc']),
                     'Psana::Bld::BldDataEBeamV0': set(['/reg/d/psdm/amo/amo01509/xtc/e8-r0125-s00-c00.xtc']),
                     'Psana::Bld::BldDataEBeamV1': set(['Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc']),
                     'Psana::Bld::BldDataEBeamV2': set(['/reg/d/psdm/cxi/cxi69113/xtc/e256-r0104-s00-c00.xtc']),
                     'Psana::Bld::BldDataEBeamV3': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Bld::BldDataEBeamV4': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0384-s00-c00.xtc']),
                     'Psana::Bld::BldDataFEEGasDetEnergy': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Bld::BldDataGMDV0': set(['/reg/d/psdm/sxr/sxr63212/xtc/e208-r0020-s00-c00.xtc']),
                     'Psana::Bld::BldDataGMDV1': set(['/reg/d/psdm/sxr/sxrtut13/xtc/e306-r0366-s00-c00.xtc']),
                     'Psana::Bld::BldDataPhaseCavity': set(['Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', 'Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Bld::BldDataSpectrometerV0': set(['/reg/d/psdm/cxi/cxis0913/xtc/e387-r0315-s00-c00.xtc']),
                     'Psana::Camera::FrameFexConfigV1': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc']),
                     'Psana::Camera::FrameV1': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc']),
                     'Psana::ControlData::ConfigV1': set(['Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc']),
                     'Psana::ControlData::ConfigV2': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::ControlData::ConfigV3': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0384-s00-c00.xtc']),
                     'Psana::CsPad2x2::ConfigV1': set(['/reg/d/psdm/mob/mob30112/xtc/e229-r0653-s00-c00.xtc']),
                     'Psana::CsPad2x2::ConfigV2': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc']),
                     'Psana::CsPad2x2::ElementV1': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc']),
                     'Psana::CsPad::ConfigV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc']),
                     'Psana::CsPad::ConfigV2': set(['/reg/d/psdm/cxi/cxi22010/xtc/e60-r0357-s00-c00.xtc']),
                     'Psana::CsPad::ConfigV3': set(['/reg/d/psdm/cxi/cxi29111/xtc/e96-r0211-s00-c00.xtc']),
                     'Psana::CsPad::ConfigV4': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc']),
                     'Psana::CsPad::ConfigV5': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc']),
                     'Psana::CsPad::DataV1': set(['/reg/d/psdm/xpp/xppi1113/xtc/e382-r0121-s00-c00.xtc']),
                     'Psana::CsPad::DataV2': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc']),
                     'Psana::Encoder::ConfigV1': set(['/reg/d/psdm/sxr/sxr33211/xtc/e103-r0845-s00-c00.xtc']),
                     'Psana::Encoder::ConfigV2': set(['/reg/d/psdm/sxr/sxr61612/xtc/e203-r0332-s00-c00.xtc']),
                     'Psana::Encoder::DataV2': set(['/reg/d/psdm/sxr/sxr61612/xtc/e203-r0332-s00-c00.xtc']),
                     'Psana::Epics::ConfigV1': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', 'Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::EvrData::ConfigV2': set(['/reg/d/psdm/amo/amo01509/xtc/e8-r0125-s00-c00.xtc']),
                     'Psana::EvrData::ConfigV4': set(['/reg/d/psdm/amo/amoi0010/xtc/e48-r0056-s00-c00.xtc']),
                     'Psana::EvrData::ConfigV5': set(['Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc']),
                     'Psana::EvrData::ConfigV6': set(['/reg/d/psdm/cxi/cxi63112/xtc/e207-r0068-s00-c00.xtc']),
                     'Psana::EvrData::ConfigV7': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::EvrData::DataV3': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/t1_dropped_src.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::FCCD::FccdConfigV2': set(['/reg/d/psdm/sxr/sxr61612/xtc/e203-r0332-s00-c00.xtc']),
                     'Psana::Fli::ConfigV1': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0021-s00-c00.xtc']),
                     'Psana::Fli::FrameV1': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0009-s00-c00.xtc']),
                     'Psana::Gsc16ai::ConfigV1': set(['/reg/d/psdm/xpp/xpp63412/xtc/e209-r0100-s01-c00.xtc']),
                     'Psana::Gsc16ai::DataV1': set(['/reg/d/psdm/xpp/xpp63412/xtc/e209-r0100-s01-c00.xtc']),
                     'Psana::Imp::ConfigV1': set(['/reg/d/psdm/cxi/cxia4113/xtc/e325-r0002-s00-c00.xtc']),
                     'Psana::Imp::ElementV1': set(['/reg/d/psdm/cxi/cxia4113/xtc/e325-r0002-s00-c00.xtc']),
                     'Psana::Ipimb::ConfigV1': set(['/reg/d/psdm/cxi/cxi22010/xtc/e60-r0357-s00-c00.xtc']),
                     'Psana::Ipimb::ConfigV2': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1_dropped_src.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Ipimb::DataV1': set(['/reg/d/psdm/cxi/cxi22010/xtc/e60-r0357-s00-c00.xtc']),
                     'Psana::Ipimb::DataV2': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1_dropped_src.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Lusi::DiodeFexConfigV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s02-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s01-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s00-c00.xtc']),
                     'Psana::Lusi::DiodeFexV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s02-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s01-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s00-c00.xtc']),
                     'Psana::Lusi::IpmFexConfigV1': set(['/reg/d/psdm/cxi/cxi22010/xtc/e60-r0357-s00-c00.xtc']),
                     'Psana::Lusi::IpmFexConfigV2': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1_dropped_src.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Lusi::IpmFexV1': set(['/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', 'Translator/data/t1.xtc', 'Translator/data/t1_new_out_of_order.xtc', 'Translator/data/t1_dropped_src.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/t1_previously_seen_out_of_order.xtc', 'Translator/data/t1_dropped.xtc', 'Translator/data/t1_initial_damage.xtc', 'Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc', 'Translator/data/t1_end_damage.xtc']),
                     'Psana::Lusi::PimImageConfigV1': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc']),
                     'Psana::OceanOptics::ConfigV1': set(['/reg/d/psdm/xpp/xppi0913/xtc/e380-r0017-s00-c00.xtc']),
                     'Psana::OceanOptics::DataV1': set(['/reg/d/psdm/xpp/xppi0913/xtc/e380-r0017-s00-c00.xtc']),
                     'Psana::Opal1k::ConfigV1': set(['Translator/data/amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc', '/reg/d/psdm/xpp/xppi0313/xtc/e283-r0010-s00-c00.xtc', '/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc', 'Translator/data/amo64913-r182-s02-OutOfOrder_Frame.xtc', '/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc', 'Translator/data/amo68413-r99-s02-userEbeamDamage.xtc']),
                     'Psana::Orca::ConfigV1': set(['/reg/d/psdm/xcs/xcsi0213/xtc/e323-r0048-s05-c00.xtc']),
                     'Psana::PNCCD::ConfigV2': set(['/reg/d/psdm/amo/amoc0113/xtc/e331-r0146-s00-c00.xtc']),
                     'Psana::PNCCD::FramesV1': set(['/reg/d/psdm/amo/amoc0113/xtc/e331-r0146-s00-c00.xtc']),
                     'Psana::PNCCD::FullFrameV1': set(['/reg/d/psdm/amo/amoc0113/xtc/e331-r0146-s00-c00.xtc']),
                     'Psana::Princeton::ConfigV1': set(['/reg/d/psdm/amo/amo31211/xtc/e91-r0288-s01-c00.xtc']),
                     'Psana::Princeton::ConfigV2': set(['/reg/d/psdm/sxr/sxr33211/xtc/e103-r0845-s00-c00.xtc']),
                     'Psana::Princeton::ConfigV3': set(['/reg/d/psdm/xcs/xcsi0112/xtc/e167-r0015-s00-c00.xtc']),
                     'Psana::Princeton::ConfigV4': set(['/reg/d/psdm/mec/mec53212/xtc/e183-r0450-s02-c00.xtc']),
                     'Psana::Princeton::ConfigV5': set(['/reg/d/psdm/mec/meca6113/xtc/e332-r0062-s03-c00.xtc']),
                     'Psana::Princeton::FrameV1': set(['/reg/d/psdm/sxr/sxr33211/xtc/e103-r0845-s00-c00.xtc']),
                     'Psana::Princeton::FrameV2': set(['/reg/d/psdm/xcs/xcs84213/xtc/e360-r0239-s00-c00.xtc']),
                     'Psana::Princeton::InfoV1': set(['/reg/d/psdm/sxr/sxr33211/xtc/e103-r0845-s00-c00.xtc']),
                     'Psana::Pulnix::TM6740ConfigV2': set(['/reg/d/psdm/cxi/cxi76413/xtc/e275-r0033-s00-c00.xtc']),
                     'Psana::Quartz::ConfigV1': set(['/reg/d/psdm/xpp/xpp61412/xtc/e202-r0025-s01-c00.xtc']),
                     'Psana::Timepix::ConfigV1': set(['Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc']),
                     'Psana::Timepix::ConfigV3': set(['/reg/d/psdm/xcs/xcsi0113/xtc/e289-r0030-s00-c00.xtc']),
                     'Psana::Timepix::DataV2': set(['Translator/data/xcscom12-r52-s0-dupTimes-splitEvents.xtc']),
                     'Psana::UsdUsb::ConfigV1': set(['/reg/d/psdm/amo/amo75113/xtc/e272-r0156-s00-c00.xtc']),
                     'Psana::UsdUsb::DataV1': set(['/reg/d/psdm/amo/amo75113/xtc/e272-r0156-s00-c00.xtc'])
                   }
# I need to test the below files
                     # These are new - not in the test files
#                     'Psana::Acqiris::TdcConfigV1': set(['/reg/d/psdm/sxr/sxr16410/xtc/e75-r0105-s00-c00.xtc']),
#                     'Psana::Acqiris::TdcDataV1': set(['/reg/d/psdm/sxr/sxr16410/xtc/e75-r0105-s00-c00.xtc']),
#                     'Psana::Fli::ConfigV1': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0021-s00-c00.xtc'])
#                     'Psana::Fli::FrameV1': set(['/reg/d/psdm/mec/meca6013/xtc/e355-r0009-s00-c00.xtc']),
#                     'Psana::OceanOptics::ConfigV1': set(['/reg/d/psdm/xpp/xppi0913/xtc/e380-r0017-s00-c00.xtc']),
#                     'Psana::OceanOptics::DataV1': set(['/reg/d/psdm/xpp/xppi0913/xtc/e380-r0017-s00-c00.xtc']),
#                     'Psana::EvrData::ConfigV2': set(['/reg/d/psdm/amo/amo01509/xtc/e8-r0125-s00-c00.xtc']),
#                     'Psana::Bld::BldDataEBeamV0': set(['/reg/d/psdm/amo/amo01509/xtc/e8-r0125-s00-c00.xtc']),

# as are these - from restored stuff - 10am Monday
#                           'Psana::CsPad::ConfigV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc']),
#                           'Psana::Gsc16ai::ConfigV1': set(['/reg/d/psdm/xpp/xpp63412/xtc/e209-r0100-s01-c00.xtc']),
#                           'Psana::Gsc16ai::DataV1': set(['/reg/d/psdm/xpp/xpp63412/xtc/e209-r0100-s01-c00.xtc']),
#                           'Psana::Lusi::DiodeFexConfigV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s02-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s01-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s00-c00.xtc']),
#                           'Psana::Lusi::DiodeFexV1': set(['/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s02-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s01-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc', '/reg/d/psdm/xpp/xppcom10/xtc/e40-r0613-s00-c00.xtc']),
#                           'Psana::Princeton::ConfigV4': set(['/reg/d/psdm/mec/mec53212/xtc/e183-r0450-s02-c00.xtc']),
#                           'Psana::Quartz::ConfigV1': set(['/reg/d/psdm/xpp/xpp61412/xtc/e202-r0025-s01-c00.xtc']),
#                           'Psana::Timepix::ConfigV3': set(['/reg/d/psdm/xcs/xcsi0113/xtc/e289-r0030-s00-c00.xtc']),


## at this point, 10am monday 12/9 these are the types that are not covered:
#['Psana::Bld::BldDataAcqADCV1',
# 'Psana::Bld::BldDataIpimbV0',
# 'Psana::Bld::BldDataIpimbV1',
# 'Psana::Bld::BldDataPimV1',
# 'Psana::Camera::FrameFccdConfigV1',
# 'Psana::Camera::TwoDGaussianV1',
# 'Psana::Encoder::DataV1',
# 'Psana::EvrData::ConfigV1',
# 'Psana::EvrData::ConfigV3',
# 'Psana::EvrData::IOConfigV1',
# 'Psana::FCCD::FccdConfigV1',
# 'Psana::L3T::ConfigV1',
# 'Psana::L3T::DataV1',
# 'Psana::Lusi::DiodeFexConfigV2',
# 'Psana::PNCCD::ConfigV1',
# 'Psana::Pulnix::TM6740ConfigV1',
# 'Psana::Rayonix::ConfigV1',
# 'Psana::Timepix::ConfigV2',
# 'Psana::Timepix::DataV1']

def getType2Files():
  '''This returns the current list of Psana types and what files to find them in.
  If the list gets out of date, return an empty dict and have findNewData() go through
  everything on disk
  '''
  global masterType2Files
  return masterType2Files

def getAllFiles2Test():
  type2Files = getType2Files()
  allFiles = set()
  for fileSet in type2Files.values():
    allFiles.update(fileSet)
  allFiles=list(allFiles)
  allFiles.sort()
  return allFiles
  
def sortMasterList():
  global masterType2Files
  keys = masterType2Files.keys()
  keys.sort()
  for ky in keys:
    print "'%s': %r" % (ky, masterType2Files[ky])

def typesInXtcFile(xtc, numEvents=120):
  '''Returns a set of the Psana types in the first numEvents of the xtc file.
  '''
  types = set()
  cmd = 'psana -n %d -m EventKeys %s | grep type=Psana | sort | uniq' % (numEvents,xtc)
  p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
  o,e = p.communicate()
  if e:
    raise Exception(e)
  for ln in o.split('\n'):
    if len(ln)<5:
      continue
    typeName = ln.split('type=')[1].split(', src=')[0]
    types.add(typeName)
  return types

def compareDicts(A,B):
  '''prints result of comparing dicts, for debugging
  '''
  Akeys = A.keys()
  Bkeys = B.keys()
  Akeys.sort()
  Bkeys.sort()
  print "**compareDicts**"
  keysEqual = (Akeys == Bkeys)
  print "keys equal=%s" % keysEqual
  if (Akeys != Bkeys):
    print "number keys %d %d" % (len(Akeys), len(Bkeys))
    print Akeys
    print Bkeys

  for key in Akeys:
    Aval = A[key]
    Bval = B[key]
    if Aval != Bval:
      print "values not equal for key=%s" % key
      print "Aval=%r" % Aval
      print "Bval=%r" % Bval

def updateType2Files(xtcs):
  '''prints the list of new entries for a given set of files. 
  Use findNewData() if possible which samples all files on disk
  '''
  type2files = collections.defaultdict(set)
  for xtc in xtcs:
    types = typesInXtcFile(xtc, numEvents=120)
    [type2files[typeName].add(xtc) for typeName in types]
    print " xtc: %s" % xtc
    print "types = %r" % types
  # check that all these types are in the master list
  masterAllTypes = set(getAllTypesList())
  notInAllTypes = [typeName for typeName in type2files.keys() if typeName not in masterAllTypes]
  if len(notInAllTypes) > 0:
    print "These types in the xtcs not in allTypes list: %r" % notInAllTypes

  masterType2Files = getType2Files()
  masterTypes = set(masterType2Files.keys())
  currentTypes = set(type2files.keys())
  newTypes = currentTypes.difference(masterTypes)
  
  print "xtcs contain newTypes= %r" % newTypes
  print "add lines"
  for typeName in newTypes:
    print "                           '%s': %r," % (typeName, type2files[typeName])

def missingTypes():
  '''Prints the types that are not covered by the types2Files list.
  '''
  masterType2Files = getType2Files()
  masterAllTypes = getAllTypesList()
  notInFileTypes = list(set(masterAllTypes).difference(set(masterType2Files.keys())))
  notInFileTypes.sort()
  print "\n".join(notInFileTypes)

def buildXtcFileList():
  '''Builds a 3D dictionary xtcs:
  xtcs[instrument][experiment][run] = list of xtcfiles
  That list is not all files in the run, just chunk0 files.
  This only includes experiments that include a xtc directory with at least one .xtc file
  in the directory.
  '''
  xtcs = {'amo':{},  'cxi':{},  'dia':{},  'mec':{},  'mob':{},  'sxr':{},  'xcs':{},  'xpp':{}}
  for instrument,experimentDictionary in xtcs.iteritems():
    print "instrument: ", instrument
    cmd = 'find /reg/d/psdm/%s | grep xtc' % instrument
    p = sb.Popen(cmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
    o,e=p.communicate()
    lns = [x for x in o.split('\n') if x.find('xtc')>0]
    experimentsWithXtcDir = [ln.split('%s/'%instrument)[1].split('/xtc')[0] for ln in lns]
    experimentsWithXtcFiles = {}
    for experimentDir in experimentsWithXtcDir:
      lsCmd = 'ls -1 /reg/d/psdm/%s/%s/xtc/*.xtc' % (instrument, experimentDir)
      lsp = sb.Popen(lsCmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
      lso,lse = lsp.communicate()
      xtcFiles = [x for x in lso.split('\n') if x.endswith('.xtc')]
      if len(xtcFiles)>0:
        experimentsWithXtcFiles[experimentDir]=xtcFiles
    for expDir, xtcFiles in experimentsWithXtcFiles.iteritems():
      experimentDictionary[expDir]=collections.defaultdict(list)
      for xtcFile in xtcFiles:
        run = int(os.path.basename(xtcFile).split('-r')[1].split('-s')[0])
        chunk = int(os.path.basename(xtcFile).split('-c')[1].split('.xtc')[0])
        if chunk != 0:
          continue
        experimentDictionary[expDir][run].append(xtcFile)
  return xtcs

def findNewData(xtcFiles=None):
  '''updates the master list of type2Files by going trhough xtc files
  INPUT
   xtcFiles - the 3D dictionary of files built by buildXtcFileList(), pass
              None and findNewData() will call the function
  findNewData() creates the instrument/experiment/run list, randomly shuffles it,
  and then checks the first xtc file from each instrument/experiment/run. 
  Any new types found in the file are used to update the masterTypesFile.

  any new types found are printed.

  RETURN  the updated dictionary
  '''
  masterTypes2Files = getType2Files()
  masterAllTypes = getAllTypesList()

  masterTypes = set(masterType2Files.keys())
  allTypesSet = set(masterAllTypes)

  filesInMasterList = set()
  for fileList in masterType2Files.values():
    filesInMasterList.update(fileList)

  if xtcFiles == None:
    xtcFiles = buildXtcFileList()

  runsToExamine = [(instrument, experiment, run)  for instrument in xtcFiles \
                   for experiment in xtcFiles[instrument] \
                   for run in xtcFiles[instrument][experiment]]
  random.shuffle(runsToExamine)
  for (instrument, experiment, run) in runsToExamine:
    xtcfile = xtcFiles[instrument][experiment][run][0] 
    types = typesInXtcFile(xtcfile,numEvents=120)
    notInAllTypes = [typeName for typeName in types if typeName not in allTypesSet]
    if len(notInAllTypes) > 0:
      print "These types in the xtc file %s are not in allTypes list: %r" % (xtcfile, notInAllTypes)
    newTypes = types.difference(masterTypes)
    if len(newTypes) == 0:
      print "nothing new in %s %s %s xtcfile=%s" % (instrument, experiment, run, xtcfile)
      continue
    print "xtcfile=%s contain newTypes= %r" % (xtcfile,newTypes)
    print "add lines"
    for typeName in newTypes:
      print "                           '%s': set(['%s'])," % (typeName, xtcfile)
      masterType2Files[typeName]=set([xtcfile])
      masterTypes.add(typeName)

  return masterType2Files

def checkRestoredFiles():
  xtcs = ['mec/mec53212/xtc/e183-r0450-s02-c00.xtc',
          'mec/mec62812/xtc/e205-r0570-s02-c00.xtc',
          'xpp/xppcom10/xtc/e40-r0670-s00-c00.xtc',
          'xpp/xppcom10/xtc/e40-r0613-s00-c00.xtc',
          'xpp/xppcom10/xtc/e40-r0613-s01-c00.xtc',
          'xpp/xppcom10/xtc/e40-r0613-s02-c00.xtc',
          'xpp/xpp63412/xtc/e209-r0100-s01-c00.xtc',
          'xpp/xpp61412/xtc/e202-r0025-s01-c00.xtc',
          'xcs/xcsi0112/xtc/e167-r0015-s00-c00.xtc',
          'xcs/xcsi0113/xtc/e289-r0030-s00-c00.xtc']
  xtcs = [os.path.join('/reg/d/psdm', xtc) for xtc in xtcs]
  updateType2Files(xtcs)
    
######  TESTING CODE ######

# To test a file, we want to do the following:
#    Make sure that it translates without any errors
#    Ideally, see that psana dumps the xtc and h5 the same way
#    Compare how it translates to o2o-translate

def captureOutput(cmd,outfile):
  p = sb.Popen(cmd,shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
  o,e = p.communicate()
  f = file(outfile,'w')
  f.write(o)
  f.close()
  return e.strip()

def filterErr(err):
  lns = [ ln.strip() for ln in err.split('\n') if len(ln.strip())>0]
  lns = [ ln for ln in lns if ln.find('unrecognized experiment name:')<0]
  lns = [ ln for ln in lns if ln.find('has no valid experiment number')<0]
  lns = [ ln for ln in lns if ln.find('EpicsH5GroupDirectory')<0 and ln.find('is the same as existing group')<0]
  lns = [ ln for ln in lns if ln.find('EpicsH5GroupDirectory')<0 and ln.find('has an empty target')<0]
  lns = [ ln for ln in lns if ln.find('Corrupted XTC, size out of range, xtc payload size:')<0]
  lns = [ ln for ln in lns if ln.find('H5Output.cpp')<0 and ln.find('has not been seen before.  Not writing blank.')<0]
  if len(lns)>0:
    raise Exception('\n'.join(lns))

def testFile(xtcFile, prefix):
  retDict = {}
  numEvents = 5
  xtcBaseName = os.path.basename(xtcFile)
  baseName = os.path.splitext(xtcBaseName)[0]

  # dump original xtc
  dumpFile = prefix + xtcBaseName + '.psana_dump'
  epicsDumpFile = prefix + xtcBaseName + '.psana_dump_epics'
  dumpAllXtcCmd = 'psana-dump -n %d %s' % (numEvents, xtcFile)
  dumpEpicsXtcCmd = 'psana-dump -n %d %s --onlyEpics' % (numEvents, xtcFile)
  filterErr(captureOutput(dumpAllXtcCmd, dumpFile))
  filterErr(captureOutput(dumpEpicsXtcCmd, epicsDumpFile))
  retDict.update({'xtcDumpAll':dumpFile,'xtcDumpEpics':epicsDumpFile,
          'xtcDumpCmd':dumpAllXtcCmd,'xtcDumpEpicsCmd':dumpEpicsXtcCmd})

  # translate first few events
  h5file = prefix + baseName +'.h5'
  translateOutput = prefix + baseName  + '.psana-translate'
  translateFirstEventsCmd = 'psana-translate -n %d %s --output_file=%s' % (numEvents, xtcFile, h5file)
  print translateFirstEventsCmd
  filterErr(captureOutput(translateFirstEventsCmd, translateOutput))
  print "done"
  retDict.update({'translateFew':translateFirstEventsCmd})

  # dump h5 file
  h5Dump = h5file + '.psana_dump'
  h5EpicsDump = h5file + '.psana_dump_epics'
  dumpAllH5Cmd = 'psana-dump -n %d %s' % (numEvents, h5file)
  dumpEpicsH5Cmd = 'psana-dump -n %d %s --onlyEpics' % (numEvents, h5file)
  filterErr(captureOutput(dumpAllH5Cmd, h5Dump))
  filterErr(captureOutput(dumpEpicsH5Cmd, h5EpicsDump))
  retDict.update({'h5DumpAll':h5Dump,'h5DumpEpics':h5EpicsDump,
          'h5DumpCmd':dumpAllH5Cmd,'h5DumpEpicsCmd':dumpEpicsH5Cmd})

  # diff dumps between xtc and h5, grep out experiment warnings
  diffFile = prefix + baseName  + '.psana_dump_diff'
  epicsDiffFile = prefix + baseName  + '.psana_dump_epics_diff'
  allDiffCmd = 'diff %s %s' % (dumpFile, h5Dump)
  filterErr(captureOutput(allDiffCmd, diffFile))
  epicsDiffCmd = 'diff %s %s' % (epicsDumpFile, h5EpicsDump)
  filterErr(captureOutput(epicsDiffCmd, epicsDiffFile))
  retDict.update({'diff':diffFile, 'epicsDiff':epicsDiffFile, 
                  'diff_cmd':allDiffCmd,'epics_diff_cmd':epicsDiffCmd})
  psanaDumpOfXtcAndHdf5Same = False
  psanaEpicsDumpOfXtcAndHdf5Same = False
  
  if file(diffFile).read().strip()=='':
    psanaDumpOfXtcAndHdf5Same = True
  if file(epicsDiffFile).read().strip()=='':
    psanaEpicsDumpOfXtcAndHdf5Same = True
  retDict.update({'diff_same':psanaDumpOfXtcAndHdf5Same,
                  'epics_same':psanaEpicsDumpOfXtcAndHdf5Same})

  return retDict

# compare longer translation, note, t
#  psanaTranslateCmd = 'psana-translate -n 120 %s --output_file=%s -Epics=exclude' % (xtcFile, h5file)
#  o2oTranslateCmd = 'o2o-translate -G --overwrite -n %s %s' % (o2oH5file, xtcFile)
#  os.system(o2ocmd)
#  compareLog = prefix + os.path.splitext(os.path.basename(xtcFile))[0] + '.compare_translation'
#  compareCmd = 'python Translator/src/compareTranslation.py %s %s --nostop > %s' % (o2oH5file, h5file, compareLog)
#  os.system(compareCmd)

def testAllFiles():
  files = getAllFiles2Test()
  # alot of times epics will look differnet
  expectedFailures = { # Belows file has FullFrame data that psana-translate does not translate
                      '/reg/d/psdm/amo/amoc0113/xtc/e331-r0146-s00-c00.xtc':['diff_same'],  
                      }
  crashes = set(['/reg/d/psdm/sxr/sxr16410/xtc/e75-r0105-s00-c00.xtc'])
  report = []
  for ii,fname in enumerate(files):
    if ii < 38:
      continue
    if fname in crashes:
      continue
    prefix = 'test_%0.3d_' % ii
    print ("*** %3d %55s " % (ii,fname)),
    retDict = testFile(fname,prefix)
    print " dump diff = %5s  epics diff = %5s ***" % (retDict['diff_same'],retDict['epics_same'])
    if not retDict['diff_same'] and (fname not in expectedFailures or ('diff_same' not in expectedFailures[fname])):
      print "  diff dump failed:"
      print "    diff dump: %s" % retDict['diff']
      print "    dump xtc cmd:  %s" % retDict['xtcDumpCmd']
      print "    create h5 cmd: %s" % retDict['translateFew']
      print "    dump h5 cmd:   %s" % retDict['h5DumpCmd']
#    if not retDict['epics_same']:
#      print "  diff epics failed:"
#      print "    epics diff dump:     %s" % retDict['epicsDiff']
#      print "    epics dump xtc cmd:  %s" % retDict['xtcDumpEpicsCmd']
#      print "    epics create h5 cmd: %s" % retDict['translateFew']
#      print "    epics dump h5 cmd:   %s" % retDict['h5DumpEpicsCmd']
      
      
        
#    print '\n'.join(['%s=%s' % (k,v) for k,v in retDict.iteritems()])
#    break

if __name__ == "__main__":
#  sortMasterList()
#  newFiles = sys.argv[1:]
#  updateType2Files(newFiles)
#  missingTypes()
#  buildXtcFileList()
#  xtcFile = sys.argv[1]
#  prefix = sys.argv[2]
#  if not os.path.exists(xtcFile):
#    print "Can't find %s" %s
#    sys.exit(1)
#  testFile(xtcFile, prefix)
  testAllFiles()
  
          

