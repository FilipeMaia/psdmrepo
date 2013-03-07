#include "psddl_hdf2psana/bld.ddlm.h"

#include <boost/make_shared.hpp>

#include "hdf5pp/CompoundType.h"

namespace {

hdf5pp::Type
BldDataEBeamV0_schemaV0_data_stored_type()
{
  typedef psddl_hdf2psana::Bld::BldDataEBeamV0_schemaV0_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert_stored<uint32_t>( "uDamageMask", offsetof(DsType, uDamageMask) ) ;
  type.insert_stored<double>( "fEbeamCharge", offsetof(DsType, fEbeamCharge) ) ;
  type.insert_stored<double>( "fEbeamL3Energy", offsetof(DsType, fEbeamL3Energy) ) ;
  type.insert_stored<double>( "fEbeamLTUPosX", offsetof(DsType, fEbeamLTUPosX) ) ;
  type.insert_stored<double>( "fEbeamLTUPosY", offsetof(DsType, fEbeamLTUPosY) ) ;
  type.insert_stored<double>( "fEbeamLTUAngX", offsetof(DsType, fEbeamLTUAngX) ) ;
  type.insert_stored<double>( "fEbeamLTUAngY", offsetof(DsType, fEbeamLTUAngY) ) ;
  return type ;
}

}

hdf5pp::Type
psddl_hdf2psana::Bld::BldDataEBeamV0_schemaV0_data::stored_type()
{
    static hdf5pp::Type type = BldDataEBeamV0_schemaV0_data_stored_type() ;
    return type ;
}

namespace {

hdf5pp::Type
BldDataEBeamV0_schemaV0_data_native_type()
{
  typedef psddl_hdf2psana::Bld::BldDataEBeamV0_schemaV0_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(DsType, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(DsType, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(DsType, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(DsType, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(DsType, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(DsType, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(DsType, fEbeamLTUAngY) ) ;
  return type ;
}

}

hdf5pp::Type
psddl_hdf2psana::Bld::BldDataEBeamV0_schemaV0_data::native_type()
{
  static hdf5pp::Type type = BldDataEBeamV0_schemaV0_data_native_type() ;
  return type ;
}

boost::shared_ptr<Psana::Bld::BldDataEBeamV0>
psddl_hdf2psana::Bld::BldDataEBeamV0::operator()(hdf5pp::Group group, uint64_t idx)
{
    typedef BldDataEBeamV0_schemaV0_data DsType;
      
    hdf5pp::DataSet<DsType> ds_data = group.openDataSet<DsType>(DsType::datasetName());
    hdf5pp::DataSpace file_dsp_data = ds_data.dataSpace().select_single(idx);

    DsType data;
    ds_data.read(hdf5pp::DataSpace::makeScalar(), file_dsp_data, &data, data.native_type());

  return boost::make_shared<Psana::Bld::BldDataEBeamV0>(data.uDamageMask, data.fEbeamCharge, data.fEbeamL3Energy,
          data.fEbeamLTUPosX, data.fEbeamLTUPosY, data.fEbeamLTUAngX, data.fEbeamLTUAngY);
}


namespace {

hdf5pp::Type
BldDataEBeamV1_schemaV0_data_stored_type()
{
  typedef psddl_hdf2psana::Bld::BldDataEBeamV1_schemaV0_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert_stored<uint32_t>( "uDamageMask", offsetof(DsType, uDamageMask) ) ;
  type.insert_stored<double>( "fEbeamCharge", offsetof(DsType, fEbeamCharge) ) ;
  type.insert_stored<double>( "fEbeamL3Energy", offsetof(DsType, fEbeamL3Energy) ) ;
  type.insert_stored<double>( "fEbeamLTUPosX", offsetof(DsType, fEbeamLTUPosX) ) ;
  type.insert_stored<double>( "fEbeamLTUPosY", offsetof(DsType, fEbeamLTUPosY) ) ;
  type.insert_stored<double>( "fEbeamLTUAngX", offsetof(DsType, fEbeamLTUAngX) ) ;
  type.insert_stored<double>( "fEbeamLTUAngY", offsetof(DsType, fEbeamLTUAngY) ) ;
  type.insert_stored<double>( "fEbeamPkCurrBC2", offsetof(DsType, fEbeamPkCurrBC2) ) ;

  return type;
}

}

hdf5pp::Type
psddl_hdf2psana::Bld::BldDataEBeamV1_schemaV0_data::stored_type()
{
    static hdf5pp::Type type = BldDataEBeamV1_schemaV0_data_stored_type() ;
    return type ;
}

namespace {

hdf5pp::Type
BldDataEBeamV1_schemaV0_data_native_type()
{
  typedef psddl_hdf2psana::Bld::BldDataEBeamV1_schemaV0_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(DsType, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(DsType, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(DsType, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(DsType, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(DsType, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(DsType, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(DsType, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(DsType, fEbeamPkCurrBC2) ) ;

  return type;
}

}

hdf5pp::Type
psddl_hdf2psana::Bld::BldDataEBeamV1_schemaV0_data::native_type()
{
    static hdf5pp::Type type = BldDataEBeamV1_schemaV0_data_native_type() ;
    return type ;
}


boost::shared_ptr<Psana::Bld::BldDataEBeamV1>
psddl_hdf2psana::Bld::BldDataEBeamV1::operator()(hdf5pp::Group group, uint64_t idx)
{
  typedef BldDataEBeamV1_schemaV0_data DsType;
    
  hdf5pp::DataSet<DsType> ds_data = group.openDataSet<DsType>(DsType::datasetName());
  hdf5pp::DataSpace file_dsp_data = ds_data.dataSpace().select_single(idx);

  DsType data;
  ds_data.read(hdf5pp::DataSpace::makeScalar(), file_dsp_data, &data, data.native_type());

  return boost::make_shared<Psana::Bld::BldDataEBeamV1>(data.uDamageMask, data.fEbeamCharge, data.fEbeamL3Energy,
          data.fEbeamLTUPosX, data.fEbeamLTUPosY, data.fEbeamLTUAngX, data.fEbeamLTUAngY, data.fEbeamPkCurrBC2);
}
