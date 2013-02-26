#ifndef PSDDL_HDF2PSANA_BLD_DDLM_H
#define PSDDL_HDF2PSANA_BLD_DDLM_H

#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "psddl_psana/bld.ddl.h"

namespace psddl_hdf2psana {
namespace Bld {

struct BldDataEBeamV0_schemaV0_data {
    
    static int schemaVersion() { return 0; }
    static const char* datasetName() { return "data"; }
    
    static hdf5pp::Type native_type() ;
    static hdf5pp::Type stored_type() ;

    uint32_t    uDamageMask;
    double      fEbeamCharge;    /* in nC */
    double      fEbeamL3Energy;  /* in MeV */
    double      fEbeamLTUPosX;   /* in mm */
    double      fEbeamLTUPosY;   /* in mm */
    double      fEbeamLTUAngX;   /* in mrad */
    double      fEbeamLTUAngY;   /* in mrad */
};

    
class BldDataEBeamV0 {
public:
  typedef Psana::Bld::BldDataEBeamV0 PsanaType;
  boost::shared_ptr<PsanaType> operator()(hdf5pp::Group group, uint64_t idx);
private:
  struct dataset_data;
};

struct BldDataEBeamV1_schemaV0_data {
    
    static int schemaVersion() { return 0; }
    static const char* datasetName() { return "data"; }
    
    static hdf5pp::Type native_type() ;
    static hdf5pp::Type stored_type() ;

    uint32_t    uDamageMask;
    double      fEbeamCharge;    /* in nC */
    double      fEbeamL3Energy;  /* in MeV */
    double      fEbeamLTUPosX;   /* in mm */
    double      fEbeamLTUPosY;   /* in mm */
    double      fEbeamLTUAngX;   /* in mrad */
    double      fEbeamLTUAngY;   /* in mrad */
    double      fEbeamPkCurrBC2; /* in Amps */
};


class BldDataEBeamV1 {
public:
  typedef Psana::Bld::BldDataEBeamV1 PsanaType;
  boost::shared_ptr<PsanaType> operator()(hdf5pp::Group group, uint64_t idx);
private:
  struct dataset_data;
};


} // namespace Bld
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_BLD.DDLM_H
