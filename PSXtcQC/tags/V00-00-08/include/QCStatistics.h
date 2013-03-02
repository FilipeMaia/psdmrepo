#ifndef PSXTCQC_QCSTATISTICS_H
#define PSXTCQC_QCSTATISTICS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QCStatistics.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <string>
#include <iostream> // for cout, puts etc.

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "pdsdata/xtc/Xtc.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TransitionId.hh"
#include "pdsdata/xtc/TypeId.hh"
//#include "pdsdata/xtc/Damage.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace Pds {
class Xtc;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcQC {

/// @addtogroup PSXtcQC

/**
 *  @ingroup PSXtcQC
 *
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */


struct BrokenTimeRecord {
  unsigned       ndg;
  uint32_t       damage_value;
  Pds::ClockTime time_dg;
  Pds::ClockTime time_dg_prev;
};


struct DamageRecord {
  unsigned ndg;
  unsigned depth;
  uint32_t damage_value;
  Pds::TypeId::Type type_id;
};


struct SizeErrorRecord {
  unsigned ndg;
  unsigned depth;
  int      remaining;
  uint32_t damage_ext;
  uint32_t damage_int;
  Pds::TypeId::Type typeid_ext;
  Pds::TypeId::Type typeid_int;
};


class QCStatistics  {
public:
    enum {Stop, Continue};

    QCStatistics(std::ostream& out); // {}
    virtual ~QCStatistics() {}

    void accumulateDgramStatistics(Pds::Dgram* dg);
    void accumulateXTCStatistics(Pds::Xtc* xtc, unsigned depth);
    void printTransIdStatistics();
    void printTypeIdStatistics();
    void printDamageStatistics();
    void printDamageBitStatistics();
    void saveBrokenTimeRecord(Pds::Dgram* dg, unsigned& ndgram);
    void printBrokenTimeRecords();
    void checkDgramTimeSequense(Pds::Dgram* dg, unsigned& ndgram);
    void processDgram(Pds::Dgram* dg, unsigned ndgram, unsigned long long dg_first_byte); 
    void saveDamageRecord(Pds::Xtc* xtc, unsigned depth, unsigned ndgram);
    void printDamageRecords(); 
    bool processXTC(Pds::Xtc* xtc, unsigned depth, unsigned ndgram); 
    void processXTCSizeError(Pds::Xtc* root, Pds::Xtc* xtc, int remaining, unsigned depth, unsigned ndgram);
    void printXTCSizeErrorRecords();
    void printQCSummary(unsigned ndgram=0);

private:
    unsigned m_transIdCounter[(int)Pds::TransitionId::NumberOf];

    unsigned  m_typeIdCounter     [(int)Pds::TypeId::NumberOf];
    unsigned  m_payloadZeroCounter[(int)Pds::TypeId::NumberOf];
    int       m_payloadMin        [(int)Pds::TypeId::NumberOf];
    int       m_payloadMax        [(int)Pds::TypeId::NumberOf];
    unsigned  m_versionMin        [(int)Pds::TypeId::NumberOf];
    unsigned  m_versionMax        [(int)Pds::TypeId::NumberOf];
    unsigned  m_srclogMin         [(int)Pds::TypeId::NumberOf];
    unsigned  m_srclogMax         [(int)Pds::TypeId::NumberOf];
    unsigned  m_srcphyMin         [(int)Pds::TypeId::NumberOf];
    unsigned  m_srcphyMax         [(int)Pds::TypeId::NumberOf];
    unsigned  m_srclevelMin       [(int)Pds::TypeId::NumberOf];
    unsigned  m_srclevelMax       [(int)Pds::TypeId::NumberOf];
    unsigned  m_depthMin          [(int)Pds::TypeId::NumberOf];
    unsigned  m_depthMax          [(int)Pds::TypeId::NumberOf];
    std::vector<std::string> v_src_names[(int)Pds::TypeId::NumberOf];

    unsigned  m_damageCounter[(int)Pds::TypeId::NumberOf][32];
    unsigned  m_damageCounterForTypeId[(int)Pds::TypeId::NumberOf];
    unsigned  m_damageCounterForBit[32];
 
    // Copy constructor and assignment are disabled by default
    QCStatistics ( const QCStatistics& );
    QCStatistics& operator = ( const QCStatistics& );

    Pds::ClockTime m_t_prev;

    std::vector<BrokenTimeRecord> v_brokenTime;
    std::vector<DamageRecord> v_damageRecords;
    std::vector<SizeErrorRecord> v_sizeErrorRecords;

    std::ostream& m_out;
};

//===================
//===================
// Global methods
//===================
//===================

    void printDgramHeader(                   Pds::Dgram* dg, unsigned& ndgram, unsigned long long& dg_first_byte); // printf
    void printDgramHeader(std::ostream& out, Pds::Dgram* dg, unsigned& ndgram, unsigned long long& dg_first_byte);
    void printXtcHeader(std::ostream& out, Pds::Xtc& xtc);
    const char* nameOfDamageBit(unsigned bit);
    const char* nameOfDamageBit(Pds::Damage::Value vid);
    void printDamagedBits(std::ostream& out, uint32_t value, unsigned offset=31);
    void printListOfDamagedBits(std::ostream& out);

} // namespace PSXtcQC

#endif // PSXTCQC_QCSTATISTICS_H
