//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QCStatistics...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcQC/QCStatistics.h"
//-----------------
// C/C++ Headers --
//-----------------
#include <string>   // for string, substring
#include <sstream>  // for streamstring
#include <iostream> // for cout, puts etc.
#include <iomanip>  // for setw
#include <unistd.h> // read()
#include <time.h>   // time

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventKey.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

using namespace PSXtcQC;
using namespace Pds;
using namespace std;

namespace PSXtcQC {
//===================

QCStatistics::QCStatistics() 
{
  std::fill_n(&m_transIdCounter[0],     (int)Pds::TransitionId::NumberOf, 0);

  std::fill_n(&m_typeIdCounter[0],      (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(&m_payloadZeroCounter[0], (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(&m_payloadMin[0],         (int)Pds::TypeId::NumberOf, 11111111);
  std::fill_n(&m_payloadMax[0],         (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(&m_versionMin[0],         (int)Pds::TypeId::NumberOf,  111);
  std::fill_n(&m_versionMax[0],         (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(& m_srclogMin[0],         (int)Pds::TypeId::NumberOf, 11111111);
  std::fill_n(& m_srclogMax[0],         (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(& m_srcphyMin[0],         (int)Pds::TypeId::NumberOf, 1111111111);
  std::fill_n(& m_srcphyMax[0],         (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(& m_srclevelMin[0],       (int)Pds::TypeId::NumberOf, 11111111);
  std::fill_n(& m_srclevelMax[0],       (int)Pds::TypeId::NumberOf,    0);
  std::fill_n(& m_depthMin[0],          (int)Pds::TypeId::NumberOf, 11111111);
  std::fill_n(& m_depthMax[0],          (int)Pds::TypeId::NumberOf,    0);

  std::fill_n(&m_damageCounter[0][0],       (int)Pds::TypeId::NumberOf*32, 0);
  std::fill_n(&m_damageCounterForTypeId[0], (int)Pds::TypeId::NumberOf, 0);
  std::fill_n(&m_damageCounterForBit[0],    32, 0);

  m_t_prev = Pds::ClockTime(0,0);
  //m_t_prev = Pds::ClockTime(0,0);

  v_brokenTime.clear();
  v_damageRecords.clear();
  v_sizeErrorRecords.clear();
}

//===================

void QCStatistics::accumulateDgramStatistics(Pds::Dgram* dg)
{
    m_transIdCounter[(int)dg->seq.service()] ++;    // Count transition Id
}

//===================

void QCStatistics::accumulateXTCStatistics(Pds::Xtc* xtc, unsigned depth) 
{
  int       ind      = (int)xtc->contains.id();
  int       payload  = xtc->sizeofPayload();
  unsigned  typevers = xtc->contains.version();
  unsigned  srclog   = ((xtc->src.log())&0xffffff);
  unsigned  srcphy   = xtc->src.phy();
  unsigned  dam_val  = xtc->damage.value();
  unsigned  level    = (unsigned)xtc->src.level();
  std::stringstream ssrc; ssrc << xtc->src; std::string src_name = ssrc.str();

  m_typeIdCounter[ind] ++;  

  if ( payload == 0 ) m_payloadZeroCounter[ind] ++;

  if ( payload  < m_payloadMin[ind] )   m_payloadMin[ind] = payload;
  if ( payload  > m_payloadMax[ind] )   m_payloadMax[ind] = payload;				        
  if ( typevers < m_versionMin[ind] )   m_versionMin[ind] = typevers;
  if ( typevers > m_versionMax[ind] )   m_versionMax[ind] = typevers;				        
  if ( srclog   < m_srclogMin[ind] )    m_srclogMin[ind]  = srclog;
  if ( srclog   > m_srclogMax[ind] )    m_srclogMax[ind]  = srclog;
  if ( srcphy   < m_srcphyMin[ind] )    m_srcphyMin[ind]  = srcphy;
  if ( srcphy   > m_srcphyMax[ind] )    m_srcphyMax[ind]  = srcphy;
  if ( level    < m_srclevelMin[ind] )  m_srclevelMin[ind]= level;
  if ( level    > m_srclevelMax[ind] )  m_srclevelMax[ind]= level;
  if ( depth    < m_depthMin[ind] )     m_depthMin[ind]   = depth;
  if ( depth    > m_depthMax[ind] )     m_depthMax[ind]   = depth; 
  if ( std::find(v_src_names[ind].begin(), v_src_names[ind].end(), src_name) == v_src_names[ind].end() )
    v_src_names[ind].push_back(src_name); // if the src_name is not in v_src_names[ind], then add it

  for (unsigned bit=0; bit<17; bit++) {
    if (dam_val & 1<<bit) {
      m_damageCounter[ind][bit] ++;
      m_damageCounterForTypeId[ind] ++;
      m_damageCounterForBit[bit] ++;
    }
  }
}

//===================

void QCStatistics::printTransIdCounter() // unsigned m_transIdCounter[]
{
  std::cout << "\n\nQCStatistics::printTransIdCounter()\nTotal number of known TransitionId : " << Pds::TransitionId::NumberOf << "\n";
  for(int id=0; id < (int)Pds::TransitionId::NumberOf; id++) {
    Pds::TransitionId::Value tid = (Pds::TransitionId::Value) id;
    std::cout << std::setw(16) << std::left << Pds::TransitionId::name(tid) << ": "  << std::setw(6) << m_transIdCounter[id] << "    "; 
    if (id&1) std::cout << "\n";
  }
  std::cout << "\n";
}

//===================

void QCStatistics::printTypeIdCounter() // unsigned m_typeIdCounter[]
{
  std::cout << "\nQCStatistics::printTypeIdCounter()\nTotal number of known TypeId : " << Pds::TypeId::NumberOf << "\n";
  std::cout << "Ind: TypeId name             : Entries Vers       payload                 Depth     Src:Level    Src:Names\n";  
  std::cout << "                             :         min/max    min/max     : #zeros    min/max   min/max \n";

  for(int id=0; id < (int)Pds::TypeId::NumberOf; id++) {
    Pds::TypeId::Type tid = (Pds::TypeId::Type) id;
    std::cout << std::setw(3)  << std::right << id  << ": " 
              << std::setw(24) << std::left  << Pds::TypeId::name(tid) 
              << ": "      << std::setw(6)   << m_typeIdCounter[id]; 
    if (m_typeIdCounter[id] == 0 ) { std::cout << "\n"; continue;}

    std::cout << "  "   << std::setw(3)   << std::right << m_versionMin[id] 
              << "/"                      << std::left  << m_versionMax[id] 
              << " "    << std::setw(8)   << std::right << m_payloadMin[id] 
              << "/"    << std::setw(8)   << std::left  << m_payloadMax[id] 
              << ": "   << std::setw(8)   << std::left  << m_payloadZeroCounter[id] 
            //<< " "    << std::setw(8)   << std::right << m_srclogMin[id] 
            //<< "/"    << std::setw(8)   << std::left  << m_srclogMax[id] 
            //<< " "    << std::setw(10)  << std::right << m_srcphyMin[id] 
            //<< "/"    << std::setw(10)  << std::left  << m_srcphyMax[id] 
              << " "    << std::setw(4)   << std::right << m_depthMin[id] 
              << "/"    << std::setw(4)   << std::left  << m_depthMax[id] 
              << " "    << std::setw(4)   << std::right << m_srclevelMin[id] 
              << "/"    << std::setw(4)   << std::left  << m_srclevelMax[id]
              << "    ";

    unsigned counter=0;
    for(std::vector<std::string>::const_iterator it  = v_src_names[id].begin();
                                                 it != v_src_names[id].end(); it++) {
      counter++; if(counter>1) std::cout << std::setw(93) << " " << std::setw(4);
      std::cout << " " << *it << "\n";
    }

  }
  std::cout << "\n";
}

//===================

void QCStatistics::printDamageCounter() // unsigned m_damageCounter[]
{
  std::string strNA = "N/A-";

  std::cout << "\nQCStatistics::printDamageCounter()\nTotal number of known TypeId : " << Pds::TypeId::NumberOf << "\n";
  std::cout << "Ind: TypeId name             : Entries  DmgTot  Bits:";
    for (unsigned bit=1; bit<17; bit++) {
      if (std::string(nameOfDamageBit(bit)).substr(0,4) == strNA) continue;
      std::cout  << std::setw(8) << std::left << bit;  // << std::setfill('0') 
    }
    std::cout << "\n";

  for(int id=0; id < (int)Pds::TypeId::NumberOf; id++) {
    Pds::TypeId::Type tid = (Pds::TypeId::Type) id;
    std::cout << std::setw(3)  << std::right << id  << ": " 
              << std::setw(24) << std::left  << Pds::TypeId::name(tid) 
              << ": "      << std::setw(7)   << m_typeIdCounter[id]; 
    if (m_typeIdCounter[id] == 0 ) { std::cout << "\n"; continue;}

    std::cout << std::setw(8) << std::right << m_damageCounterForTypeId[id];

    for (unsigned bit=1; bit<17; bit++) {
      if (std::string(nameOfDamageBit(bit)).substr(0,4) == strNA) continue;
      std::cout << std::setw(8) << std::right  << m_damageCounter[id][bit];
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

//===================

void QCStatistics::printDamageBitCounter()
{  
  std::cout << "\nQCStatistics::printDamageBitCounter()   Damage counter\n";
  for (unsigned bit=1; bit<17; bit++) {
    if (std::string(nameOfDamageBit(bit)).substr(0,4) == (std::string)"N/A-") continue;

    std::cout << "Damage bit:" << std::setw(2)  << bit << " : "  
              << std::left     << std::setw(24) << nameOfDamageBit(bit)  
              << std::right    << std::setw(8)  << m_damageCounterForBit[bit] << "\n"; 
  }
}      

//===================

void QCStatistics::saveBrokenTimeRecord(Pds::Dgram* dg, unsigned& ndgram)
{
  Pds::Xtc* xtc = &(dg->xtc);

  BrokenTimeRecord btr = {ndgram, xtc->damage.value(), dg->seq.clock(), m_t_prev};
  v_brokenTime.push_back(btr);
}

//===================

void QCStatistics::printBrokenTimeRecords()
{
  std::cout << "\nQCStatistics::printBrokenTimeRecords()\nTotal number of broken time records : " << v_brokenTime.size() << "\n";

  for( vector<BrokenTimeRecord>::const_iterator p  = v_brokenTime.begin();
                                                p != v_brokenTime.end(); p++ ) {
    time_t t0_sec  = p->time_dg     .seconds();
    time_t t0_nsec = p->time_dg     .nanoseconds();
    time_t t0_msec = (time_t)(1e-6 * t0_nsec);

    time_t t1_sec  = p->time_dg_prev.seconds();
    time_t t1_nsec = p->time_dg_prev.nanoseconds();
    time_t t1_msec = (time_t)(1e-6 * t1_nsec);

    struct tm* tm0; tm0 = localtime(&t0_sec); 
    struct tm* tm1; tm1 = localtime(&t1_sec); 
    char   ct0_buf[40];  strftime (ct0_buf,80,"%Y-%m-%d %H:%M:%S",tm0);
    char   ct1_buf[40];  strftime (ct1_buf,80,"%H:%M:%S",tm1);
 
    uint32_t dam_value = p->damage_value;

    std::cout << "BrokenTimeRecord: Dgram:" << std::setw(6) << std::right  << p->ndg 
              << "  t:"           << ct0_buf << "." << std::right << std::setfill('0') << std::setw(3) << t0_msec
	      << "  t-previous:"  << ct1_buf << "." << std::right << std::setfill('0') << std::setw(3) << t1_msec
	      << "  damage:"      << dam_value 
              << "\n" << std::setfill(' ');  

    printDamagedBits(dam_value);
  }
}
 
//===================

void QCStatistics::checkDgramTimeSequense(Pds::Dgram* dg, unsigned& ndgram)
{
    if( m_t_prev > dg->seq.clock() ) {
      std::cout << "Datagram:" << ndgram 
                << "  time: "                    << dg->seq.clock().seconds() << "."  << std::setw(9) << std::setfill('0') << dg->seq.clock().nanoseconds()
                << "  less then previous time:"  << m_t_prev       .seconds() << "."  << std::setw(9) << std::setfill('0') << m_t_prev       .nanoseconds()
                << "\n" << std::setfill(' ');

      saveBrokenTimeRecord(dg, ndgram);
    }
    m_t_prev = dg->seq.clock();
}

//===================
// Check everything for Dgram
void QCStatistics::processDgram(Pds::Dgram* dg, unsigned ndgram, unsigned long long dg_first_byte)
{    
    this->checkDgramTimeSequense(dg,ndgram);              // Check Dgram time sequense
    this->accumulateDgramStatistics(dg);                  // counts TransitionID etc.

    if(  (ndgram < 5) 
      || (ndgram < 50 && ndgram%10==0) 
      || (ndgram%100==0) ) 
      printDgramHeader(dg,ndgram,dg_first_byte);          // Print Dgram header
}

//===================

void QCStatistics::saveDamageRecord(Pds::Xtc* xtc, unsigned depth, unsigned ndgram)
{
  DamageRecord damage_record = {ndgram, depth, xtc->damage.value(), xtc->contains.id()};
  v_damageRecords.push_back(damage_record);
}

//===================

void QCStatistics::printDamageRecords()
{
  std::cout << "\nQCStatistics::printDamageRecords()\nTotal number of damage records : " << v_damageRecords.size() << "\n";

  for( vector<DamageRecord>::const_iterator p  = v_damageRecords.begin();
                                            p != v_damageRecords.end(); p++ ) {

    std::cout << "Damage Record: Dgram:" << std::setfill(' ') << std::setw(6) << std::right  << p->ndg 
	      << "  depth:"       << p->depth
	      << "  type:"        << std::left << std::setw(24) << Pds::TypeId::name(p->type_id)
	      << "  damage:"      << std::right<< std::setw(6)  << p->damage_value ;  

    //printDamagedBits(p->damage_value);
    // Print damaged bits
    uint32_t value = p->damage_value;
    if (!value) return;
      unsigned counter=0;
      for (unsigned bit=0; bit<17; bit++) {
        if (value & 1<<bit) { 
	  counter++; if (counter>1) std::cout << std::setw(82) << " " << std::setw(4);
          std::cout << "  bit:" << std::setw(2) << bit << ": " << std::left << nameOfDamageBit(bit) << "\n"; 
        }
      }
    }
}

//===================

bool QCStatistics::processXTC(Pds::Xtc* xtc, unsigned depth, unsigned ndgram) 
{
    // Check damage
    uint32_t damage_value = xtc->damage.value();

    if (damage_value & 0xffffffff) {
      std::cout << "Datagram:" << ndgram << "  Depth:" << depth << "  Type:" << Pds::TypeId::name(xtc->contains.id());
      printDamagedBits(damage_value);
      saveDamageRecord(xtc, depth, ndgram);
    }
 
    // Work on xtc
    //std::cout << "Depth:" << depth; printXtcHeader( (*xtc) );
    accumulateXTCStatistics(xtc,depth);

    // Fatal damage -> stop
    if (damage_value & ( 1 << Pds::Damage::IncompleteContribution)) return (int)Stop;
    return (int)Continue;
}

//===================

void QCStatistics::processXTCSizeError(Pds::Xtc* root, Pds::Xtc* xtc, int remaining, unsigned depth, unsigned ndgram)
{
  SizeErrorRecord size_error_record = {ndgram, depth, remaining, root->damage.value(), xtc->damage.value(), root->contains.id(), xtc->contains.id()};
  v_sizeErrorRecords.push_back(size_error_record);
}
//===================

void QCStatistics::printXTCSizeErrorRecords()
{
  std::cout << "\nQCStatistics::printXTCSizeErrorRecords()\nTotal number of size error records : " << v_sizeErrorRecords.size() << "\n";

  for( vector<SizeErrorRecord>::const_iterator  p  = v_sizeErrorRecords.begin();
                                                p != v_sizeErrorRecords.end(); p++ ) {

    std::cout << "Size Error Record: Dgram:" 
              << std::setw(6) << std::right  << p->ndg 
	      << "  depth:"                  << p->depth
	      << "  remaining:"              << p->remaining
	      << "  damage ext:"             << p->damage_ext
	      << "  damage int:"             << p->damage_int
	      << "  name ext:"               << Pds::TypeId::name(p->typeid_ext)
	      << "  name int:"               << Pds::TypeId::name(p->typeid_int)
              << "\n";  
    printDamagedBits(p->damage_ext);
  }
}

//===================

void QCStatistics::printQCSummary(unsigned ndgram)
{
  std::cout << "Total number of dgrams=" << ndgram << "\n";
  printTransIdCounter();
  printTypeIdCounter();
  printDamageCounter();
  printDamageBitCounter();
  printDamageRecords();
  printBrokenTimeRecords();
  printXTCSizeErrorRecords();
}

//===================
//===================
// Global methods
//===================
//===================

void printDgramHeader(Pds::Dgram* dg, unsigned& ndgram, unsigned long long& dg_first_byte)
{
    time_t t_sec  = dg->seq.clock().seconds();
    time_t t_nsec = dg->seq.clock().nanoseconds();
    int    t_msec = (int)(1e-6 * t_nsec);
    struct tm* stm; stm = localtime(&t_sec); 
    char   ct_buf[40];  strftime (ct_buf,80,"%Y-%m-%d %H:%M:%S",stm);

    printf("Datagram:%d  Transition:%s/%2d  seqtype:%1d  %s.%03d  fiducial:%06x  ticks:%06x  pos 0x%x \n",  // dmg %08x  payloadSize 0x%x  size(always=40) %u 
             ndgram,
             Pds::TransitionId::name(dg->seq.service()),
             dg->seq.service(), // TransitionId id 
             dg->seq.type(),
             ct_buf,
             t_msec,
             dg->seq.stamp().fiducials(),
             dg->seq.stamp().ticks(),
             //dg->xtc.damage.value(),
             //dg->xtc.sizeofPayload(),
	     //(int)sizeof(*dg),
	     (uint)dg_first_byte);
}

//===================

void printXtcHeader(Pds::Xtc& xtc)
{
  std::stringstream ssrc; ssrc << xtc.src;

  std::cout // << std::hex << std::showbase << std::setw(8) << std::setfill('0') 
            << "  XTC header: dmg:"  << xtc.damage.value() 
            << "  payload:"   << xtc.sizeofPayload()
            << "  extent:"    << xtc.extent
            << "  Src:Level:" << (int)xtc.src.level()
            << "  "           << Pds::Level::name(xtc.src.level()) 
            << ":"            << ssrc.str()
          //<< " log:"        << ((xtc.src.log())&0xffffff) // mask removes 8-bit level 
          //<< " phy:"        << xtc.src.phy()
            << "  TypeId:ver:"<< xtc.contains.version()
            << " id:"         << xtc.contains.id()
            << ":"            << Pds::TypeId::name(xtc.contains.id())
            << std::dec
            << "\n";
}      

//===================

const char* nameOfDamageBit(unsigned bit)
{ 
  static const char* _namesOfDamageBit[] = {
    "N/A-0",
    "DroppedContribution",    //  1 
    "N/A-2","N/A-3","N/A-4","N/A-5","N/A-6","N/A-7","N/A-8","N/A-9","N/A-10","N/A-11",
    "OutOfOrder",             // 12 
    "OutOfSynch",             // 13 
    "UserDefined",            // 14 
    "IncompleteContribution", // 15 
    "ContainsIncomplete",     // 16
    "N/A-17","N/A-18","N/A-19","N/A-20","N/A-21","N/A-22","N/A-23","N/A-24","N/A-25","N/A-26","N/A-27","N/A-28","N/A-29","N/A-30","N/A-31"
  };
  return (bit < 32 ? _namesOfDamageBit[bit] : "-Invalid-");
}

//===================

const char* nameOfDamageBit(Pds::Damage::Value vid)
{ 
  return nameOfDamageBit( (unsigned) vid );
}

//===================

void printDamagedBits(uint32_t value)
{
  if (!value) return;
    for (unsigned bit=0; bit<17; bit++) {
      if (value & 1<<bit) { std::cout << "  Damage bit " << std::setw(2) << bit << ": " << nameOfDamageBit(bit) << "\n"; }
    }
}      

//===================

void printListOfDamagedBits()
{  
  std::string strNA = "N/A-";
  std::cout << "\nPSXtcQC::printListOfDamagedBits() List of defined damage bits\n";
  for (unsigned bit=1; bit<17; bit++) {

    if (std::string(nameOfDamageBit(bit)).substr(0,4) == strNA) continue;

    std::cout << "Damage bit:" << std::setw(2) << bit 
              << " : "         << nameOfDamageBit(bit) << "\n"; 
  }
}      

//===================

} // namespace PSXtcQC
