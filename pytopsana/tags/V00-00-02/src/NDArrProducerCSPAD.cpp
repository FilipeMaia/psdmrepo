//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//---------------
// -- Headers --
//---------------

#include "pytopsana/NDArrProducerCSPAD.h"

//-----------------------------
namespace pytopsana {

typedef NDArrProducerCSPAD::data_t data_t;

//-----------------------------

NDArrProducerCSPAD::NDArrProducerCSPAD(const PSEvt::Source& source)
  : NDArrProducerBase(source)
  , m_as_data(false)
  , m_numSect(N2x1InCSPAD)
  , m_count_evt(0)
  , m_count_cfg(0)
  , m_count_msg(0)
{
  for (uint32_t q=0; q<NQuadsMax; ++q) m_roiMask[q] = 0xff;

  //m_dettype = ImgAlgos::detectorTypeForSource(m_source);
  //std::cout << "m_nda_def.size() = " << m_nda_def.size() << '\n';
  //std::stringstream ss; ss << source;
  //std::string str_src = ss.str();
}

//-----------------------------

/*
NDArrProducerCSPAD::NDArrProducerCSPAD(const std::string& str_src)
  : NDArrProducerBase(str_src)
{
  NDArrProducerCSPAD(PSEvt::Source(str_src));
}
*/

//-----------------------------

NDArrProducerCSPAD::~NDArrProducerCSPAD ()
{
}

//-----------------------------

void 
NDArrProducerCSPAD::getConfigPars(PSEnv::Env& env)
{
  if ( getConfigParsForType<Psana::CsPad::ConfigV2>(env) ) return;
  if ( getConfigParsForType<Psana::CsPad::ConfigV3>(env) ) return;
  if ( getConfigParsForType<Psana::CsPad::ConfigV4>(env) ) return;
  if ( getConfigParsForType<Psana::CsPad::ConfigV5>(env) ) return;

  m_count_msg++;
  if (m_count_msg < 11 && m_pbits) {
    MsgLog(name(), warning, "CsPad::ConfigV2-V5 is not available in this run, event:" << m_count_evt << " for source:" << m_str_src);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src);
  }
}

//-----------------------------

ndarray<data_t,3>  
NDArrProducerCSPAD::getNDArr(PSEvt::Event& evt, PSEnv::Env& env)
{
  m_count_evt ++;

  if ( m_count_cfg==0 ) getConfigPars(env);
  if ( m_count_cfg==0 ) return m_nda_def;

  // Check if the requested src and key are consistent with Psana::CsPad::DataV1, or V2
  ndarray<data_t,3> nda1 = getNDArrForType<Psana::CsPad::DataV1, Psana::CsPad::ElementV1, data_t>(evt);
  if (nda1.size()) return nda1;

  ndarray<data_t,3> nda2 = getNDArrForType<Psana::CsPad::DataV2, Psana::CsPad::ElementV2, data_t>(evt);
  if (nda2.size()) return nda2;

  m_count_msg++;
  if (m_count_msg < 11 && m_pbits) {
    MsgLog(name(), warning, "procEvent(...): Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in event:"
	   << m_count_evt << " for source:" << m_str_src << " key:\"" << m_key << '\"');
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:"
           << m_str_src << " key:\"" << m_key << '\"');
  }
  return m_nda_def;
}

//-----------------------------
} // namespace pytopsana
//-----------------------------
