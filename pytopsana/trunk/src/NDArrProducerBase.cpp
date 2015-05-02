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

#include "pytopsana/NDArrProducerBase.h"

//-----------------------------
namespace pytopsana {

  //typedef NDArrProducerBase::data_t data_t;

//-----------------------------

NDArrProducerBase::NDArrProducerBase(const PSEvt::Source& source)
  : m_source(source)
  , m_key(std::string())
  , m_mode(0)
  , m_val_def(0.)
  , m_pbits(1)
  , m_count_msg(0)
{
  std::stringstream ss; ss << source;
  m_str_src = ss.str();
  m_dettype = ImgAlgos::detectorTypeForSource(m_source);

  if (m_pbits) print();  
}

//-----------------------------

NDArrProducerBase::NDArrProducerBase(const std::string& str_src = std::string())
{
  NDArrProducerBase(PSEvt::Source(str_src));
}

//-----------------------------

NDArrProducerBase::~NDArrProducerBase ()
{
}

//-----------------------------
//-----------------------------

/*
ndarray<data_t,3>  
NDArrProducerBase::getNDArr(PSEvt::Event& evt, PSEnv::Env& env)
{
  return make_ndarray<data_t>(2,4,8);
}
*/

//-----------------------------

void
NDArrProducerBase::print()
{
  MsgLog(name(), info, "\n Input parameters:"
         << "\n source        : " << m_source
         << "\n str_src       : " << m_str_src
         << "\n key           : " << m_key      
         << "\n mode          : " << m_mode      
         << "\n val_def       : " << m_val_def
         << "\n pbits         : " << m_pbits
         << "\n dettype       : " << m_dettype
         << "\n detector      : " << ImgAlgos::stringForDetType(m_dettype)
	 );
}
//-----------------------------

void
NDArrProducerBase::print_def(const char* method)
{
  MsgLog(name(), info, "Default method: " << method << " should be re-implemented in the derived class"); 
}

//-----------------------------
} // namespace pytopsana
//-----------------------------
