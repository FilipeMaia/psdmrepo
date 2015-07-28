//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ExampleDumpImg...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ExampleDumpImg.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// header from psddl_psana package
#include "psddl_psana/cspad.ddl.h"
//#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ExampleDumpImg)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ExampleDumpImg::ExampleDumpImg (const std::string& name)
  : Module(name)
  , m_source()
  , m_key()
  , m_print_bits()
  , m_row_dump()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_source     = configSrc("source", ":Cspad");
  m_key        = configStr("key",          "");
  m_print_bits = config   ("print_bits",  255);
  m_row_dump   = config   ("row_dump",    200); 
}

ExampleDumpImg::~ExampleDumpImg ()
{
}

/// Method which is called once at the beginning of the job
void 
ExampleDumpImg::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
ExampleDumpImg::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ExampleDumpImg::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ExampleDumpImg::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) MsgLog(name(), info, "Event: " << m_count);

  procEvent(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
ExampleDumpImg::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ExampleDumpImg::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ExampleDumpImg::endJob(Event& evt, Env& env)
{
}

//-----------------------------

void 
ExampleDumpImg::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters :"
        << "\n source                   : " << m_source
        << "\n m_key                    : " << m_key      
        << "\n m_print_bits             : " << m_print_bits
        << "\n One of data types        : " << typeid(data_t).name() << " of size " << sizeof(data_t)
        << "\n";     
  }
}

//-----------------------------

void 
ExampleDumpImg::procEvent(Event& evt)
{
  if ( procEventForType<int16_t>(evt) ) return;
  if ( procEventForType<int>(evt) ) return;
  if ( procEventForType<double>(evt) ) return;
  if ( procEventForType<data_t>(evt) ) return;

  MsgLog(name(), warning, "ndarray object is not available in the event(...) for source:" << m_source << " key:" << m_key);
}

//-----------------------------

template <typename T>
bool 
ExampleDumpImg::procEventForType(Event& evt)
{
     shared_ptr< ndarray<const T,2> > shp = evt.get(m_source, m_key, &m_src);
     if (shp.get()) { 

         const T* data_arr = shp->data(); 
         const ndarray<const T,2>& nda = *shp.get(); 

	 //size_t rows = nda.shape()[0]; // = 1750 for CSPAD;
	 size_t cols = nda.shape()[1]; // = 1750 for CSPAD;


	 // Print ndarray using stream
	 if( m_print_bits & 4 ) {
	   std::cout << "\nPrint ndarray using stream operator:\n";
           std::cout << nda << "\n";
	 }

         if( m_print_bits & 8 ) {
	   std::cout << "\nPrint ndarray using pointer to its data:\n";
	   int i0 = m_row_dump * cols + cols/3;
	   for (int i=i0; i<i0+10; ++i) std::cout << " " << data_arr[i]; 
           std::cout << "\n";
	 }
	 
         if( m_print_bits & 16 ) {
	   std::cout << "\nPrint ndarray<T,2> using two indexes:\n";
	   for (unsigned row=m_row_dump; row<m_row_dump+10; ++row) {
	     for (unsigned col=cols/3; col<cols/3+10; ++col)
               std::cout << " " << nda[row][col]; 
             std::cout << "\n";
	   }
	 }
	 
         if( m_print_bits & 32 ) {
	   std::cout << "\nPrint ndarray using iterator:\n";
	   typename ndarray<const T, 2>::iterator it;
	   //for ( it=nda.begin(); it!=nda.end(); ++it) {
	     for ( it=&nda[m_row_dump][0]; it!=&nda[m_row_dump][cols]; ++it) {
             std::cout << " " << *it;
           }
           std::cout << "\n";
	 }
	 
         return true; 
     } 

     return false;
}

//-----------------------------

} // namespace ImgAlgos

//-----------------------------
