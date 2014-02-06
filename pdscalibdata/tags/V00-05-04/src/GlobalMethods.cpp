//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace pdscalibdata;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
GlobalMethods::GlobalMethods ()
{
}

//--------------
// Destructor --
//--------------
GlobalMethods::~GlobalMethods ()
{
}

//--------------------

std::string 
stringFromUint(unsigned number, unsigned width, char fillchar)
{
  stringstream ssNum; ssNum << setw(width) << setfill(fillchar) << number;
  return ssNum.str();
}

//--------------------

void 
printSizeOfTypes()
{
  std::cout << "Size Of Types:" 
            << "\nsizeof(bool    ) = " << sizeof(bool    ) << " with typeid(bool    ).name(): " << typeid(bool    ).name() 
            << "\nsizeof(uint8_t ) = " << sizeof(uint8_t ) << " with typeid(uint8_t ).name(): " << typeid(uint8_t ).name()  
            << "\nsizeof(uint16_t) = " << sizeof(uint16_t) << " with typeid(uint16_t).name(): " << typeid(uint16_t).name()  
            << "\nsizeof(uint32_t) = " << sizeof(uint32_t) << " with typeid(uint32_t).name(): " << typeid(uint32_t).name()  
            << "\nsizeof(int     ) = " << sizeof(int     ) << " with typeid(int     ).name(): " << typeid(int     ).name()  
            << "\nsizeof(int16_t ) = " << sizeof(int16_t ) << " with typeid(int16_t ).name(): " << typeid(int16_t ).name()  
            << "\nsizeof(int32_t ) = " << sizeof(int32_t ) << " with typeid(int32_t ).name(): " << typeid(int32_t ).name()  
            << "\nsizeof(float   ) = " << sizeof(float   ) << " with typeid(float   ).name(): " << typeid(float   ).name()  
            << "\nsizeof(double  ) = " << sizeof(double  ) << " with typeid(double  ).name(): " << typeid(double  ).name()  
            << "\n\n";
}

//--------------------
//--------------------
} // namespace pdscalibdata
