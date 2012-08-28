//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadMaskV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadMaskV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <stdexcept>
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

using namespace std;

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

CSPadMaskV1::CSPadMaskV1 ()
{
  // Fill the mask array by ones (transparent) by default
  std::fill_n(&m_mask[0][0][0][0], int(SIZE_OF_ARRAY), mask_t(1));
}

//----------------

CSPadMaskV1::CSPadMaskV1 (mask_t value)
{
  // Fill the mask array by value
  std::fill_n(&m_mask[0][0][0][0], int(SIZE_OF_ARRAY), value);
}

//----------------

CSPadMaskV1::CSPadMaskV1( const std::string& fname )
{
  MsgLog("CSPadMaskV1:",  info, "Read mask from file: " << fname.c_str() );
  // Open file

  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open the mask file: "+fname+"\nWill use default mask of all units.";
    MsgLogRoot(error, msg);
    //throw std::runtime_error(msg);
    CSPadMaskV1 (1);
    return;
  }

  // Read the entire file content in vector
  std::vector<mask_t> v_pars;
  std::string str;  
  do{ 
      in >> str; 
      if   ( in.good() ) v_pars.push_back(mask_t(std::atof(str.c_str()))); 
    } while( in.good() );

  // Close file
  in.close();

  // Check and copy the vector in array 
  fillArrFromVector(v_pars);
}

//----------------

void CSPadMaskV1::fillArrFromVector( const std::vector<mask_t> v_parameters )
{
  //cout << "\nCSPadMaskV1:\n";
    if (v_parameters.size() != SIZE_OF_ARRAY) {
        MsgLog("CSPadMaskV1", error, 
        "Expected number of parameters is " << SIZE_OF_ARRAY 
        << ", read from file " << v_parameters.size()
        << ": check the file."
	)
        const std::string msg = "The data size available in file for CSPad mask is wrong.";
        MsgLogRoot(error, msg);
        throw std::runtime_error(msg);
        //abort();
    }

    size_t arr_size = sizeof( mask_t ) * v_parameters.size();
    std::memcpy( &m_mask, &v_parameters[0], arr_size );
    //this->print();
}

//----------------

void CSPadMaskV1::print()
{
  MsgLog("CSPadMaskV1::print()",  info, "Print part of the data for test purpose only.");
    for (int iq = 0; iq != Quads; ++ iq) {
      cout << "Quad: " << iq << "\n"; 
      for (int is = 0; is != Sectors; ++ is) {
      cout << "Segment: " << is << "\n"; 
          //for (int ic = 0; ic != Columns; ++ ic) {
	  //for (int ir = 0; ir != Rows; ++ ir) {
	    for (int ic = 0; ic < 4; ++ ic) {
	    for (int ir = 0; ir < 10; ++ ir) {
	      //cout << "  " << iq << " " << is << " " << ic << " " << ir << " "; 
              cout << m_mask[iq][is][ic][ir] << " "; 
          }
              cout << endl;
        }
      }
    }
}

//----------------

void CSPadMaskV1::printMaskStatistics()
{

  mask_t* p = &m_mask[0][0][0][0];

  int Nof0=0;
  int Nof1=0;

  for (int i = 0; i < SIZE_OF_ARRAY; ++ i) {
      if (p[i] == 0) Nof0++;
      if (p[i] == 1) Nof1++;
  }

  MsgLog("CSPadMaskV1::printMaskStatistics()",  info, "Mask statistics: Nof0: " << Nof0 
                                       << " Nof1: " << Nof1
                                       << " Ntot: " << SIZE_OF_ARRAY
	                               << " Nof0 / Ntot = " << float(Nof0)/SIZE_OF_ARRAY );
}

//----------------
//--------------
// Destructor --
//--------------
CSPadMaskV1::~CSPadMaskV1 ()
{
}

} // namespace ImgAlgos
