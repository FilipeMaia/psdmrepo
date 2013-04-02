//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: ImgParametersV1.cpp 2726 2012-08-21 14:50:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class ImgParametersV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgParametersV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream> // for std::cout
#include <sstream>
#include <iomanip>  // for std::setw

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgParametersV1::ImgParametersV1 (const unsigned* shape, pars_t val)
{
  m_fname= std::string("");
  m_rows = shape[0];
  m_cols = shape[1];
  m_size = m_rows * m_cols;
  m_pars = new pars_t [m_size];
  m_factor = 1;
  std::fill_n(m_pars, int(m_size), val);
}

//--------------------

ImgParametersV1::ImgParametersV1 (const std::string& fname, const unsigned* shape) 
{ 
  m_fname= fname;
  m_rows = shape[0];
  m_cols = shape[1];
  m_size = m_rows * m_cols;
  m_pars = new pars_t [m_size];
  m_factor = 1;

  // open file
  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open file: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // read all numbers
  pars_t* it = m_pars;
  size_t count = 0;
  while(in and count != m_size) {
    in >> *it++;
    ++ count;
  }

  // check that we read whole array
  if (count != m_size) {
    const std::string msg = "File "+fname+" does not have enough data: ";
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // and no data left after we finished reading
  float tmp ;
  if ( in >> tmp ) {
    const std::string msg = "File "+fname
                          +" has extra data; read:" + stringFromUint(count,10,' ') 
                          + " expecting:"           + stringFromUint(m_size,10,' ');
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }
}

//--------------------

ImgParametersV1::ImgParametersV1 (const std::string& fname, double factor) 
{ 
  m_fname  = fname;
  m_factor = (pars_t) factor;

  // open file
  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open file: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  v_work.clear();
  char buf[LINESIZEMAX];
  unsigned row = 0;
  unsigned cols=0, cols_prev=0;

  while(true) {
    in.getline(buf,LINESIZEMAX,EOL);
    if(!in.good()) break;
    std::string s = std::string(buf);
    cols = input_data_from_string(s);
    ++ row;
    // check if the number of input columns consistent with previous line
    if (row>1 && cols!=cols_prev) {
      const std::string msg = "The file "+fname
                            + " has a whong structure;\nrow:" + stringFromUint(row,5,' ')
			    + " has a number of cols:"        + stringFromUint(cols,5,' ')
			    + " different from previous:"     + stringFromUint(cols_prev,5,' ');
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }
    cols_prev = cols;
  } 
 
  m_rows = row;
  m_cols = cols;
  m_size = m_rows * m_cols;

  m_pars = new pars_t [m_size];
  std::memcpy(m_pars,&v_work[0],m_size*sizeof(pars_t)); // copy v_work -> m_pars
  v_work.clear();
}

//--------------------

unsigned ImgParametersV1::input_data_from_string(std::string& s)
{
  s+=" "; // THIS SPACE IS IMPORTANT: othervise stringstream skips the last value...

  std::stringstream ss(s); 
      unsigned col = 0;
      pars_t val;
      while(true) { 
        ss >> val; 
	if(!ss.good()) break;
        v_work.push_back(val * m_factor);
        col++;
      }
      return col;
}

//--------------
// Destructor --
//--------------

ImgParametersV1::~ImgParametersV1 ()
{
  delete [] m_pars;
  v_work.clear();
}

//--------------------

void ImgParametersV1::print(std::string comment)
{
  WithMsgLog("ImgParametersV1::print()", info, log) {
    log << "\n Private pars : " << comment
        << "\n fname        : " << m_fname
        << "\n rows         : " << m_rows
        << "\n cols         : " << m_cols
        << "\n size         : " << m_size
        << "\n factor       : " << m_factor
        << "\n pars         : "
        << "\n col:     ";

    unsigned col_min=5, col_max=20;
    unsigned row_min=5, row_max=30;
    unsigned width=8;
    for(unsigned c=col_min; c<col_max; c++) log << std::setw(width) << c << " ";
    log << "\n";

    for(unsigned r=row_min; r<row_max; r++) {
      log << "\n row" << std::setw(5) << r << ":" << std::fixed << std::setprecision(2);
      for(unsigned c=col_min; c<col_max; c++) log << std::setw(width) << m_pars[r*m_cols + c] << " ";
    }
    log << "\n";
  }
}

//--------------------
} // namespace ImgAlgos
//--------------------
