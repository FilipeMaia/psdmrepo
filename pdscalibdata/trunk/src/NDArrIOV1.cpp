//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
// 	$Revision$
//
// Author: Mikhail Dubrovin
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/NDArrIOV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>   // for stringstream
#include <stdlib.h> // for atoi
#include <cstring>  // for memcpy

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

namespace pdscalibdata {

//-----------------------------

template <typename TDATA, unsigned NDIM>
NDArrIOV1<TDATA, NDIM>::NDArrIOV1 ( const std::string& fname
				  , const shape_t* shape_def
				  , const TDATA& val_def 
                                  , const unsigned print_bits ) 
  : p_nda(0)
  , m_fname(fname)
  , m_val_def(val_def)
  , m_print_bits(print_bits)
{
  if (shape_def != 0) std::memcpy (m_shape, shape_def, c_ndim*sizeof(shape_t));  
  MsgLog(__name__(), debug, "ctor, fname=" << fname);
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::load_ndarray()
{
    // if file is not available - create default ndarray
    if (! file_is_available()) { 
        create_ndarray(true); 
        return; 
    }

    if( m_print_bits & 1 ) MsgLog(__name__(), info, "Load file " << m_fname);

    m_count_str_data = 0;
    m_count_str_comt = 0;
    m_count_data     = 0;

    // open file
    std::ifstream in(m_fname.c_str());
    if (not in.good()) { MsgLogRoot(error, "Failed to open file: "+m_fname); return; }
  
    // read and process all strings
    std::string str; 
    while(getline(in,str)) { 

        // 1. parse lines with comments marked by # in the 1st position
        if(str[0] == '#') parse_str_of_comment(str.substr(1));

        // 2. skip empty lines 
        else if (str.find_first_not_of(" ")==string::npos) continue; 

        // 3. parse 1st line and load other data
        else load_data(in,str);
    }

    //close file
    in.close();
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
bool NDArrIOV1<TDATA, NDIM>::file_is_available()
{
  if(m_fname.empty()) {
    MsgLog(__name__(), warning, "File name IS EMPTY!");
    return false;
  }

  std::ifstream file(m_fname.c_str());
  if(!file.good()) {
      MsgLog(__name__(), warning, "File: " << m_fname << " DOES NOT EXIST!");
      return false;
  }
  file.close();
  return true;  
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::parse_str_of_comment(const std::string& str)
{
    m_count_str_comt ++;
    // cout << "comment, str.size()=" << str.size() << '\n';

    std::string field;
    std::stringstream ss(str);

    ss >> field;


    if (field=="TYPE") { 
       ss >> m_str_type;
       m_enum_type = enumDataTypeForString(m_str_type);

       if (m_enum_type != enumDataType<TDATA>()) {
	   std::stringstream smsg; 
	   smsg << "(enum) TYPE in file metadata: " << m_enum_type 
                << " is different from declaration: " << enumDataType<TDATA>();
           MsgLog(__name__(), warning, smsg.str());
       }
    }

    else if (field=="NDIM") {
      ss >> m_ndim;
      if (m_ndim != ndim()) {	
	std::stringstream smsg; 
        smsg << "NDIM in file metadata: " << m_ndim 
             << " is different from declaration: " << ndim();
        // MsgLog(__name__(), error, smsg.str());
        throw std::runtime_error(smsg.str());
      }
    }

    else if (field.substr(0,4)=="DIM:") { 
        //cout << "field.substr(0,4)" << field.substr(0,4) << "field[4]" << field[4] << endl;
        int dim = atoi(&field[4]); 
	shape_t val;    
        ss >> val;
	if (m_shape[dim] !=val) {
	   std::stringstream smsg; 
	   smsg << "NDArray metadata shape field " << field
                << " = " << val 
	        << " is different from expected " << m_shape[dim] 
                << " in file " << m_fname
	        << "\nCheck that calibration file has expected shape and data...";
           MsgLogRoot(error, smsg.str());
           throw std::runtime_error(smsg.str());
	   // override or not ?
	   //m_shape[dim] = val;
	   //MsgLog(__name__(), debug, "Dimension: " << dim << " set to stride: " << m_shape[dim] );
	}
    }

    else 
      //MsgLog(__name__(), info, "Ignore comment: " << str );
      return;
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::create_ndarray(const bool& fill_def)
{
    p_nda = new ndarray<TDATA, NDIM>(m_shape);
    p_data = p_nda->data();
    m_size = p_nda->size();

    if (fill_def) std::fill_n (p_data, m_size, m_val_def);

    //MsgLog(__name__(), info, "Begin to load data p_nda->size(): " << m_size);
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::load_data(std::ifstream& in, const std::string& str)
{
    if (! m_count_str_data++) create_ndarray();

    // parse the 1st string
    TDATA val;
    TDATA* it=p_data; 

    std::stringstream ss(str);
    while (ss >> val and m_count_data != m_size) { 
      *it++ = val;
      ++m_count_data;
      //cout << "count:data = " << m_count_data << " : " << val << '\n';
    }

    // load all data by the end
    while(in and m_count_data != m_size) {
      in >> *it++;
      ++m_count_data;
    }

    // check that we read whole array
    if (m_count_data != m_size) {
      const std::string msg = "NDArray file does not have enough data: "+m_fname;
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }

    // and no data left after we finished reading
    if ( in >> val ) {
      ++ m_count_data;
      const std::string msg = "NDArray file has extra data: "+m_fname;
      MsgLogRoot(error, msg);
      MsgLogRoot(error, "read " << m_count_data << " numbers, expecting " << m_size );
      throw std::runtime_error(msg);
    }
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
ndarray<const TDATA, NDIM> 
NDArrIOV1<TDATA, NDIM>::get_ndarray(const std::string& fname)
{
  if (!fname.empty() and fname != m_fname) {
    m_fname = fname;
    load_ndarray(); 
  }

  if (!p_nda) load_ndarray();
  if (!p_nda) MsgLog(__name__(), error, "ndarray IS NOT LOADED! Check file: " << m_fname );
  return *p_nda;
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::print()
{
    std::stringstream ss; 
    ss << "print():"
       << "\n  Number of dimensions : " << ndim()
       << "\n  Data type and size   : " << strOfDataTypeAndSize<TDATA>()
       << "\n  Enumerated data type : " << enumDataType<TDATA>()
       << "\n  String data type     : " << strDataType<TDATA>()
       << '\n';
    MsgLog(__name__(), info, ss.str());
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::print_file()
{
    if (! file_is_available() ) return;

    MsgLog(__name__(), info, "print_file()\nContent of the file: " << m_fname);

    // open file
    std::ifstream in(m_fname.c_str());
    if (not in.good()) { MsgLogRoot(error, "Failed to open file: "+m_fname); return; }
  
    // read and dump all fields
    //std::string s; while(in) { in >> s; cout << s << " "; }

    // read and dump all strings
    std::string str; 
    while(getline(in,str)) cout << str << '\n';
    cout << '\n';
    //close file
    in.close();
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::print_ndarray()
{
    if (! p_nda) load_ndarray();
    if (! p_nda) return;

    std::stringstream smsg; 
    smsg << "Print ndarray<" << strDataType<TDATA>() 
         << "," << ndim()
         << "> of sise=" << p_nda->size()
         << ":\n" << *p_nda;
    MsgLog(__name__(), info, smsg.str());
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void NDArrIOV1<TDATA, NDIM>::save_ndarray(const ndarray<const TDATA, NDIM>& nda, 
                                          const std::string& fname,
                                          const std::vector<std::string>& vcoms)
{
    const unsigned ndim = NDIM;
    std::string str_dtype = strDataType<TDATA>();
    std::stringstream sstype; sstype << "ndarray<" << str_dtype 
                                     << "," << ndim << ">";

    std::stringstream smsg; 
    smsg << "Save " << sstype.str()
         << " of size=" << nda.size()
         << " in file: " << fname;
    MsgLog(__name__(), info, smsg.str());

    // open file
    std::ofstream out(fname.c_str());
    if (not out.good()) { MsgLogRoot(error, "Failed to open output file: " + fname); return; }
  
    // write comments if available
    if (!vcoms.empty()) {
      for(vector<string>::const_iterator it = vcoms.begin(); it != vcoms.end(); it++)
        out << "# " << *it << '\n';
      out << '\n';
    }

    // write permanent comments
    out << "# DATE_TIME  " << strTimeStamp() << '\n';
    out << "# AUTHOR     " << strEnvVar("LOGNAME") << '\n';
    out << '\n';

    // write metadata
    out << "# Metadata for " << sstype.str() << '\n';
    out << "# TYPE     " << str_dtype << '\n';
    out << "# NDIM     " << ndim << '\n';
    //shape_t shape = nda.shape()
    for(unsigned i=0; i<ndim; i++) out << "# DIM:" << i << "    " << nda.shape()[i] << '\n';
    out << '\n';

    // save data
    unsigned nmax_in_line = (ndim>1) ? nda.shape()[ndim-1] : 10; 
    unsigned count_in_line=0; 

    typename ndarray<const TDATA, NDIM>::iterator it = nda.begin();
    for (; it!=nda.end(); ++it) {
      out << std::setw(10) << *it << " ";
      if( ++count_in_line < nmax_in_line) continue;
          count_in_line = 0;
          out << '\n';
    }

    //close file
    out.close();
}

//-----------------------------

} // namespace pdscalibdata

//-----------------------------
//-----------------------------
//-----------------------------

template class pdscalibdata::NDArrIOV1<int,1>; 
template class pdscalibdata::NDArrIOV1<float,1>; 
template class pdscalibdata::NDArrIOV1<double,1>; 

template class pdscalibdata::NDArrIOV1<int,2>; 
template class pdscalibdata::NDArrIOV1<float,2>; 
template class pdscalibdata::NDArrIOV1<double,2>; 

template class pdscalibdata::NDArrIOV1<int,3>; 
template class pdscalibdata::NDArrIOV1<float,3>; 
template class pdscalibdata::NDArrIOV1<double,3>; 

template class pdscalibdata::NDArrIOV1<int,4>; 
template class pdscalibdata::NDArrIOV1<float,4>; 
template class pdscalibdata::NDArrIOV1<double,4>; 

template class pdscalibdata::NDArrIOV1<int,5>; 
template class pdscalibdata::NDArrIOV1<float,5>; 
template class pdscalibdata::NDArrIOV1<double,5>; 

//-----------------------------
//-----------------------------
//-----------------------------
