//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GeometryObject...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/GeometryObject.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream> // for cout
#include <sstream>  // for stringstream
#include <iomanip>  // for setw, setfill
#include <cmath>    // for sqrt, atan2, etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSCalib {

//----------------
// Constructors --
//----------------
  GeometryObject::GeometryObject (  std::string pname,
                                    unsigned    pindex,
                                    std::string oname,
                                    unsigned    oindex,
                                    double      x0,
                                    double      y0,
                                    double      z0,
                                    double      rot_z,
                                    double      rot_y,
                                    double      rot_x,                  
                                    double      tilt_z,
                                    double      tilt_y,
                                    double      tilt_x 
				  )
    : m_pname  (pname) 
    , m_pindex (pindex)
    , m_oname  (oname)
    , m_oindex (oindex)
    , m_x0     (x0)    
    , m_y0     (y0)    
    , m_z0     (z0)  
    , m_rot_z  (rot_z) 
    , m_rot_y  (rot_y)
    , m_rot_x  (rot_x) 
    , m_tilt_z (tilt_z)
    , m_tilt_y (tilt_y)
    , m_tilt_x (tilt_x) 
{

  //m_pix_coords_2x1 = 0;

  if      ( m_oname == "SENS2X1:V1" ) { m_algo = SENS2X1V1; m_pix_coords_2x1 = new PC2X1(); }
  else if ( m_oname == "SENS2X1:V2" ) { m_algo = SENS2X1V2; }
  else                                { m_algo = NONDEF; }

  m_parent = shpGO();
  v_list_of_children.clear();
  p_xarr = 0;
  p_yarr = 0;
  p_zarr = 0;
  m_size = 0;
}

//--------------
// Destructor --
//--------------
GeometryObject::~GeometryObject ()
{
}

//-------------------
std::string GeometryObject::string_geo()
{
  std::stringstream ss;
  ss << "parent:"   << std::setw(10) << std::right << m_pname 
     << "  pind:"   << std::setw(2)  << m_pindex
     << "  geo:"    << std::setw(10) << m_oname 
     << "  oind:"   << std::setw(2)  << m_oindex << std::right << std::fixed
     << "  x0:"     << std::setw(8)  << std::setprecision(0) << m_x0    
     << "  y0:"     << std::setw(8)  << std::setprecision(0) << m_y0    
     << "  z0:"     << std::setw(8)  << std::setprecision(0) << m_z0    
     << "  rot_z:"  << std::setw(6)  << std::setprecision(1) << m_rot_z 
     << "  rot_y:"  << std::setw(6)  << std::setprecision(1) << m_rot_y 
     << "  rot_x:"  << std::setw(6)  << std::setprecision(1) << m_rot_x 
     << "  tilt_z:" << std::setw(8)  << std::setprecision(5) << m_tilt_z
     << "  tilt_y:" << std::setw(8)  << std::setprecision(5) << m_tilt_y
     << "  tilt_x:" << std::setw(8)  << std::setprecision(5) << m_tilt_x;
  return ss.str();
}

//-------------------
void GeometryObject::print_geo()
{
  //std::cout << string_geo() << '\n';
  MsgLog(name(), info, string_geo());
}

//-------------------
std::string GeometryObject::string_geo_children()
{
  std::stringstream ss;
  ss << "parent:" << std::setw(10) << std::left << m_pname 
     << " i:"     << std::setw(2)  << m_pindex
     << "  geo:"  << std::setw(10) << m_oname 
     << " i:"     << std::setw(2)  << m_oindex 
     << "  #children:" << v_list_of_children.size();
  for(std::vector<shpGO>::iterator it  = v_list_of_children.begin(); 
                                   it != v_list_of_children.end(); ++ it) {
    ss << " " << (*it)->get_geo_name() << " " << (*it)->get_geo_index(); 
  }
  return ss.str();
}

//-------------------
void GeometryObject::print_geo_children()
{
  //std::cout << string_geo_children() << '\n';
  MsgLog(name(), info, string_geo_children());
}

//-------------------
void GeometryObject::transform_geo_coord_arrays(const double* X, 
                                                const double* Y,  
                                                const double* Z, 
                                                const unsigned size,
                                                double*  Xt,  
                                                double*  Yt,  
                                                double*  Zt,
                                                const bool do_tilt)
{
  // take rotation ( +tilt ) angles in degrees
  double angle_x = (do_tilt) ? m_rot_x + m_tilt_x : m_rot_x;
  double angle_y = (do_tilt) ? m_rot_y + m_tilt_y : m_rot_y;
  double angle_z = (do_tilt) ? m_rot_z + m_tilt_z : m_rot_z;

  // allocate memory for intermediate transformation
  double* X1 = new double [size];
  double* Y1 = new double [size];
  double* Z2 = new double [size];

  // apply three rotations around Z, Y, and X axes
  rotation(X,  Y,  size, angle_z, X1, Y1);
  rotation(Z,  X1, size, angle_y, Z2, Xt);
  rotation(Y1, Z2, size, angle_x, Yt, Zt);

  // apply translation
  for(unsigned i=0; i<size; ++i) {
    Xt[i] += m_x0;
    Yt[i] += m_y0;
    Zt[i] += m_z0;
  }

  // release allocated memory
  delete [] X1;
  delete [] Y1;
  delete [] Z2;
}

//-------------------
unsigned GeometryObject::get_size_geo_array()
{
  if(m_algo == SENS2X1V1) return m_pix_coords_2x1 -> get_size();

  unsigned size=0;  
  for(std::vector<shpGO>::iterator it  = v_list_of_children.begin(); 
                                   it != v_list_of_children.end(); ++it) {
    size += (*it)->get_size_geo_array();
  }    
  return size;
}

//-------------------
void GeometryObject::get_pixel_coords(const double*& X, const double*& Y, const double*& Z, unsigned& size)
{
  if(p_xarr==0) evaluate_pixel_coords();
  X    = p_xarr;
  Y    = p_yarr;
  Z    = p_zarr;
  size = m_size;
}


//-------------------
void GeometryObject::evaluate_pixel_coords()
{
  // allocate memory for pixel coordinate arrays
  m_size = get_size_geo_array();

  p_xarr = new double [m_size];
  p_yarr = new double [m_size];
  p_zarr = new double [m_size];


  if(m_algo == SENS2X1V1) {
       const double* x_arr = m_pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_X, PC2X1::UM);
       const double* y_arr = m_pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_Y, PC2X1::UM);
       const double* z_arr = m_pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_Z, PC2X1::UM);

       transform_geo_coord_arrays(x_arr, y_arr, z_arr, m_size, p_xarr, p_yarr, p_zarr);
       return;
  }

  unsigned ibase=0;
  unsigned ind=0;
  for(std::vector<shpGO>::iterator it  = v_list_of_children.begin(); 
                                   it != v_list_of_children.end(); ++it, ++ind) {

    if((*it)->get_geo_index() != ind) {
      std::stringstream ss;
      ss << "WARNING! Geometry object:" << (*it)->get_geo_name() << ":" << (*it)->get_geo_index()
         << " has non-consequtive index at reconstruction: " << ind << '\n';
      //std::cout << ss.str();
      MsgLog(name(), warning, ss.str());
    }

    const double* pXch; 
    const double* pYch;
    const double* pZch; 
    unsigned      sizech;

    (*it)->get_pixel_coords(pXch, pYch, pZch, sizech);
       
    transform_geo_coord_arrays(pXch, pYch, pZch, sizech, &p_xarr[ibase], &p_yarr[ibase], &p_zarr[ibase]);
    ibase += sizech;
  }
}

//-------------------
//-------------------
//-------------------
//-------------------

void GeometryObject::rotation(const double* X, const double* Y, const unsigned size,
               const double C, const double S, 
               double* Xrot, double* Yrot)
  {
    for(unsigned i=0; i<size; ++i) {
      Xrot[i] = X[i]*C - Y[i]*S; 
      Yrot[i] = Y[i]*C + X[i]*S; 
    } 
  }

//-------------------

void GeometryObject::rotation(const double* X, const double* Y, const unsigned size, const double angle_deg, 
                double* Xrot, double* Yrot)
  {
    const double angle_rad = angle_deg * DEG_TO_RAD;
    const double C = cos(angle_rad);
    const double S = sin(angle_rad);
    rotation(X, Y, size, C, S, Xrot, Yrot);
  }

//-------------------

} // namespace PSCalib


//-------------------
//-------------------
//-------------------
