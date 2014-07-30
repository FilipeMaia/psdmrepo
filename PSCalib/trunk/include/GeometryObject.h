#ifndef PSCALIB_GEOMETRYOBJECT_H
#define PSCALIB_GEOMETRYOBJECT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GeometryObject.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <math.h>      // sin, cos

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "ndarray/ndarray.h"
//#include "CSPadPixCoords/PixCoords2x1V2.h"
#include "PSCalib/PixCoords2x1V2.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

typedef PSCalib::PixCoords2x1V2 PC2X1;

using namespace std;

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {

/// @addtogroup PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief Class elementary building block for hierarchial geometry description
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include
 *  @code
 *  #include "PSCalib/GeometryObject.h"
 *  typedef boost::shared_ptr<GeometryObject> shpGO;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  @code
 *    PSCalib::GeometryObject* geo = new PSCalib::GeometryObject(pname, 
 *  							       pindex,
 *  							       oname, 
 *  							       oindex,
 *  							       x0,    
 *  							       y0,    
 *  							       z0,    
 *  							       rot_z, 
 *  							       rot_y, 
 *  							       rot_x, 
 *  							       tilt_z,
 *  							       tilt_y,
 *  							       tilt_x );
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *    const double* X;
 *    const double* Y;
 *    const double* Z;
 *    unsigned   size;
 *    geo->get_pixel_coords(X, Y, Z, size);
 *
 *    shpGO parobj = geo->get_parent();
 *    std::vector<shpGO> lst = geo->get_list_of_children();
 *
 *    std::string oname  = geo->get_geo_name();
 *    unsigned    oindex = geo->get_geo_index();
 *    std::string pname  = geo->get_parent_name();
 *    unsigned    pindex = geo->get_parent_index();
 *  
 *    // Next methods are used in class GeometryAccess for building of hierarchial geometry structure.
 *    geo->set_parent(parent_geo);
 *    geo->add_child(child_geo);
 *  @endcode
 *  
 *  @li Print methods
 *  @code
 *    geo->print_geo();
 *    geo->print_geo_children();
 *    cout << "Size of geo: " << geo->get_size_geo_array(); 
 *  @endcode
 */

//-------------------

//typedef ndarray<double,1> NDA;
//typedef ndarray<const double,1> CNDA;

//-------------------

class GeometryObject  {
public:

  //typedef GeometryObject* shpGO;
  typedef boost::shared_ptr<GeometryObject> shpGO;

  enum ALGO_TYPE  {NONDEF, SENS2X1V1, SENS2X1V2};

  /**
   *  @brief Class constructor accepts path to the calibration "geometry" file and verbosity control bit-word 
   *  
   *  @param[in] pname  - parent name
   *  @param[in] pindex - parent index
   *  @param[in] oname  - this object name
   *  @param[in] oindex - this object index
   *  @param[in] x0     - object origin coordinate x[um] in parent frame
   *  @param[in] y0     - object origin coordinate y[um] in parent frame
   *  @param[in] z0     - object origin coordinate z[um] in parent frame
   *  @param[in] rot_z  - object rotation/design angle [deg] around axis z of the parent frame
   *  @param[in] rot_y  - object rotation/design angle [deg] around axis y of the parent frame
   *  @param[in] rot_x  - object rotation/design angle [deg] around axis x of the parent frame
   *  @param[in] tilt_z - object tilt/deviation angle [deg] around axis z of the parent frame
   *  @param[in] tilt_y - object tilt/deviation angle [deg] around axis y of the parent frame
   *  @param[in] tilt_x - object tilt/deviation angle [deg] around axis x of the parent frame
   */
  GeometryObject (  std::string pname  = std::string(),
                    unsigned    pindex = 0,
                    std::string oname  = std::string(),
                    unsigned    oindex = 0,
                    double      x0     = 0,
                    double      y0     = 0,
                    double      z0     = 0,
                    double      rot_z  = 0,
                    double      rot_y  = 0,
                    double      rot_x  = 0,                  
                    double      tilt_z = 0,
                    double      tilt_y = 0,
                    double      tilt_x = 0
                  ) ;

  // Destructor
  virtual ~GeometryObject () ;

  std::string string_geo();
  std::string string_geo_children();
  void print_geo();
  void print_geo_children();
  void set_parent(shpGO parent) { m_parent = parent; }
  void add_child (shpGO child) { v_list_of_children.push_back(child); }
  shpGO get_parent() { return m_parent; }
  std::vector<shpGO> get_list_of_children() { return v_list_of_children; }

  std::string get_geo_name()     { return m_oname; }
  unsigned    get_geo_index()    { return m_oindex; }
  std::string get_parent_name()  { return m_pname; }
  unsigned    get_parent_index() { return m_pindex; }
  /**
   *  @brief Returns pointers to pixel coordinate arrays
   *  @param[out] X - pointer to x pixel coordinate array
   *  @param[out] Y - pointer to y pixel coordinate array
   *  @param[out] Z - pointer to z pixel coordinate array
   *  @param[out] size - size of the pixel coordinate array (number of pixels)
   */
  void get_pixel_coords(const double*& X, const double*& Y, const double*& Z, unsigned& size);
  unsigned get_size_geo_array();

protected:

private:

  // Data members
  std::string m_pname;
  unsigned    m_pindex;

  std::string m_oname;
  unsigned    m_oindex;

  double      m_x0;
  double      m_y0;
  double      m_z0;

  double      m_rot_z;
  double      m_rot_y;
  double      m_rot_x;

  double      m_tilt_z;
  double      m_tilt_y;
  double      m_tilt_x;

  shpGO m_parent;
  std::vector<shpGO> v_list_of_children;

  ALGO_TYPE m_algo;
  PC2X1* m_pix_coords_2x1;

  unsigned m_size;
  double*  p_xarr;
  double*  p_yarr;
  double*  p_zarr;

  //  NDA  m_X;
  //  NDA  m_Y;
  //  NDA  m_Z;

  void transform_geo_coord_arrays( const double* X, 
                                   const double* Y,  
                                   const double* Z, 
                                   const unsigned size,
                                   double*  Xt,  
                                   double*  Yt,  
                                   double*  Zt,
                                   const bool do_tilt=true
                                  );
  void evaluate_pixel_coords();

  const static double DEG_TO_RAD = 3.141592653589793238463 / 180; 

  static void rotation(const double* X, const double* Y, const unsigned size,
                       const double C, const double S, 
		       double* Xrot, double* Yrot);

  static void rotation(const double* X, const double* Y, const unsigned size, const double angle_deg, 
                       double* Xrot, double* Yrot);

  /// Returns class name for MsgLogger
  static const std::string name() {return "PSCalib";}

  // Copy constructor and assignment are disabled by default
  GeometryObject ( const GeometryObject& ) ;
  GeometryObject& operator = ( const GeometryObject& ) ;
};

//-------------------

} // namespace PSCalib

#endif // PSCALIB_GEOMETRYOBJECT_H

//-------------------
//-------------------
//-------------------
