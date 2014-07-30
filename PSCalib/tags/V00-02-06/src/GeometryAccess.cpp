//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GeometryAccess...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/GeometryAccess.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream> // for cout
#include <fstream>  // for ifstream 
#include <sstream>  // for stringstream
#include <iomanip>  // for setw, setfill

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSCalib {

//----------------
// Constructors --
//----------------

GeometryAccess::GeometryAccess (const std::string& path, unsigned pbits)
  : m_path(path)
  , m_pbits(pbits)
{
  load_pars_from_file();

  if(m_pbits & 2) print_list_of_geos();
  if(m_pbits & 4) print_list_of_geos_children();
  if(m_pbits & 8) print_comments_from_dict();
}

//--------------
// Destructor --
//--------------

GeometryAccess::~GeometryAccess ()
{
}

//-------------------
void GeometryAccess::load_pars_from_file()
{
  m_dict_of_comments.clear();
  v_list_of_geos.clear();

  std::ifstream f(m_path.c_str());

  if(not f.good()) { MsgLog(name(), error, "Calibration file " << m_path << " does not exist"); }
  else             { if(m_pbits & 1) MsgLog(name(), info, "load_pars_from_file(): " << m_path); }

  std::string line;
  while (std::getline(f, line)) {    
    if(line.empty()) continue;    // discard empty lines
    if(line[0] == '#') {          // process line of comments 
       add_comment_to_dict(line); 
       continue;
    }
    // make geometry object and add it in the list
    v_list_of_geos.push_back( parse_line(line) );    
  }

  f.close();

  set_relations();
}

//-------------------

void GeometryAccess::add_comment_to_dict(const std::string& line)
{ 
  std::size_t p1 = line.find_first_not_of("# ");
  std::size_t p2 = line.find_first_of(" ", p1);
  std::size_t p3 = line.find_first_not_of(" ", p2);
  std::string beginline(line, p1, p2-p1);
  std::string endline(line, p3);
  //if (p1 == std::string::npos) ...
  //std::cout << "comment: " << line << '\n'; 
  //std::cout << "  p1:" << p1  << "  p2:" << p2 << "  p3:" << p3 << '\n';
  //std::cout << "   split line: [" << beginline << "] = " << endline << '\n'; 
  m_dict_of_comments[beginline] = endline;
}

//-------------------

GeometryAccess::shpGO GeometryAccess::parse_line(const std::string& line)
{
  std::string pname;
  unsigned    pindex;
  std::string oname;
  unsigned    oindex;
  double      x0;
  double      y0;
  double      z0;
  double      rot_z;
  double      rot_y;
  double      rot_x;                  
  double      tilt_z;
  double      tilt_y;
  double      tilt_x; 

  std::stringstream ss(line);

  if(ss >> pname >> pindex >> oname >> oindex >> x0 >> y0 >> z0 
        >> rot_z >> rot_y >> rot_x >> tilt_z >> tilt_y >> tilt_x) {
      GeometryAccess::shpGO shp( new GeometryObject::GeometryObject (pname,
                              		     pindex,
                              		     oname,
                              		     oindex,
                              		     x0,
                              		     y0,
                              	  	     z0,
                              		     rot_z,
                              		     rot_y,
                              		     rot_x,                  
                              		     tilt_z,
                              		     tilt_y,
                              		     tilt_x 
		                             ));
      return shp;
  }
  else {
      std::string msg = "parse_line(...) can't parse line: " + line;
      //std::cout << msg;
      MsgLog(name(), info, msg);
      return GeometryAccess::shpGO();
  }
}

//-------------------

GeometryAccess::shpGO GeometryAccess::find_parent(const GeometryAccess::shpGO& geobj)
{
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    if(*it == geobj) continue; // skip geobj themself
    if(   (*it)->get_geo_index() == geobj->get_parent_index()
       && (*it)->get_geo_name()  == geobj->get_parent_name() ) {
      return (*it);
    }
  }

  //The name of parent object is not found among geos in the v_list_of_geos
  // add top parent object to the list

  if( ! geobj->get_parent_name().empty() ) { // skip top parent itself
    GeometryAccess::shpGO shp_top_parent( new GeometryObject::GeometryObject (std::string(),
                            		                      0,
                            		                      geobj->get_parent_name(),
                            		                      geobj->get_parent_index()));
    v_list_of_geos.push_back( shp_top_parent );
    return shp_top_parent;		  
  }

  return GeometryAccess::shpGO(); // for top parent itself
}

//-------------------

void GeometryAccess::set_relations()
{
  std::stringstream ss; ss << "set_relations():";
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {

    GeometryAccess::shpGO shp_parent = find_parent(*it);
    //std::cout << "set_relations(): found parent name:" << shp_parent->get_parent_name()<<'\n';

    if( shp_parent == GeometryAccess::shpGO() ) continue; // skip parent of the top object
    
    (*it)->set_parent(shp_parent);
    shp_parent->add_child(*it);

    if(m_pbits & 16) 
      ss << "\n  geo:"     << std::setw(10) << (*it) -> get_geo_name()
         << " : "                           << (*it) -> get_geo_index()
         << " has parent:" << std::setw(10) << shp_parent -> get_geo_name()
         << " : "                           << shp_parent -> get_geo_index();
  }
  if(m_pbits & 16) MsgLog(name(), info, ss.str());
}

//-------------------

GeometryAccess::shpGO GeometryAccess::get_geo(const std::string& oname, const unsigned& oindex)
{
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    if(   (*it)->get_geo_index() == oindex
       && (*it)->get_geo_name()  == oname ) 
          return (*it);
  }
  return GeometryAccess::shpGO(); // None
}

//-------------------

GeometryAccess::shpGO GeometryAccess::get_top_geo()
{
  return v_list_of_geos.back();
}

//-------------------

void
GeometryAccess::get_pixel_coords( const double*& X, 
                                  const double*& Y, 
                                  const double*& Z, 
				  unsigned& size,
                                  const std::string& oname, 
                                  const unsigned& oindex)
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  if(m_pbits & 32) {
    std::string msg = "get_pixel_coords(...) for geo:\n" + geo -> string_geo_children();
    MsgLog(name(), info, msg);
  }
  geo -> get_pixel_coords(X, Y, Z, size);
}

//-------------------

void GeometryAccess::print_list_of_geos()
{
  std::stringstream ss; ss << "print_list_of_geos():";
  if( v_list_of_geos.empty() ) ss << "List of geos is empty...";
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    ss << '\n' << (*it)->string_geo();
  }
  //std::cout << ss.str();
  MsgLog(name(), info, ss.str());
}

//-------------------

void GeometryAccess::print_list_of_geos_children()
{
  std::stringstream ss; ss << "print_list_of_geos_children(): ";
  if( v_list_of_geos.empty() ) ss << "List of geos is empty...";

  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    ss << '\n' << (*it)->string_geo_children();
  }
  //std::cout << ss.str() << '\n';
  MsgLog(name(), info, ss.str());
}

//-------------------

void GeometryAccess::print_comments_from_dict()
{ 
  std::stringstream ss; ss << "print_comments_from_dict():\n"; 

  std::map<std::string, std::string>::iterator iter;

  for (iter = m_dict_of_comments.begin(); iter != m_dict_of_comments.end(); ++iter) {
    ss << "  key: " << std::setw(10) << std::left << iter->first;
    ss << "  val: " << iter->second << '\n';
  }
  //std::cout << ss.str();
  MsgLog(name(), info, ss.str());
}
//-------------------

void
GeometryAccess::print_pixel_coords( const std::string& oname, 
                                    const unsigned& oindex)
{
  const double* X;
  const double* Y;
  const double* Z;
  unsigned   size;
  get_pixel_coords(X,Y,Z,size,oname,oindex);

  std::stringstream ss; ss << "print_pixel_coords():\n"
			   << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  ss << "X: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << X[i] << ", "; ss << "...\n";
  ss << "Y: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Y[i] << ", "; ss << "...\n"; 
  ss << "Z: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Z[i] << ", "; ss << "...\n"; 
  //cout << ss.str();
  MsgLog(name(), info, ss.str());
}

//-------------------
//-------------------
//-------------------

} // namespace PSCalib

//-------------------
//-------------------
//-------------------
