//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpOpal1k...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpOpal1k.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/opal1k.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpOpal1k)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpOpal1k"; 
  
  void
  printFrameCoord(std::ostream& str, const Psana::Camera::FrameCoord& coord) 
  {
    str << "(" << coord.column() << ", " << coord.row() << ")";
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpOpal1k::DumpOpal1k (const std::string& name)
  : Module(name)
{
  m_src = configStr("opalSource", "DetInfo(:Opal1000)");
}

//--------------
// Destructor --
//--------------
DumpOpal1k::~DumpOpal1k ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpOpal1k::beginCalibCycle(Env& env)
{
  MsgLog(logger, info, name() << ": in beginCalibCycle()");

  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_src);
  if (config.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Psana::Opal1k::ConfigV1:";
      str << "\n  black_level = " << config->black_level();
      str << "\n  gain_percent = " << config->gain_percent();
      str << "\n  output_resolution = " << config->output_resolution();
      str << "\n  vertical_binning = " << config->vertical_binning();
      str << "\n  output_mirroring = " << config->output_mirroring();
      str << "\n  vertical_remapping = " << int(config->vertical_remapping());
      str << "\n  output_offset = " << config->output_offset();
      str << "\n  output_resolution_bits = " << config->output_resolution_bits();
      str << "\n  defect_pixel_correction_enabled = " << int(config->defect_pixel_correction_enabled());
      str << "\n  output_lookup_table_enabled = " << int(config->output_lookup_table_enabled());
      
      if (config->output_lookup_table_enabled()) {        
        const uint16_t* output_lookup_table = config->output_lookup_table();
        str << "\n  output_lookup_table =";
        for (int i = 0; i < Psana::Opal1k::ConfigV1::Output_LUT_Size; ++ i) {
          str << ' ' << output_lookup_table[i];
        }
        
      }
      
      
      if (config->number_of_defect_pixels()) {
        str << "\n  defect_pixel_coordinates =";
        for (unsigned i = 0; i < config->number_of_defect_pixels(); ++ i) {
          const Psana::Camera::FrameCoord& coord = config->defect_pixel_coordinates(i);
          str << " ";
          printFrameCoord(str, coord);
        }
      }
    }
    
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpOpal1k::event(Event& evt, Env& env)
{
}
  
} // namespace psana_examples
