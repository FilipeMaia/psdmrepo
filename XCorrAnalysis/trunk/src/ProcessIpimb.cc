//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessIpimb...
//
// Author List:
//      Ingrid Ofte
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/ProcessIpimb.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/ipimb.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(ProcessIpimb)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
ProcessIpimb::ProcessIpimb (const std::string& name)
  : Module(name)
  , m_ipmSrc()
  , m_ipmOffset()
  , m_NormalizationFlag()
  , m_IpimbLowerThreshold()
  , m_IpimbUpperThreshold()
{
  // get the values from configuration or use defaults
  m_ipmSrc = configStr("ipimb_source", "DetInfo(:Ipimb)");
  m_ipmI0 = config("I0_channel",99);// which channel to use as I0
  m_ipmOffset = config("offset",0.0); //1.22
  m_NormalizationFlag = config("NormalizationFlag",0);
  m_IpimbLowerThreshold = config("IpimbLowerThreshold",0.18);
  m_IpimbUpperThreshold = config("IpimbUpperThreshold", 2.0);
  m_OutLabel = configStr("output_label", name );
}

//--------------
// Destructor --
//--------------
ProcessIpimb::~ProcessIpimb ()
{
}

/// Method which is called once at the beginning of the job
void 
ProcessIpimb::beginJob(Event& evt, Env& env)
{
  m_count = 0;

  std::string fFolder = configStr("ofile_folder","temp_ppd/");
  std::string fPrefix = configStr("ofile_prefix","Ipimb");
  std::string fSuffix = configStr("ofile_suffix",".txt");
  sprintf(m_filename,"%s%s%s",fFolder.data(),fPrefix.data(),fSuffix.data());
  

}

/// Method which is called at the beginning of the run
void 
ProcessIpimb::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ProcessIpimb::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ProcessIpimb::event(Event& evt, Env& env)
{
  // Retreive the Ipimb information for normalization

  // Get IpimbBeamline Data 
  float IpimbVolts[4];

  shared_ptr<Psana::Ipimb::DataV2> myIpimb = evt.get(m_ipmSrc);
  if (myIpimb.get()) {
    
    WithMsgLog(name(), debug, str) {
      str << "Ipimb::DataV2:"
	  << "\n  triggerCounter = " << myIpimb->triggerCounter()
	  << "\n  config = " << myIpimb->config0()
	  << "," << myIpimb->config1()
	  << "," << myIpimb->config2()
	  << "\n  channel = " << myIpimb->channel0()
	  << "," << myIpimb->channel1()
	  << "," << myIpimb->channel2()
	  << "," << myIpimb->channel3()
	  << "\n  volts = " << myIpimb->channel0Volts()
	  << "," << myIpimb->channel1Volts()
	  << "," << myIpimb->channel2Volts()
	  << "," << myIpimb->channel3Volts()
	  << "\n  channel-ps = " << myIpimb->channel0ps()
	  << "," << myIpimb->channel1ps()
	  << "," << myIpimb->channel2ps()
	  << "," << myIpimb->channel3ps()
	  << "\n  volts-ps = " << myIpimb->channel0psVolts()
	  << "," << myIpimb->channel1psVolts()
	  << "," << myIpimb->channel2psVolts()
	  << "," << myIpimb->channel3psVolts()
	  << "\n  checksum = " << myIpimb->checksum();
    }

    IpimbVolts[0] = myIpimb->channel0Volts();
    IpimbVolts[1] = myIpimb->channel1Volts();
    IpimbVolts[2] = myIpimb->channel2Volts();
    IpimbVolts[3] = myIpimb->channel3Volts();

    m_IpimbArray[0][m_count]=IpimbVolts[0]-m_ipmOffset;
    m_IpimbArray[1][m_count]=IpimbVolts[1]-m_ipmOffset; 
    m_IpimbArray[2][m_count]=IpimbVolts[2]-m_ipmOffset;
    m_IpimbArray[3][m_count]=IpimbVolts[3]-m_ipmOffset;        

    // This channel is the I0 Foil
    if (m_ipmI0 < 4){
      double i0 = IpimbVolts[m_ipmI0]-m_ipmOffset;

      // put it into the event for later
      shared_ptr<double> i0_ptr(new double(i0));
      char i0_str[256];
      sprintf(i0_str,"%s:I0", m_OutLabel.data());
      evt.put(i0_ptr,i0_str);
	      
	if(m_NormalizationFlag == 1 && 
	   (i0 < m_IpimbLowerThreshold || i0 > m_IpimbUpperThreshold))
	  {  
	  // printf("Ipimb below Threshold:  Ipimb = %f\n",m_IpimbArray[1][EventIndex]);
	  
	}    
    }
  } else {
    //printf("No IPIMB found at address %s \n", m_ipmSrc);
  }

  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ProcessIpimb::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ProcessIpimb::endRun(Event& evt, Env& env)
{
  FILE* file = fopen(m_filename, "w" );
  for (int k=0;k<m_count;k++)
    {
      fprintf(file, "%f %f %f %f\n", m_IpimbArray[0][k],m_IpimbArray[1][k],m_IpimbArray[2][k],m_IpimbArray[3][k] );
      
    }
  fclose(file);
}

/// Method which is called once at the end of the job
void 
ProcessIpimb::endJob(Event& evt, Env& env)
{
} 
} // namespace XCorrAnalysis
  
