#ifndef O2OTRANSLATOR_O2OHDF5WRITER_H
#define O2OTRANSLATOR_O2OHDF5WRITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OHdf5Writer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <memory>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/O2OXtcScannerI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "H5DataTypes/ObjectContainer.h"
#include "H5DataTypes/XtcClockTime.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Scanner file that sends all data to HDF5 file
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace O2OTranslator {

class O2OFileNameFactory ;

class O2OHdf5Writer : public O2OXtcScannerI {
public:

  enum SplitMode { NoSplit, Family } ;

  // current XTC state
  enum State { Undefined, Mapped, Configured, Running } ;

  // Default constructor
  O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                  bool overwrite=false,
                  SplitMode split=NoSplit,
                  hsize_t splitSize=0xFFFFFFFF ) ;

  // Destructor
  virtual ~O2OHdf5Writer () ;

  // signal start/end of the event (datagram)
  virtual void eventStart ( const Pds::Sequence& seq ) ;
  virtual void eventEnd ( const Pds::Sequence& seq ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::Acqiris::DataDescV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::Camera::FrameFexConfigV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::Camera::FrameV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::Camera::TwoDGaussianV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::EvrData::ConfigV1& data, const Pds::DetInfo& detInfo ) ;
  virtual void dataObject ( const Pds::Opal1k::ConfigV1& data, const Pds::DetInfo& detInfo ) ;

protected:

private:

  struct CmpDetInfo {
    bool operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const ;
  };

  typedef std::map<Pds::DetInfo,Pds::Acqiris::ConfigV1, CmpDetInfo> AcqConfigMap ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  hdf5pp::File m_file ;
  State m_state ;
  hdf5pp::Group m_mapGroup ;      // Group for current Map transition
  hdf5pp::Group m_configGroup ;   // Group for Configure transition
  hdf5pp::Group m_eventGroup ;    // Group for event data
  H5DataTypes::XtcClockTime m_eventTime ;

  /// typedefs for various container types
  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTime> XtcClockTimeCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::CameraTwoDGaussianV1> CameraTwoDGaussianV1Cont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::CameraFrameV1> CameraFrameV1Cont ;
  typedef H5DataTypes::ObjectContainer<const unsigned char> CameraFrameV1ImageCont ;

  std::auto_ptr<CameraTwoDGaussianV1Cont> m_cameraTwoDGaussianV1Cont ;
  std::auto_ptr<XtcClockTimeCont> m_cameraTwoDGaussianV1TimeCont ;

  std::auto_ptr<CameraFrameV1Cont> m_cameraFrameV1Cont ;
  std::auto_ptr<CameraFrameV1ImageCont> m_cameraFrameV1ImageCont ;
  std::auto_ptr<XtcClockTimeCont> m_cameraFrameV1TimeCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OHDF5WRITER_H
