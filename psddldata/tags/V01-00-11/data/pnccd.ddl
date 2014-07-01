@package PNCCD  {


//------------------ ConfigV1 ------------------
/* pnCCD configuration class ConfigV1 */
@type ConfigV1
  [[type_id(Id_pnCCDconfig, 1)]]
  [[config_type]]
{
  uint32_t _numLinks -> numLinks;	/* Number of links in the pnCCD. */
  uint32_t _payloadSizePerLink -> payloadSizePerLink;	/* Size of the payload in bytes for single link */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
/* pnCCD configuration class ConfigV2 */
@type ConfigV2
  [[type_id(Id_pnCCDconfig, 2)]]
  [[config_type]]
{
  uint32_t _numLinks -> numLinks;	/* Number of links in the pnCCD. */
  uint32_t _payloadSizePerLink -> payloadSizePerLink;	/* Size of the payload in bytes for single link */
  uint32_t _numChannels -> numChannels;	/* Number of channels */
  uint32_t _numRows -> numRows;	/* Number of rows */
  uint32_t _numSubmoduleChannels -> numSubmoduleChannels;	/* Number of submodule channels */
  uint32_t _numSubmoduleRows -> numSubmoduleRows;	/* Number of submodule rows */
  uint32_t _numSubmodules -> numSubmodules;	/* Number of submodules */
  uint32_t _camexMagic -> camexMagic;	/* Magic word from CAMEX */
  char _info[256] -> info;	/* Information data string */
  char _timingFName[256] -> timingFName;	/* Timing file name string */

  /* Construct from values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ FrameV1 ------------------
/* pnCCD class FrameV1, this is a class which is defined by origianl pdsdata package. */
@type FrameV1
  [[config(ConfigV1, ConfigV2)]]
{
  uint32_t _specialWord -> specialWord;	/* Special values */
  uint32_t _frameNumber -> frameNumber;	/* Frame number */
  uint32_t _timeStampHi -> timeStampHi;	/* Most significant part of timestamp */
  uint32_t _timeStampLo -> timeStampLo;	/* Least significant part of timestamp */
  uint16_t __data[(@config.payloadSizePerLink()-16)/2] -> _data;	/* Frame data */

  uint16_t[][] data()  [[inline]]
  [[language("C++")]] @{ return make_ndarray(_data(@config).data(), 512, 512); @}
}


//------------------ FullFrameV1 ------------------
/* This is a "synthetic" pnCCD frame which is four original 512x512 frames
            glued together. This class does not exist in original pdsdata, it has been 
            introduced to psana to simplify access to full frame data in the user code. */
@type FullFrameV1
  [[type_id(Id_pnCCDframe, 1)]]
{
  uint32_t _specialWord -> specialWord;	/* Special values */
  uint32_t _frameNumber -> frameNumber;	/* Frame number */
  uint32_t _timeStampHi -> timeStampHi;	/* Most significant part of timestamp */
  uint32_t _timeStampLo -> timeStampLo;	/* Least significant part of timestamp */
  uint16_t _data[1024][1024] -> data;	/* Full frame data, image size is 1024x1024. */
}


//------------------ FramesV1 ------------------
/* pnCCD class FramesV1 which is a collection of FrameV1 objects, number of 
            frames in collection is determined by numLinks() method (which should return 4 
            in most cases). This class does not exist in original pdsdata, has been 
            introduced to psana to help in organizing 4 small pnCCD frames together. */
@type FramesV1
  [[type_id(Id_pnCCDframe, 1)]]
  [[config(ConfigV1, ConfigV2)]]
{
  FrameV1 _frames[@config.numLinks()] -> frame  [[shape_method(frame_shape)]];	/* Number of frames is determined by numLinks() method. */

  uint32_t numLinks()  [[inline]]
  [[language("C++")]] @{ return @config.numLinks(); @}
}
} //- @package PNCCD
