@include "psddldata/pnccd.ddl";
@package PNCCD  {


//------------------ FrameV1 ------------------
@h5schema FrameV1
  [[version(0)]]
  [[external("psddl_hdf2psana/pnccd.h")]]
  [[embedded]]
{
}


//------------------ FullFrameV1 ------------------
@h5schema FullFrameV1
  [[version(0)]]
  [[external("psddl_hdf2psana/pnccd.h")]]
{
}


//------------------ FramesV1 ------------------
@h5schema FramesV1
  [[version(0)]]
  [[external("psddl_hdf2psana/pnccd.h")]]
{
}
} //- @package PNCCD
