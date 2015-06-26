@include "psddldata/usdusb.ddl";
@package UsdUsb  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute counting_mode;
    @attribute quadrature_mode;
  }
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
  [[external("psddl_hdf2psana/usdusb.h")]]
{
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(1)]]
{
  @dataset data {
    @attribute digital_in;
    @attribute timestamp;
    @attribute status;
    @attribute analog_in;
    @attribute encoder_count[Encoder_Inputs];
  }
}
} //- @package UsdUsb
