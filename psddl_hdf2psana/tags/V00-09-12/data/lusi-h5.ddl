@include "psddldata/lusi.ddl";
@package Lusi  {


//------------------ DiodeFexConfigV1 ------------------
@h5schema DiodeFexConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute base;
    @attribute scale;
  }
}


//------------------ DiodeFexConfigV2 ------------------
@h5schema DiodeFexConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute base;
    @attribute scale;
  }
}


//------------------ IpmFexConfigV1 ------------------
@h5schema IpmFexConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute diode;
    @attribute xscale;
    @attribute yscale;
  }
}


//------------------ IpmFexConfigV2 ------------------
@h5schema IpmFexConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute diode;
    @attribute xscale;
    @attribute yscale;
  }
}


//------------------ IpmFexV1 ------------------
@h5schema IpmFexV1
  [[version(0)]]
{
  @dataset data {
    @attribute channel;
    @attribute sum;
    @attribute xpos;
    @attribute ypos;
  }
}
} //- @package Lusi
