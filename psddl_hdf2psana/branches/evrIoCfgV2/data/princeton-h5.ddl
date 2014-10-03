@include "psddldata/princeton.ddl";
@package Princeton  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute width;
    @attribute height;
    @attribute orgX;
    @attribute orgY;
    @attribute binX;
    @attribute binY;
    @attribute exposureTime;
    @attribute coolingTemp;
    @attribute readoutSpeedIndex;
    @attribute readoutEventCode;
    @attribute delayMode;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute width;
    @attribute height;
    @attribute orgX;
    @attribute orgY;
    @attribute binX;
    @attribute binY;
    @attribute exposureTime;
    @attribute coolingTemp;
    @attribute gainIndex;
    @attribute readoutSpeedIndex;
    @attribute readoutEventCode;
    @attribute delayMode;
  }
}


//------------------ ConfigV3 ------------------
@h5schema ConfigV3
  [[version(0)]]
{
  @dataset config {
    @attribute width;
    @attribute height;
    @attribute orgX;
    @attribute orgY;
    @attribute binX;
    @attribute binY;
    @attribute exposureTime;
    @attribute coolingTemp;
    @attribute gainIndex;
    @attribute readoutSpeedIndex;
    @attribute exposureEventCode;
    @attribute numDelayShots;
  }
}


//------------------ ConfigV4 ------------------
@h5schema ConfigV4
  [[version(0)]]
{
  @dataset config {
    @attribute width;
    @attribute height;
    @attribute orgX;
    @attribute orgY;
    @attribute binX;
    @attribute binY;
    @attribute maskedHeight;
    @attribute kineticHeight;
    @attribute vsSpeed;
    @attribute exposureTime;
    @attribute coolingTemp;
    @attribute gainIndex;
    @attribute readoutSpeedIndex;
    @attribute exposureEventCode;
    @attribute numDelayShots;
  }
}


//------------------ ConfigV5 ------------------
@h5schema ConfigV5
  [[version(0)]]
{
  @dataset config {
    @attribute width;
    @attribute height;
    @attribute orgX;
    @attribute orgY;
    @attribute binX;
    @attribute binY;
    @attribute exposureTime;
    @attribute coolingTemp;
    @attribute gainIndex;
    @attribute readoutSpeedIndex;
    @attribute maskedHeight;
    @attribute kineticHeight;
    @attribute vsSpeed;
    @attribute infoReportInterval;
    @attribute exposureEventCode;
    @attribute numDelayShots;
  }
}


//------------------ FrameV1 ------------------
@h5schema FrameV1
  [[version(0)]]
{
  @dataset frame {
    @attribute shotIdStart;
    @attribute readoutTime;
  }
  @dataset data;
}


//------------------ FrameV2 ------------------
@h5schema FrameV2
  [[version(0)]]
{
  @dataset frame {
    @attribute shotIdStart;
    @attribute readoutTime;
    @attribute temperature;
  }
  @dataset data;
}
} //- @package Princeton
