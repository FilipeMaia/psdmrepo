@include "psddldata/timepix.ddl";
@package Timepix  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute readoutSpeed;
    @attribute triggerMode;
    @attribute shutterTimeout;
    @attribute dac0Ikrum;
    @attribute dac0Disc;
    @attribute dac0Preamp;
    @attribute dac0BufAnalogA;
    @attribute dac0BufAnalogB;
    @attribute dac0Hist;
    @attribute dac0ThlFine;
    @attribute dac0ThlCourse;
    @attribute dac0Vcas;
    @attribute dac0Fbk;
    @attribute dac0Gnd;
    @attribute dac0Ths;
    @attribute dac0BiasLvds;
    @attribute dac0RefLvds;
    @attribute dac1Ikrum;
    @attribute dac1Disc;
    @attribute dac1Preamp;
    @attribute dac1BufAnalogA;
    @attribute dac1BufAnalogB;
    @attribute dac1Hist;
    @attribute dac1ThlFine;
    @attribute dac1ThlCourse;
    @attribute dac1Vcas;
    @attribute dac1Fbk;
    @attribute dac1Gnd;
    @attribute dac1Ths;
    @attribute dac1BiasLvds;
    @attribute dac1RefLvds;
    @attribute dac2Ikrum;
    @attribute dac2Disc;
    @attribute dac2Preamp;
    @attribute dac2BufAnalogA;
    @attribute dac2BufAnalogB;
    @attribute dac2Hist;
    @attribute dac2ThlFine;
    @attribute dac2ThlCourse;
    @attribute dac2Vcas;
    @attribute dac2Fbk;
    @attribute dac2Gnd;
    @attribute dac2Ths;
    @attribute dac2BiasLvds;
    @attribute dac2RefLvds;
    @attribute dac3Ikrum;
    @attribute dac3Disc;
    @attribute dac3Preamp;
    @attribute dac3BufAnalogA;
    @attribute dac3BufAnalogB;
    @attribute dac3Hist;
    @attribute dac3ThlFine;
    @attribute dac3ThlCourse;
    @attribute dac3Vcas;
    @attribute dac3Fbk;
    @attribute dac3Gnd;
    @attribute dac3Ths;
    @attribute dac3BiasLvds;
    @attribute dac3RefLvds;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute readoutSpeed;
    @attribute triggerMode;
    @attribute timepixSpeed;
    @attribute dac0Ikrum;
    @attribute dac0Disc;
    @attribute dac0Preamp;
    @attribute dac0BufAnalogA;
    @attribute dac0BufAnalogB;
    @attribute dac0Hist;
    @attribute dac0ThlFine;
    @attribute dac0ThlCourse;
    @attribute dac0Vcas;
    @attribute dac0Fbk;
    @attribute dac0Gnd;
    @attribute dac0Ths;
    @attribute dac0BiasLvds;
    @attribute dac0RefLvds;
    @attribute dac1Ikrum;
    @attribute dac1Disc;
    @attribute dac1Preamp;
    @attribute dac1BufAnalogA;
    @attribute dac1BufAnalogB;
    @attribute dac1Hist;
    @attribute dac1ThlFine;
    @attribute dac1ThlCourse;
    @attribute dac1Vcas;
    @attribute dac1Fbk;
    @attribute dac1Gnd;
    @attribute dac1Ths;
    @attribute dac1BiasLvds;
    @attribute dac1RefLvds;
    @attribute dac2Ikrum;
    @attribute dac2Disc;
    @attribute dac2Preamp;
    @attribute dac2BufAnalogA;
    @attribute dac2BufAnalogB;
    @attribute dac2Hist;
    @attribute dac2ThlFine;
    @attribute dac2ThlCourse;
    @attribute dac2Vcas;
    @attribute dac2Fbk;
    @attribute dac2Gnd;
    @attribute dac2Ths;
    @attribute dac2BiasLvds;
    @attribute dac2RefLvds;
    @attribute dac3Ikrum;
    @attribute dac3Disc;
    @attribute dac3Preamp;
    @attribute dac3BufAnalogA;
    @attribute dac3BufAnalogB;
    @attribute dac3Hist;
    @attribute dac3ThlFine;
    @attribute dac3ThlCourse;
    @attribute dac3Vcas;
    @attribute dac3Fbk;
    @attribute dac3Gnd;
    @attribute dac3Ths;
    @attribute dac3BiasLvds;
    @attribute dac3RefLvds;
    @attribute driverVersion;
    @attribute firmwareVersion;
    @attribute pixelThreshSize;
    @attribute pixelThresh;
    @attribute chip0Name [[vlen]];
    @attribute chip1Name [[vlen]];
    @attribute chip2Name [[vlen]];
    @attribute chip3Name [[vlen]];
    @attribute chip0ID;
    @attribute chip1ID;
    @attribute chip2ID;
    @attribute chip3ID;
    @attribute chipCount;
  }
}


//------------------ ConfigV3 ------------------
@h5schema ConfigV3
  [[version(0)]]
{
  @dataset config {
    @attribute readoutSpeed;
    @attribute timepixMode;
    @attribute timepixSpeed;
    @attribute dac0Ikrum;
    @attribute dac0Disc;
    @attribute dac0Preamp;
    @attribute dac0BufAnalogA;
    @attribute dac0BufAnalogB;
    @attribute dac0Hist;
    @attribute dac0ThlFine;
    @attribute dac0ThlCourse;
    @attribute dac0Vcas;
    @attribute dac0Fbk;
    @attribute dac0Gnd;
    @attribute dac0Ths;
    @attribute dac0BiasLvds;
    @attribute dac0RefLvds;
    @attribute dac1Ikrum;
    @attribute dac1Disc;
    @attribute dac1Preamp;
    @attribute dac1BufAnalogA;
    @attribute dac1BufAnalogB;
    @attribute dac1Hist;
    @attribute dac1ThlFine;
    @attribute dac1ThlCourse;
    @attribute dac1Vcas;
    @attribute dac1Fbk;
    @attribute dac1Gnd;
    @attribute dac1Ths;
    @attribute dac1BiasLvds;
    @attribute dac1RefLvds;
    @attribute dac2Ikrum;
    @attribute dac2Disc;
    @attribute dac2Preamp;
    @attribute dac2BufAnalogA;
    @attribute dac2BufAnalogB;
    @attribute dac2Hist;
    @attribute dac2ThlFine;
    @attribute dac2ThlCourse;
    @attribute dac2Vcas;
    @attribute dac2Fbk;
    @attribute dac2Gnd;
    @attribute dac2Ths;
    @attribute dac2BiasLvds;
    @attribute dac2RefLvds;
    @attribute dac3Ikrum;
    @attribute dac3Disc;
    @attribute dac3Preamp;
    @attribute dac3BufAnalogA;
    @attribute dac3BufAnalogB;
    @attribute dac3Hist;
    @attribute dac3ThlFine;
    @attribute dac3ThlCourse;
    @attribute dac3Vcas;
    @attribute dac3Fbk;
    @attribute dac3Gnd;
    @attribute dac3Ths;
    @attribute dac3BiasLvds;
    @attribute dac3RefLvds;
    @attribute dacBias;
    @attribute flags;
    @attribute driverVersion;
    @attribute firmwareVersion;
    @attribute pixelThreshSize;
    @attribute pixelThresh;
    @attribute chip0Name [[vlen]];
    @attribute chip1Name [[vlen]];
    @attribute chip2Name [[vlen]];
    @attribute chip3Name [[vlen]];
    @attribute chip0ID;
    @attribute chip1ID;
    @attribute chip2ID;
    @attribute chip3ID;
    @attribute chipCount;
  }
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
{
  @dataset data {
    @attribute timestamp;
    @attribute frameCounter;
    @attribute lostRows;
  }
  @dataset image [[method(data)]];
}


//------------------ DataV2 ------------------
@h5schema DataV2
  [[version(0)]]
{
  @dataset data {
    @attribute width;
    @attribute height;
    @attribute timestamp;
    @attribute frameCounter;
    @attribute lostRows;
  }
  @dataset image [[method(data)]];
}
} //- @package Timepix
