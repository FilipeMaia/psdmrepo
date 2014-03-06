@package Ipimb  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_IpimbConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @enum CapacitorValue (uint8_t) {
    c_1pF,
    c_100pF,
    c_10nF,
  }

  uint64_t _triggerCounter -> triggerCounter;
  uint64_t _serialID -> serialID;
  uint16_t _chargeAmpRange -> chargeAmpRange;
  uint16_t _calibrationRange -> calibrationRange;
  uint32_t _resetLength -> resetLength;
  uint32_t _resetDelay -> resetDelay;
  float _chargeAmpRefVoltage -> chargeAmpRefVoltage;
  float _calibrationVoltage -> calibrationVoltage;
  float _diodeBias -> diodeBias;
  uint16_t _status -> status;
  uint16_t _errors -> errors;
  uint16_t _calStrobeLength -> calStrobeLength;
  uint16_t _pad0;
  uint32_t _trigDelay -> trigDelay;

  /* Returns CapacitorValue enum for given channel number (0..3). */
  CapacitorValue capacitorValue(uint32_t ch)  [[inline]]
  [[language("C++")]] @{ return CapacitorValue((@self.chargeAmpRange() >> (ch*2)) & 0x3); @}

  /* Returns array of CapacitorValue enums. */
  CapacitorValue[] capacitorValues()
  [[language("C++")]] @{
    /* return type is actually ndarray<uint8_t, 1> (at least for now) */
    ndarray<uint8_t, 1> cap = make_ndarray<uint8_t>(4);
    for (int ch = 0; ch != 4; ++ ch) {
      cap[ch] = uint8_t((@self.chargeAmpRange() >> (ch*2)) & 0x3);
    }
    return cap;
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_IpimbConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  @enum CapacitorValue (uint8_t) {
    c_1pF,
    c_4p7pF,
    c_24pF,
    c_120pF,
    c_620pF,
    c_3p3nF,
    c_10nF,
    expert,
  }

  uint64_t _triggerCounter -> triggerCounter;
  uint64_t _serialID -> serialID;
  uint16_t _chargeAmpRange -> chargeAmpRange;
  uint16_t _calibrationRange -> calibrationRange;
  uint32_t _resetLength -> resetLength;
  uint32_t _resetDelay -> resetDelay;
  float _chargeAmpRefVoltage -> chargeAmpRefVoltage;
  float _calibrationVoltage -> calibrationVoltage;
  float _diodeBias -> diodeBias;
  uint16_t _status -> status;
  uint16_t _errors -> errors;
  uint16_t _calStrobeLength -> calStrobeLength;
  uint16_t _pad0;
  uint32_t _trigDelay -> trigDelay;
  uint32_t _trigPsDelay -> trigPsDelay;
  uint32_t _adcDelay -> adcDelay;

  /* Returns CapacitorValue enum for given channel number (0..3). */
  CapacitorValue capacitorValue(uint32_t ch)  [[inline]]
  [[language("C++")]] @{ return CapacitorValue((@self.chargeAmpRange() >> (ch*4)) & 0xf); @}

  /* Returns array of CapacitorValue enums. */
  CapacitorValue[] capacitorValues()
  [[language("C++")]] @{
    /* return type is actually ndarray<uint8_t, 1> (at least for now) */
    ndarray<uint8_t, 1> cap = make_ndarray<uint8_t>(4);
    for (int ch = 0; ch != 4; ++ ch) {
      cap[ch] = uint8_t((@self.chargeAmpRange() >> (ch*4)) & 0xf);
    }
    return cap;
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_IpimbData, 1)]]
  [[pack(4)]]
{
  uint64_t _triggerCounter -> triggerCounter;
  uint16_t _config0 -> config0;
  uint16_t _config1 -> config1;
  uint16_t _config2 -> config2;
  uint16_t _channel0 -> channel0;	/* Raw counts value returned from channel 0. */
  uint16_t _channel1 -> channel1;	/* Raw counts value returned from channel 1. */
  uint16_t _channel2 -> channel2;	/* Raw counts value returned from channel 2. */
  uint16_t _channel3 -> channel3;	/* Raw counts value returned from channel 3. */
  uint16_t _checksum -> checksum;

  /* Value of of channel0() converted to Volts. */
  float channel0Volts()
  [[language("C++")]] @{ return float(@self._channel0)*3.3/65535; @}

  /* Value of of channel1() converted to Volts. */
  float channel1Volts()
  [[language("C++")]] @{ return float(@self._channel1)*3.3/65535; @}

  /* Value of of channel2() converted to Volts. */
  float channel2Volts()
  [[language("C++")]] @{ return float(@self._channel2)*3.3/65535; @}

  /* Value of of channel3() converted to Volts. */
  float channel3Volts()
  [[language("C++")]] @{ return float(@self._channel3)*3.3/65535; @}
}


//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_IpimbData, 2)]]
  [[pack(4)]]
{
  @const int32_t ipimbAdcRange = 5;
  @const int32_t ipimbAdcSteps = 65536;

  uint64_t _triggerCounter;
  uint16_t _config0 -> config0;
  uint16_t _config1 -> config1;
  uint16_t _config2 -> config2;
  uint16_t _channel0 -> channel0;	/* Raw counts value returned from channel 0. */
  uint16_t _channel1 -> channel1;	/* Raw counts value returned from channel 1. */
  uint16_t _channel2 -> channel2;	/* Raw counts value returned from channel 2. */
  uint16_t _channel3 -> channel3;	/* Raw counts value returned from channel 3. */
  uint16_t _channel0ps -> channel0ps;	/* Raw counts value returned from channel 0. */
  uint16_t _channel1ps -> channel1ps;	/* Raw counts value returned from channel 1. */
  uint16_t _channel2ps -> channel2ps;	/* Raw counts value returned from channel 2. */
  uint16_t _channel3ps -> channel3ps;	/* Raw counts value returned from channel 3. */
  uint16_t _checksum -> checksum;

  /* Value of of channel0() converted to Volts. */
  float channel0Volts()
  [[language("C++")]] @{ return float(@self._channel0)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel1() converted to Volts. */
  float channel1Volts()
  [[language("C++")]] @{ return float(@self._channel1)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel2() converted to Volts. */
  float channel2Volts()
  [[language("C++")]] @{ return float(@self._channel2)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel3() converted to Volts. */
  float channel3Volts()
  [[language("C++")]] @{ return float(@self._channel3)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel0ps() converted to Volts. */
  float channel0psVolts()
  [[language("C++")]] @{ return float(@self._channel0ps)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel1ps() converted to Volts. */
  float channel1psVolts()
  [[language("C++")]] @{ return float(@self._channel1ps)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel2ps() converted to Volts. */
  float channel2psVolts()
  [[language("C++")]] @{ return float(@self._channel2ps)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Value of of channel3ps() converted to Volts. */
  float channel3psVolts()
  [[language("C++")]] @{ return float(@self._channel3ps)*ipimbAdcRange/(ipimbAdcSteps - 1); @}

  /* Trigger counter value. */
  uint64_t triggerCounter()  [[inline]]
  [[language("C++")]] @{ 
    return (((_triggerCounter >> 48) & 0x000000000000ffffLL) |
	((_triggerCounter >> 16) & 0x00000000ffff0000LL) |
	((_triggerCounter << 16) & 0x0000ffff00000000LL) |
	((_triggerCounter << 48) & 0xffff000000000000LL)); 
  @}
}
} //- @package Ipimb
