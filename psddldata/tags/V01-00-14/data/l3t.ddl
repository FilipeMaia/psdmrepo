@package L3T  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_L3TConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _module_id_len -> module_id_len;	/* Length of the module identification string */
  uint32_t _desc_len -> desc_len;	/* Length of the description string */
  char _module_id[@self.module_id_len()] -> module_id;	/* The module identification string */
  char _desc[@self.desc_len()] -> desc;	/* The description string */

  /* Constructor with values for each attribute */
  @init()  [[auto]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_L3TData, 1)]]
  [[pack(4)]]
{
  uint32_t _accept -> accept;	/* Module trigger decision */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}

//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_L3TData, 2)]]
  [[pack(4)]]
{
  @enum Result(uint8_t) {
    Fail, 
    Pass, 
    None,
  }

  @enum Bias(uint8_t) {
    Unbiased,
    Biased
  }

  uint32_t _accept -> accept {
    /*  Returns L3T Decision : None = insufficient information/resources */
    Result   _bf_result:2 -> result;
    /*  Returns L3T Bias : Unbiased = recorded independent of decision */
    Bias     _bf_bias:1   -> bias;
    uint32_t _z:29;
  }

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}
} //- @package L3T
