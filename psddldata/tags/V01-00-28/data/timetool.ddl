@include "psddldata/camera.ddl";
@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/Src.hh")]];

@package TimeTool {

//------------------ EventLogic ------------------
@type EventLogic
  [[value_type]]
  [[pack(4)]]
{
  @enum LogicOp (uint8_t) {
    L_OR=0,
    L_AND=1,
    L_OR_NOT=2,
    L_AND_NOT=3
  }
  uint32_t _Code {
    /*  Event Code */
    uint8_t  _event_code:8  -> event_code;
    /*  Logic Operation */
    LogicOp  _logic_op:2    -> logic_op;

    uint32_t _z:22;
  }

  /* Constructor which takes values for every attribute */
  @init() [[auto, inline]];
}

//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_TimeToolConfig,1)]]
  [[config_type]]
  [[pack(4)]]
{
  @enum Axis (uint8_t) {
    X =0,
    Y =1
  }

  uint32_t _Control {
    /*  Time Axis of Image */
    Axis     _project_axis:1       -> project_axis;
    /*  Record Raw Image into Event */
    uint8_t  _write_image:1        -> write_image;
    /*  Record Time Axis Projections into Event */
    uint8_t  _write_projections:1  -> write_projections;
    /*  Subtract Sideband Region */
    uint8_t  _subtract_sideband:1  -> subtract_sideband;
    /*  Number of Digital Filter Weights */
    uint16_t _number_of_weights:16 -> number_of_weights;
    /*  Pixel to Time Calibration Polynomial Dimension */
    uint8_t  _calib_poly_dim:4     -> calib_poly_dim;
    /*  Length of EPICS PV base name */
    uint8_t  _base_name_length:8   -> base_name_length;
  }

  /*  Number of Beam Logic Event Codes */
  uint16_t _number_of_beam_event_codes -> number_of_beam_event_codes;

  /*  Number of Laser Logic Event Codes */
  uint16_t _number_of_laser_event_codes -> number_of_laser_event_codes;

  /*  Projection Minimum Value for Validation */
  uint32_t _signal_cut -> signal_cut;

  /*  Signal Region Coordinates Start */
  Camera.FrameCoord _sig_roi_lo -> sig_roi_lo;
  /*  Signal Region Coordinates End */
  Camera.FrameCoord _sig_roi_hi -> sig_roi_hi;

  /*  Sideband Region Coordinates Start */
  Camera.FrameCoord _sb_roi_lo  -> sb_roi_lo;
  /*  Sideband Region Coordinates End */
  Camera.FrameCoord _sb_roi_hi  -> sb_roi_hi;

  /*  Sideband Rolling Average Factor (1/NFrames) */
  double _sb_convergence  -> sb_convergence;

  /*  Reference Rolling Average Factor (1/NFrames) */
  double _ref_convergence -> ref_convergence;

  /*  Beam Logic Event Codes */
  EventLogic _beam_logic[@self.number_of_beam_event_codes()] -> beam_logic;

  /*  Laser Logic Event Codes */
  EventLogic _laser_logic[@self.number_of_laser_event_codes()] -> laser_logic;

  /*  Digital Filter Weights */
  double _weights[@self.number_of_weights()] -> weights;

  /*  Pixel to Time Calibration Polynomial */
  double   _calib_poly[@self.calib_poly_dim()] -> calib_poly;

  /*  EPICS PV base name */
  char _base_name[@self.base_name_length()] -> base_name;

  /*  Size of projections */
  uint32_t signal_projection_size()
  [[language("C++")]] @{ return (@self.write_projections() ? (@self.project_axis()==X ?
  @self.sig_roi_hi().column()-@self.sig_roi_lo().column()+1 :
  @self.sig_roi_hi().row   ()-@self.sig_roi_lo().row   ()+1) : 0); @}

  uint32_t sideband_projection_size()
  [[language("C++")]] @{ return ((@self.write_projections()&&@self.subtract_sideband()) ?
  (@self.project_axis()==X ?
     @self.sb_roi_hi().column()-@self.sb_roi_lo().column()+1 :
     @self.sb_roi_hi().row   ()-@self.sb_roi_lo().row   ()+1) : 0); @}

  /* Constructor which takes values for every attribute */
  @init() [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(number_of_beam_event_codes -> _number_of_beam_event_codes,
        number_of_laser_event_codes -> _number_of_laser_event_codes,
        number_of_weights -> _number_of_weights,
        calib_poly_dim    -> _calib_poly_dim,
        base_name_length  -> _base_name_length) [[inline]];
}

//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_TimeToolConfig,2)]]
  [[config_type]]
  [[pack(4)]]
{
  @enum Axis (uint8_t) {
    X =0,
    Y =1
  }

  uint32_t _Control {
    /*  Time Axis of Image */
    Axis     _project_axis:1       -> project_axis;
    /*  Record Raw Image into Event */
    uint8_t  _write_image:1        -> write_image;
    /*  Record Time Axis Projections into Event */
    uint8_t  _write_projections:1  -> write_projections;
    /*  Subtract Sideband Region */
    uint8_t  _subtract_sideband:1  -> subtract_sideband;
    /*  Use Separate Reference Region */
    uint8_t  _use_reference_roi:1  -> use_reference_roi;
    /*  Number of Digital Filter Weights */
    uint16_t _number_of_weights:16 -> number_of_weights;
    /*  Pixel to Time Calibration Polynomial Dimension */
    uint8_t  _calib_poly_dim:4     -> calib_poly_dim;
    /*  Length of EPICS PV base name */
    uint8_t  _base_name_length:8   -> base_name_length;
  }

  /*  Number of Beam Logic Event Codes */
  uint16_t _number_of_beam_event_codes -> number_of_beam_event_codes;

  /*  Number of Laser Logic Event Codes */
  uint16_t _number_of_laser_event_codes -> number_of_laser_event_codes;

  /*  Projection Minimum Value for Validation */
  uint32_t _signal_cut -> signal_cut;

  /*  Signal Region Coordinates Start */
  Camera.FrameCoord _sig_roi_lo -> sig_roi_lo;
  /*  Signal Region Coordinates End */
  Camera.FrameCoord _sig_roi_hi -> sig_roi_hi;

  /*  Sideband Region Coordinates Start */
  Camera.FrameCoord _sb_roi_lo  -> sb_roi_lo;
  /*  Sideband Region Coordinates End */
  Camera.FrameCoord _sb_roi_hi  -> sb_roi_hi;

  /*  Sideband Rolling Average Factor (1/NFrames) */
  double _sb_convergence  -> sb_convergence;

  /*  Reference Region Coordinates Start */
  Camera.FrameCoord _ref_roi_lo  -> ref_roi_lo;
  /*  Sideband Region Coordinates End */
  Camera.FrameCoord _ref_roi_hi  -> ref_roi_hi;

  /*  Reference Rolling Average Factor (1/NFrames) */
  double _ref_convergence -> ref_convergence;

  /*  Beam Logic Event Codes */
  EventLogic _beam_logic[@self.number_of_beam_event_codes()] -> beam_logic;

  /*  Laser Logic Event Codes */
  EventLogic _laser_logic[@self.number_of_laser_event_codes()] -> laser_logic;

  /*  Digital Filter Weights */
  double _weights[@self.number_of_weights()] -> weights;

  /*  Pixel to Time Calibration Polynomial */
  double   _calib_poly[@self.calib_poly_dim()] -> calib_poly;

  /*  EPICS PV base name */
  char _base_name[@self.base_name_length()] -> base_name;

  /*  Size of projections */
  uint32_t signal_projection_size()
  [[language("C++")]] @{ return (@self.write_projections() ? (@self.project_axis()==X ?
  @self.sig_roi_hi().column()-@self.sig_roi_lo().column()+1 :
  @self.sig_roi_hi().row   ()-@self.sig_roi_lo().row   ()+1) : 0); @}

  uint32_t sideband_projection_size()
  [[language("C++")]] @{ return ((@self.write_projections()&&@self.subtract_sideband()) ?
  (@self.project_axis()==X ?
     @self.sb_roi_hi().column()-@self.sb_roi_lo().column()+1 :
     @self.sb_roi_hi().row   ()-@self.sb_roi_lo().row   ()+1) : 0); @}

  uint32_t reference_projection_size()
  [[language("C++")]] @{ return ((@self.write_projections()&&@self.use_reference_roi()) ?
  (@self.project_axis()==X ?
     @self.ref_roi_hi().column()-@self.ref_roi_lo().column()+1 :
     @self.ref_roi_hi().row   ()-@self.ref_roi_lo().row   ()+1) : 0); @}

  /* Constructor which takes values for every attribute */
  @init() [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(number_of_beam_event_codes -> _number_of_beam_event_codes,
        number_of_laser_event_codes -> _number_of_laser_event_codes,
        number_of_weights -> _number_of_weights,
        calib_poly_dim    -> _calib_poly_dim,
        base_name_length  -> _base_name_length) [[inline]];
}

//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_TimeToolData, 1)]]
  [[config(ConfigV1)]]
{
  @enum EventType (uint32_t) {
    Dark,           // No Laser
    Reference,      // No Beam
    Signal          // Laser and Beam
  }

  /*  Event designation */
  EventType _event_type   -> event_type;
  
  uint32_t  _z;

  /*  Amplitude of the edge */
  double _amplitude      -> amplitude;

  /*  Filtered pixel position of the edge */
  double _position_pixel -> position_pixel;

  /*  Filtered time position of the edge */
  double _position_time  -> position_time;

  /*  Full-width half maximum of filtered edge (in pixels) */
  double _position_fwhm  -> position_fwhm;

  /*  Amplitude of reference at the edge */
  double _ref_amplitude  -> ref_amplitude;

  /*  Amplitude of the next largest edge */
  double _nxt_amplitude  -> nxt_amplitude;

  /*  Projected signal */
  int32_t _projected_signal   [@config.signal_projection_size()]   -> projected_signal;

  /*  Projected reference */
  int32_t _projected_sideband [@config.sideband_projection_size()] -> projected_sideband;

  /* Constructor with values for each attributes */
  @init(event_type     -> _event_type,
        amplitude      -> _amplitude, 
        position_pixel -> _position_pixel,
        position_time  -> _position_time,
        position_fwhm  -> _position_fwhm,
        ref_amplitude  -> _ref_amplitude,
        nxt_amplitude  -> _nxt_amplitude) [[inline]];
}

//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_TimeToolData, 2)]]
  [[config(ConfigV2)]]
{
  @enum EventType (uint32_t) {
    Dark,           // No Laser
    Reference,      // No Beam
    Signal          // Laser and Beam
  }

  /*  Event designation */
  EventType _event_type   -> event_type;
  
  uint32_t  _z;

  /*  Amplitude of the edge */
  double _amplitude      -> amplitude;

  /*  Filtered pixel position of the edge */
  double _position_pixel -> position_pixel;

  /*  Filtered time position of the edge */
  double _position_time  -> position_time;

  /*  Full-width half maximum of filtered edge (in pixels) */
  double _position_fwhm  -> position_fwhm;

  /*  Amplitude of reference at the edge */
  double _ref_amplitude  -> ref_amplitude;

  /*  Amplitude of the next largest edge */
  double _nxt_amplitude  -> nxt_amplitude;

  /*  Projected signal */
  int32_t _projected_signal   [@config.signal_projection_size()]   -> projected_signal;

  /*  Projected sideband */
  int32_t _projected_sideband [@config.sideband_projection_size()] -> projected_sideband;

  /*  Projected reference */
  int32_t _projected_reference [@config.reference_projection_size()] -> projected_reference;

  /* Constructor with values for each attributes */
  @init(event_type     -> _event_type,
        amplitude      -> _amplitude, 
        position_pixel -> _position_pixel,
        position_time  -> _position_time,
        position_fwhm  -> _position_fwhm,
        ref_amplitude  -> _ref_amplitude,
        nxt_amplitude  -> _nxt_amplitude) [[inline]];
}
} //- @package TimeTool

