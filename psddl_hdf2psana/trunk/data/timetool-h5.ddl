@include "psddldata/timetool.ddl";

@package TimeTool  {


//------------------ EventLogic ------------------
@h5schema EventLogic
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute event_code;
    @attribute logic_op;
  }
} 

//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute project_axis;
    @attribute write_image;
    @attribute write_projections;
    @attribute subtract_sideband;
    @attribute number_of_weights;
    @attribute calib_poly_dim;
    @attribute number_of_beam_event_codes;
    @attribute number_of_laser_event_codes;
    @attribute signal_cut;
    @attribute sig_roi_lo;
    @attribute sig_roi_hi;
    @attribute sb_roi_lo;
    @attribute sb_roi_hi;
    @attribute sb_convergence;
    @attribute ref_convergence;
    @attribute base_name_length;
    @attribute base_name [[vlen]];
  }
  @dataset beam_logic;
  @dataset laser_logic;
  @dataset weights;
  @dataset calib_poly;
}

//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute project_axis;
    @attribute write_image;
    @attribute write_projections;
    @attribute subtract_sideband;
    @attribute use_reference_roi;
    @attribute number_of_weights;
    @attribute calib_poly_dim;
    @attribute number_of_beam_event_codes;
    @attribute number_of_laser_event_codes;
    @attribute signal_cut;
    @attribute sig_roi_lo;
    @attribute sig_roi_hi;
    @attribute sb_roi_lo;
    @attribute sb_roi_hi;
    @attribute ref_roi_lo;
    @attribute ref_roi_hi;
    @attribute sb_convergence;
    @attribute ref_convergence;
    @attribute base_name_length;
    @attribute base_name [[vlen]];
  }
  @dataset beam_logic;
  @dataset laser_logic;
  @dataset weights;
  @dataset calib_poly;
}

//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
{
  @dataset data {
    @attribute event_type;
    @attribute amplitude;
    @attribute position_pixel;
    @attribute position_time;
    @attribute position_fwhm;
    @attribute ref_amplitude;
    @attribute nxt_amplitude;
  }
  @dataset projected_signal [[zero_dims]];
  @dataset projected_sideband [[zero_dims]];
}  

//------------------ DataV2 ------------------
@h5schema DataV2
  [[version(0)]]
{
  @dataset data {
    @attribute event_type;
    @attribute amplitude;
    @attribute position_pixel;
    @attribute position_time;
    @attribute position_fwhm;
    @attribute ref_amplitude;
    @attribute nxt_amplitude;
  }
  @dataset projected_signal [[zero_dims]];
  @dataset projected_sideband [[zero_dims]];
  @dataset projected_reference [[zero_dims]];
}  

} //- @TimeTool
