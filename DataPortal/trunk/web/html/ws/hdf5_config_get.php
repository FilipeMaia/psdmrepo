<?php

/**
 * Return the current configuration of the HDF5 translation service
 * for the experiment.
 * 
 * PARAMETERS:
 * 
 *   <exper_id> [<service>]
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use \FileMgr\IfaceCtrlDb ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $service  = $SVC->optional_enum('service' ,
                                     array('STANDARD', 'MONITORING') ,
                                     'STANDARD' ,
                                     array('ignore_case' => true, 'convert' => 'toupper')) ;

    $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $section    = '' ;
    $instr_name = $experiment->instrument()->name() ;
    $exp_name   = $experiment->name() ;

    $config = array (
        'auto'        => $experiment                ->find_param_by_name(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service]) ? 1 : 0 ,
        'ffb'         => $SVC->ifacectrldb($service)->get_config_param_val_r('live-mode', 'dataset', $instr_name, $exp_name) === IfaceCtrlDb::$DATASET_FFB ? 1 : 0 ,
        'release_dir' => $SVC->ifacectrldb($service)->get_config_param_val_r('',          'release', $instr_name, $exp_name) ,
        'config_file' => $SVC->ifacectrldb($service)->get_config_param_val_r('',          'config',  $instr_name, $exp_name)
    ) ;
    switch ($service) {
        case 'MONITORING' :
            $config['njobs']      = $SVC->ifacectrldb($service)->get_config_param_val_r('pazlib', 'lsf-numproc',      $instr_name, $exp_name) ;
            $config['outdir']     = $SVC->ifacectrldb($service)->get_config_param_val_r('',       'output-dir',       $instr_name, $exp_name) ;
            $config['ccinsubdir'] = $SVC->ifacectrldb($service)->get_config_param_val_r('',       'output-cc-subdir', $instr_name, $exp_name) ? 1 : 0 ;
            break ;
    }
    return array('config' => $config) ;
}) ;

?>
