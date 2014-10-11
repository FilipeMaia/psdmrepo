<?php

/**
 * Return the current configuration of the HDF5 translation service
 * for the experiment.
 * 
 * PARAMETERS:
 * 
 *   <exper_id>
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use \FileMgr\IfaceCtrlDb ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id  = $SVC->required_int('exper_id') ;

    $experiment = $SVC->regdb()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $section    = '' ;
    $instr_name = $experiment->instrument()->name() ;
    $exp_name   = $experiment->name() ;

    return array('config' => array (
        'auto'        => $experiment        ->find_param_by_name    (IfaceCtrlDb::$AUTO_TRANSLATE_HDF5) ? 1 : 0 ,
        'ffb'         => $SVC->ifacectrldb()->get_config_param_val_r('live-mode', 'dataset', $instr_name, $exp_name) === IfaceCtrlDb::$DATASET_FFB ? 1 : 0 ,
        'release_dir' => $SVC->ifacectrldb()->get_config_param_val_r('',          'release', $instr_name, $exp_name) ,
        'config_file' => $SVC->ifacectrldb()->get_config_param_val_r('',          'config',  $instr_name, $exp_name)
    )) ;
}) ;

?>
