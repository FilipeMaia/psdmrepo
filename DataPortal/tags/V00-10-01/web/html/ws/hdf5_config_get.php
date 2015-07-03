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
require_once 'regdb/regdb.inc.php' ;

use \FileMgr\IfaceCtrlDb ;
use \RegDB\RegDBDataSet ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $service  = $SVC->optional_enum('service' ,
                                     array('STANDARD', 'MONITORING') ,
                                     'STANDARD' ,
                                     array('ignore_case' => true, 'convert' => 'toupper')) ;

    $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $instr_name = $experiment->instrument()->name() ;
    $exp_name   = $experiment->name() ;

    $dataset = new RegDBDataSet($SVC->ifacectrldb($service)->get_config_param_val_r (
        'live-mode' ,
        'dataset' ,
        $instr_name ,
        $exp_name)) ;

    $config = array (
        'auto'        => $experiment->find_param_by_name(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service]) ? 1 : 0 ,
        'ffb'         => $dataset->is_ffb() ? 1 : 0 ,
        'release_dir' => $SVC->ifacectrldb($service)->get_config_param_val_r('', 'release', $instr_name, $exp_name) ,
        'config_file' => $SVC->ifacectrldb($service)->get_config_param_val_r('', 'config',  $instr_name, $exp_name) ,
        'stream'      => is_null($dataset->get_stream()) ? '' : $dataset->get_stream()
    ) ;
    switch ($service) {
        case 'MONITORING' :
            $config['outdir']      =        $SVC->ifacectrldb($service)->get_config_param_val_r('',          'output-dir',       $instr_name, $exp_name) ;
            $config['ccinsubdir']  = intval($SVC->ifacectrldb($service)->get_config_param_val_r('',          'output-cc-subdir', $instr_name, $exp_name));
            $config['exclusive']   = intval($SVC->ifacectrldb($service)->get_config_param_val_r('lsf',       'lsf-exclusive',    $instr_name, $exp_name)) ;
            $config['njobs']       = intval($SVC->ifacectrldb($service)->get_config_param_val_r('lsf',       'lsf-numproc',      $instr_name, $exp_name)) ;
            $config['ptile']       = intval($SVC->ifacectrldb($service)->get_config_param_val_r('lsf',       'lsf-ptile',        $instr_name, $exp_name)) ;
            $config['livetimeout'] = intval($SVC->ifacectrldb($service)->get_config_param_val_r('live-mode', 'live-timeout',     $instr_name, $exp_name)) ;
            break ;
    }
    return array('config' => $config) ;
}) ;

?>
