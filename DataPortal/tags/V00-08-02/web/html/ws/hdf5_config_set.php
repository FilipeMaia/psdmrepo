<?php

/**
 * Update specified configuration parameters of the HDF5 translation service
 * for the experiment and return the new state.
 * 
 * PARAMETERS:
 * 
 *   <exper_id> [<auto>] [<ffb>] [<release_dir>] [<config_file>]
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use \FileMgr\IfaceCtrlDb ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id    = $SVC->required_int('exper_id') ;
    $auto        = $SVC->optional_int('auto', null) ;
    $ffb         = $SVC->optional_int('ffb',  null) ;
    $release_dir = $SVC->optional_str('release_dir', null) ;
    $config_file = $SVC->optional_str('config_file', null) ;

    $experiment = $SVC->regdb()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $instr_name = $experiment->instrument()->name() ;
    $exp_name   = $experiment->name() ;

    if (!is_null($auto)) {
        $auto_current = $experiment->find_param_by_name(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5) ? 1 : 0 ;
        if ($auto_current != $auto) {
            if ($auto) $experiment->set_param   (IfaceCtrlDb::$AUTO_TRANSLATE_HDF5, '1') ;
            else       $experiment->remove_param(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5) ;
        }
    }
    if (!is_null($ffb)) {
        $dataset = $ffb ? IfaceCtrlDb::$DATASET_FFB : IfaceCtrlDb::$DATASET_PSDM ;
        $dataset_current = $SVC->ifacectrldb()->get_config_param_val_r('live-mode', 'dataset', $instr_name, $exp_name) ;
        if ($dataset_current !== $dataset) {
            $dataset_descr = 'Format of the dataset in live mode' ;
            $SVC->ifacectrldb()->set_config_param_val('live-mode', 'dataset', $instr_name, $exp_name, $dataset, $dataset_descr) ;
        }
    }
    if (!is_null($release_dir)) {
        $release_dir         = trim($release_dir) ;
        $release_dir_current = $SVC->ifacectrldb()->get_config_param_val_r('', 'release', $instr_name, $exp_name) ;
        if ($release_dir_current !== $release_dir) {
            if ($release_dir === '') {
                $SVC->ifacectrldb()->remove_config_param('', 'release', $instr_name, $exp_name) ;
            } else {
                $release_dir_descr = 'release directory from where to run the Translator' ;
                $SVC->ifacectrldb()->set_config_param_val('', 'release', $instr_name, $exp_name, $release_dir, $descr) ;
            }
        }
    }
    if (!is_null($config_file)) {
        $config_file          = trim($config_file) ;
        $config_filer_current = $SVC->ifacectrldb()->get_config_param_val_r('', 'config', $instr_name, $exp_name) ;
        if ($config_filer_current !== $config_file) {
            if ($config_file === '') {
                $SVC->ifacectrldb()->remove_config_param('', 'config', $instr_name, $exp_name) ;
            } else {
                $descr = 'configuration file for the Translator psana job' ;
                $SVC->ifacectrldb()->set_config_param_val('', 'config', $instr_name, $exp_name, $config_file, $descr) ;
            }
        }
    }
    return array('config' => array (
        'auto'        => $experiment        ->find_param_by_name    (IfaceCtrlDb::$AUTO_TRANSLATE_HDF5) ? 1 : 0 ,
        'ffb'         => $SVC->ifacectrldb()->get_config_param_val_r('live-mode', 'dataset', $instr_name, $exp_name) === IfaceCtrlDb::$DATASET_FFB ? 1 : 0 ,
        'release_dir' => $SVC->ifacectrldb()->get_config_param_val_r('',          'release', $instr_name, $exp_name) ,
        'config_file' => $SVC->ifacectrldb()->get_config_param_val_r('',          'config',  $instr_name, $exp_name)
    )) ;
}) ;

?>
