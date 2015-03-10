<?php

/**
 * Update specified configuration parameters of the HDF5 translation service
 * for the experiment and return the new state.
 * 
 * PARAMETERS:
 * 
 *   <exper_id> [<service>] [<auto>] [<ffb>] [<release_dir>] [<config_file>]
 *              [<outdir>] [<ccinsubdir>]
 *              [<exclusive>] [<njobs>] [<ptile>]
 *              [<livetimeout>]
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use \FileMgr\IfaceCtrlDb ;

/**
 * Update the specified configuration parameter in the Interface
 * Controller database if teh parameter's val;ue is not null.
 * Return back the most recent state of the parameter.
 *
 * @param RegDB\RegDBExperiment  $exper    - experiment
 * @param DataPortal\ServiceJSON $SVC      - Web service handler
 * @param String                 $service  - the name of the HDF5 translation service
 * @param String                 $sect     - configuration section
 * @param String                 $param    - configuration parameter
 * @param String                 $descr    - its description
 * @param String                 $val      - its value
 * @return String
 */
function update_config_param (
    $exper ,
    $SVC ,
    $service ,
    $sect ,
    $param ,
    $descr ,
    $val ,
    $type='String')
{
    $val_current = $SVC->ifacectrldb($service)->get_config_param_val_r (
        $sect ,
        $param ,
        $exper->instrument()->name() ,
        $exper->name()) ;

    if (is_null($val)) return $val_current ;

    $val = trim("{$val}") ;     // always turn it into the string even if it's a number
                                // to allow detecting '' which means a command to remove
                                // the parameter from the database.

    if ($val_current !== $val) {
        if ($val === '') {
            $SVC->ifacectrldb($service)->remove_config_param (
                $sect ,
                $param ,
                $exper->instrument()->name() ,
                $exper->name()) ;
        } else {
            $SVC->ifacectrldb($service)->set_config_param_val (
                $sect ,
                $param ,
                $exper->instrument()->name() ,
                $exper->name() ,
                $val ,
                $descr ,
                $type) ;
        }
    }
    return $val ;
}

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id    = $SVC->required_int ('exper_id') ;
    $service     = $SVC->optional_enum('service' ,
                                       array('STANDARD', 'MONITORING') ,
                                       'STANDARD' ,
                                       array('ignore_case' => true, 'convert' => 'toupper')) ;
    $auto        = $SVC->optional_int ('auto', null) ;
    $ffb         = $SVC->optional_int ('ffb',  null) ;
    $release_dir = $SVC->optional_str ('release_dir', null) ;
    $config_file = $SVC->optional_str ('config_file', null) ;

    $outdir      = $SVC->optional_str ('outdir', null) ;
    $ccinsubdir  = $SVC->optional_str('ccinsubdir', null) ;         // accepting String to allow removel when '' is provided

    $exclusive   = $SVC->optional_int ('exclusive', null) ;
    $njobs       = $SVC->optional_str ('njobs',     null) ;         // accepting String to allow removel when '' is provided
    $ptile       = $SVC->optional_str ('ptile',     null) ;         // accepting String to allow removel when '' is provided

    $livetimeout = $SVC->optional_str ('livetimeout', null) ;       // accepting String to allow removel when '' is provided

    $exper = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                "no experiment found for id={$exper_id}") ;

    /*
     * Update configuration parameters in the database and fill
     * in the dictionary to b ereturned by the service.
     */
    $config = array () ;

    $config['auto'] = $exper->find_param_by_name(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service]) ? 1 : 0 ;
    if (!is_null($auto)) {
        if ($config['auto'] !== $auto) {
            if ($auto) $exper->set_param   (IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service], '1') ;
            else       $exper->remove_param(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service]) ;
            $config['auto'] = $auto ;
        }
    }

    $config['ffb'] = update_config_param (
        $exper ,
        $SVC ,
        $service ,
        'live-mode' ,
        'dataset' ,
        'Format of the dataset in live mode' ,
        $ffb ? IfaceCtrlDb::$DATASET_FFB : IfaceCtrlDb::$DATASET_PSDM) === IfaceCtrlDb::$DATASET_FFB ? 1 : 0 ;

    $config['release_dir'] = update_config_param (
        $exper ,
        $SVC ,
        $service ,
        '' ,
        'release' ,
        'release directory from where to run the Translator' ,
        $release_dir) ;

    $config['config_file'] = update_config_param (
        $exper ,
        $SVC ,
        $service ,
        '' ,
        'config' ,
        'configuration file for the Translator psana job' ,
        $config_file) ;

    switch ($service) {
        case 'MONITORING' :

            $config['outdir'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                '' ,
                'output-dir' ,
                'configuration file for the Translator psana job' ,
                $outdir) ;

            $config['ccinsubdir'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                '' ,
                'output-cc-subdir' ,
                'Place Calib Cycle files at a separate subfolder of the output directory' ,
                $ccinsubdir ,
                'Integer') ? 1 : 0 ;

            $config['exclusive'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                'lsf' ,
                'lsf-exclusive' ,
                'Request exclusive use of batch nodes (dangerous!)' ,
                $exclusive ,
                'Integer') ? 1 : 0 ;

            $config['njobs'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                'lsf' ,
                'lsf-numproc' ,
                'The number of paralell MPI jobs used by the translation service' ,
                $njobs ,
                'Integer') ;

            $config['ptile'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                'lsf' ,
                'lsf-ptile' ,
                'Maximum number of processes per node. Set to 0 to not use this option.' ,
                $ptile ,
                'Integer') ;

            $config['livetimeout'] = update_config_param (
                $exper ,
                $SVC ,
                $service ,
                'live-mode' ,
                'live-timeout' ,
                'The number of seconds to wait in the live mode translation' ,
                $livetimeout ,
                'Integer') ;

            break ;
    }
    return array('config' => $config) ;
}) ;

?>
