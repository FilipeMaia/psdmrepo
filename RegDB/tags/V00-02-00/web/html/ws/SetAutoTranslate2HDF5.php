<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBookAuth;
use LogBook\LogBookException;

use RegDB\RegDB;
use RegDB\RegDBException;

/*
 * This script will process a request for setting/updating a value
 * of a parameter triggering the auto-translation of HDF5 files of
 * an experiment.
 */
define(    'PARAM_NAME',    'AUTO_TRANSLATE_HDF5');
define('FFB_PARAM_NAME','FFB_AUTO_TRANSLATE_HDF5');

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print json_encode(
        array(
            "Status" => "error", "Message" => $msg ,
            "status" => "error", "message" => $msg
        )
    );
    exit;
}
function report_success() {
    print json_encode(
        array(
            "Status" => "success" ,
            "status" => "success"
        )
    );
}

if( isset( $_POST['exper_id'] )) {
    $exper_id = trim( $_POST['exper_id'] );
    if( $exper_id == '' ) report_error( "experiment identifier can't be empty" );
} else report_error( "no valid experiment identifier" );

$autotranslate2hdf5 = null;
if( isset( $_POST['autotranslate2hdf5'] ))
    $autotranslate2hdf5 = intval(trim( $_POST['autotranslate2hdf5'] ));

$ffb_autotranslate2hdf5 = null;
if( isset( $_POST['ffb_autotranslate2hdf5'] ))
    $ffb_autotranslate2hdf5 = intval(trim( $_POST['ffb_autotranslate2hdf5'] ));

try {
    $regdb = RegDB::instance();
    $regdb->begin();

    $experiment = $regdb->find_experiment_by_id( $exper_id );
    if( !$experiment ) report_error( 'no such experiment' );

    LogBookAuth::instance()->canPostNewMessages( $experiment->id()) or
        report_error( 'not autorized to manage Auto-Translation of HDF5 files for the experiment' );

    if(!is_null($autotranslate2hdf5)) {
        $param = $experiment->find_param_by_name( PARAM_NAME);
        if($autotranslate2hdf5) {
            if(is_null($param)) {
                $param = $experiment->set_param(PARAM_NAME, '1' );
                if( is_null( $param )) report_error( 'failed to modify the parameter' );
            }
        } else {
            if(!is_null($param)) $param = $experiment->remove_param(PARAM_NAME);
        }
    }
    if(!is_null($ffb_autotranslate2hdf5)) {
        $ffb_param = $experiment->find_param_by_name( FFB_PARAM_NAME);
        if($ffb_autotranslate2hdf5) {
            if(is_null($ffb_param)) {
                $ffb_param = $experiment->set_param(FFB_PARAM_NAME, '1' );
                if( is_null( $ffb_param )) report_error( 'failed to modify the parameter' );
            }
        } else {
            if(!is_null($ffb_param)) $ffb_param = $experiment->remove_param(FFB_PARAM_NAME);
        }
    }
    $regdb->commit();

    report_success();

} catch( LogBookException $e ) { report_error( $e->toHtml()); }
  catch( RegDBException   $e ) { report_error( $e->toHtml()); }

?>
