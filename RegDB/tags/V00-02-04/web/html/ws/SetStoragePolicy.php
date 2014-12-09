<?php

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use DataPortal\Config;
use DataPortal\DataPortalException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;


/*
 * This script will process a request for setting/updating a value
 * of the data retention policy parameters in the global scope or
 * for an experiment.
 */
header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */

function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
    print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
    exit;
}

$exper_id         = null;
$short_ctime      = null;
$short_retention  = null;
$medium_quota     = null;
$medium_ctime     = null;
$medium_retention = null;

if( isset( $_GET['exper_id'] )) {

    $exper_id = trim( $_GET['exper_id'] );
    if( $exper_id == '' ) report_error( "experiment identifier can't be empty" );

    if( isset( $_GET['short_ctime'] )) {
        $str = trim( $_GET['short_ctime'] );
        if( $str ) {
            $short_ctime = LusiTime::parse($str);
            if( is_null($short_ctime )) report_error( "invalid value for the SHORT-TERM CTIME overrride parameter" );
        }
    }
    if( isset( $_GET['short_retention'] )) {
        $short_retention = intval( trim( $_GET['short_retention'] ));
    }
    if( isset( $_GET['medium_quota'] )) {
        $medium_quota = intval( trim( $_GET['medium_quota'] ));
    }
    if( isset( $_GET['medium_ctime'] )) {
        $str = trim( $_GET['medium_ctime'] );
        if( $str ) {
            $medium_ctime = LusiTime::parse($str);
            if( is_null($medium_ctime )) report_error( "invalid value for the MEDIUM-TERM CTIME overrride parameter" );
        }
    }
    if( isset( $_GET['medium_retention'] )) {
        $medium_retention = intval( trim( $_GET['medium_retention'] ));
    }

} else {

    if( isset( $_GET['default_short_ctime'] )) {
        $short_ctime = trim( $_GET['default_short_ctime'] );
        if( $short_ctime != '' ) {
            $short_ctime = LusiTime::parse($str);
            if( is_null($short_ctime )) report_error( "invalid value for the default SHORT-TERM CTIME overrride parameter" );
        }
    }
    if( isset( $_GET['default_short_retention'] )) {
        $short_retention = intval( trim( $_GET['default_short_retention'] ));
    }
    if( isset( $_GET['default_medium_quota'] )) {
        $medium_quota = intval( trim( $_GET['default_medium_quota'] ));
    }
    if( isset( $_GET['default_medium_ctime'] )) {
        $medium_ctime = trim( $_GET['default_medium_ctime'] );
        if( $medium_ctime != '' ) {
            $medium_ctime = LusiTime::parse($str);
            if( is_null($medium_ctime )) report_error( "invalid value for the default MEDIUM-TERM CTIME overrride parameter" );
        }
    }
    if( isset( $_GET['default_medium_retention'] )) {
        $medium_retention = intval( trim( $_GET['default_medium_retention'] ));
    }
}
try {

	if( !RegDBAuth::instance()->canEdit())
		report_error( 'not autorized to modify MEDIUM-TERM storage quota parameter for the experiment' );

    if( is_null($exper_id)) {

        $config = Config::instance();
        $config->begin();

        if( !is_null($short_ctime     )) $config->set_policy_param('SHORT-TERM',  'CTIME',     $short_ctime == '' ? $short_ctime : $short_ctime->toStringDay());
        if( !is_null($short_retention )) $config->set_policy_param('SHORT-TERM',  'RETENTION', $short_retention );
        if( !is_null($medium_quota    )) $config->set_policy_param('MEDIUM-TERM', 'QUOTA',     $medium_quota );
        if( !is_null($medium_ctime    )) $config->set_policy_param('MEDIUM-TERM', 'CTIME',     $medium_ctime == '' ? $medium_ctime : $medium_ctime->toStringDay());
        if( !is_null($medium_retention)) $config->set_policy_param('MEDIUM-TERM', 'RETENTION', $medium_retention );

        $config->commit();

    } else {

        $regdb = RegDB::instance();
    	$regdb->begin();
        
        $experiment = $regdb->find_experiment_by_id( $exper_id );
        if( !$experiment ) report_error( 'no such experiment' );

        function set_param($name, $value) {
            global $experiment;
            $param = $experiment->set_param( $name, is_null($value) ? '' : $value );
            if( is_null( $param )) report_error( "failed to set value='{$value}' to experiment-specific parameter {$name}" );
        }
        set_param('SHORT-TERM-DISK-QUOTA-CTIME',      $short_ctime ? $short_ctime->toStringDay() : '');
        set_param('SHORT-TERM-DISK-QUOTA-RETENTION',  $short_retention);
        set_param('MEDIUM-TERM-DISK-QUOTA',           $medium_quota);
        set_param('MEDIUM-TERM-DISK-QUOTA-CTIME',     $medium_ctime ? $medium_ctime->toStringDay() : '' );
        set_param('MEDIUM-TERM-DISK-QUOTA-RETENTION', $medium_retention);

        $regdb->commit();
    }

    $status_encoded = json_encode( "success" );
    print <<< HERE
{
  "Status": {$status_encoded}
}
HERE;

} catch( DataPortalException $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( LusiTimeException   $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( RegDBException      $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>