<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDB;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print json_encode ( array (
        "ResultSet" => array("Result" => array()),
        "Message" => $msg
    ));
    exit;
}
if( !isset( $_GET['role_id'] )) report_error( "no valid role identifier in the request" );
$role_id = trim( $_GET['role_id'] );

function role2array ($role) {
    $player = $role->player();
    $group_url = $player['group'];
    if( $group_url == '' ) {
        $group_url = '';
        $user_url  = $player['user'];
        $comment = '';
    } else {
        $group_url = '<a href="javascript:view_group('."'".$group_url."'".')">'.$group_url.'</a>';
        $user_url  = '';
        $comment = 'all members of the group';
    }

    $instrument_url = '&lt;any&gt;';
    $experiment_url = '&lt;any&gt;';
    $exper_id = $role->exper_id();
    if( !is_null( $exper_id )) {
        $experiment = RegDB::instance()->find_experiment_by_id( $exper_id )
            or report_error( "no experiment with id={$exper_id} found." );
        $instrument_url = $experiment->instrument()->name();
        $experiment_url = $experiment->name();
    }
    return array (
        "instrument" => $instrument_url,
        "experiment" => $experiment_url,
        "group"      => $group_url,
        "user"       => $player['user'],
        "comment"    => $comment
    );
}

/*
 * Analyze and process the request
 */
try {
    AuthDB::instance()->begin();
    RegDB::instance()->begin();

    $role_payers = array();
    foreach (AuthDB::instance()->roles_by_id( $role_id ) as $r)
        array_push ($role_payers, role2array($r));

    RegDB::instance()->commit();
    AuthDB::instance()->commit();


    print json_encode (array ( "ResultSet" => array("Result" => $role_payers)));

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }


?>
