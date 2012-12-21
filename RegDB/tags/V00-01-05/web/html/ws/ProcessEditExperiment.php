<?php

require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LusiTime\LusiTime;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for editing an experiment
 * in the database.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );

if( isset( $_POST['begin_time'])) {
    $begin_time = LusiTime::parse( trim( $_POST['begin_time'] ));
    if( is_null( $begin_time ))
        die("begin time has invalid format");
} else
    die( "no begin time for experiment" );

if( isset( $_POST['end_time'])) {
    $end_time = LusiTime::parse( trim( $_POST['end_time'] ));
    if( is_null( $end_time ))
        die("end time has invalid format");
} else
    die( "no end time for experiment" );

if( isset( $_POST['contact'] )) {
    $contact = trim( $_POST['contact'] );
    if( $contact == '' )
        die( "contact information field can't be empty" );
} else
    die( "no valid contact information" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "experiment description field can't be empty" );
} else
    die( "no valid experiment description" );

if( isset( $_POST['params'] )) {
    $str = stripslashes( trim( $_POST['params'] ));
    if( $str == 'null' ) $params = null;
    else {
        $params = json_decode( $str );
        if( is_null( $params ))
            die( "failed to translate JSON object with a list of parameters" );
    }
} else
    die( "no valid experiment parameters collection" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id ( $id ) or die( "no such experiment" );
    $experiment->set_description( $description );
    $experiment->set_contact_info( $contact );
    $experiment->set_interval( $begin_time, $end_time );


    /* Replace parameters if the list is passed
     *
     * TODO: Perhaps we should implement a smarter algorithm here?
     */
    if( !is_null( $params )) {
        $experiment->remove_all_params();
        foreach( $params as $p ) {
            $param = $experiment->add_param( $p[0], $p[1], $p[2] )
                or die( "failed to add experiment parameter: {$pa}");
        }
    }
    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home')             header('Location: ../index.php');
        elseif ($actionSuccess == 'list_experiments') header('Location: ../index.php?action=list_experiments');
        elseif ($actionSuccess == 'view_experiment')  header('Location: ../index.php?action=view_experiment&id='.$experiment->id().'&name='.$experiment->name());
        elseif ($actionSuccess == 'edit_experiment')  header('Location: ../index.php?action=edit_experiment&id='.$experiment->id().'&name='.$experiment->name());
    }
    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>