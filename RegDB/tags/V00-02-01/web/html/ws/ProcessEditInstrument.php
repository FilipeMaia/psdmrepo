<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for editing an instrument
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
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "instrument description field can't be empty" );
} else
    die( "no valid instrument description" );

if( isset( $_POST['params'] )) {
    $str = stripslashes( trim( $_POST['params'] ));
    if( $str == 'null' ) $params = null;
    else {
        $params = json_decode( $str );
        if( is_null( $params ))
            die( "failed to translate JSON object with a list of parameters" );
    }
} else
    die( "no valid instrument parameters collection" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $instrument = RegDB::instance()->find_instrument_by_id ( $id ) or die( "no such instrument" );
    $instrument->set_description( $description );

    /* Replace parameters if the list is passed
     *
     * TODO: Perhaps we should implement a smarter algorithm here?
     */
    if( !is_null( $params )) {
        $instrument->remove_all_params();
        foreach( $params as $p ) {
            $param = $instrument->add_param( $p[0], $p[1], $p[2] )
                or die( "failed to add instrument parameter: {$pa}");
        }
    }
    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home')             header('Location: ../index.php');
        elseif ($actionSuccess == 'list_instruments') header('Location: ../index.php?action=list_instruments');
        elseif ($actionSuccess == 'view_instrument')  header('Location: ../index.php?action=view_instrument&id='.$instrument->id().'&name='.$instrument->name());
        elseif ($actionSuccess == 'edit_instrument')  header('Location: ../index.php?action=edit_instrument&id='.$instrument->id().'&name='.$instrument->name());
    }
    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>