<?php

require_once( 'RegDB/RegDB.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for creating a new instrument
 * in the database.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
    if( $instrument_name == '' )
        die( "instrument name can't be empty" );
} else
    die( "no valid instrument name" );

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
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->register_instrument (
        $instrument_name, $description );

    /* Add parameters if any
     */
    if( !is_null( $params ))
        foreach( $params as $p )
            $param = $instrument->add_param( $p[0], $p[1], $p[2] )
                or die( "failed to add instrument parameter: {$pa}");

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_instruments' )
            header( 'Location: index.php?action=list_instruments' );
        else if( $actionSuccess == 'view_instrument' )
            header( 'Location: index.php?action=view_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else if( $actionSuccess == 'edit_instrument' )
            header( 'Location: index.php?action=edit_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>