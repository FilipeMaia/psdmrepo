<?php

require_once('AuthDB/AuthDB.inc.php');

/*
 * This script will lay out a form for creating a new role player.
 */

// TODO: This needs to be changed with the real test
//
//if( !RegDBAuth::isAuthenticated()) return;

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB();
    $authdb->begin();
    $applications = $authdb->applications();

    $regdb = new RegDB();
    $regdb->begin();
    $instruments = $regdb->instrument_names();
    array_push( $instruments, '' );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 900, 175 );
    echo $con
        ->label(   0,   0, 'Application:' )->select_input  ( 100,   0, 'application_name', $applications )
        ->label(   0,  30, 'Role:'        )->value_input   ( 100,  30, 'role_name' )
        ->label( 350,   0, 'Instrument:'  )->select_input  ( 450,   0, 'instrument_name', $instruments )->label( 625,  0, '( leave blank for &lt;any&gt; )', $false )
        ->label( 350,  30, 'Experiment:'  )->value_input   ( 450,  30, 'experiment_name'               )->label( 625, 30, '( leave blank for &lt;any&gt; )', $false )
        ->label(   0,  75, 'User:'        )->value_input   ( 100,  75, 'user' )
                                           ->label         ( 170, 100, 'or' )
        ->label(   0, 125, 'POSIX Group:' )->value_input   ( 100, 125, 'group' )
        ->html();

    $authdb->commit();
    $regdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>