<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDB;
use RegDB\RegDBHtml;

/*
 * This script will lay out a form for creating a new role player.
 */
header( 'Content-type: text/html' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print $msg;
    exit;
}

/* Proceed with the operation
 */
try {
    AuthDB::instance()->begin();
    $applications = AuthDB::instance()->applications();
    AuthDB::instance()->commit();

    RegDB::instance()->begin();
    $instruments = RegDB::instance()->instrument_names();
    array_push( $instruments, '' );
    RegDB::instance()->commit();



    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 900, 175 );
    echo $con
        ->label(   0,   0, 'Application:' )->select_input  ( 100,   0, 'application_name', $applications )
        ->label(   0,  30, 'Role:'        )->value_input   ( 100,  30, 'role_name' )
        ->label( 350,   0, 'Instrument:'  )->select_input  ( 450,   0, 'instrument_name', $instruments )->label( 625,  0, '( leave blank for &lt;any&gt; )', false )
        ->label( 350,  30, 'Experiment:'  )->value_input   ( 450,  30, 'experiment_name'               )->label( 625, 30, '( leave blank for &lt;any&gt; )', false )
        ->label(   0,  75, 'User:'        )->value_input   ( 100,  75, 'user' )
                                           ->label         ( 170, 100, 'or' )
        ->label(   0, 125, 'POSIX Group:' )->value_input   ( 100, 125, 'group' )
        ->html();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>