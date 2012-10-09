<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDBHtml;

/*
 * This script will lay out a form for creating a new role.
 */
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

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 700, 125 );
    echo $con
        ->label         (   0,  25, 'Application:' )
        ->value_input   ( 100,  25, 'application_name' )
        ->label         ( 275,  25, 'or' )
        ->select_input  ( 305,  22, 'application_name_select', $applications )
        ->label         (   0,  75, 'Role:' )
        ->value_input   ( 100,  75, 'role_name' )
        ->html();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>