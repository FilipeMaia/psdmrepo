<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDBAuth;
use RegDB\RegDBHtml;

/*
 * This script will generate a module with input elements for the filter form
 * which is used as afilter for user accounts.
 */
header( 'Content-type: text/html' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print $msg;
    exit;
}
if( isset( $_GET['string2search'] )) {
    $string2search = trim( $_GET['string2search'] );
} else {
    report_error( RegDBAuth::reporErrorHtml( "no valid string to search in user accounts" ));
}
if( isset( $_GET['scope'] )) {
    $scope = trim( $_GET['scope'] );
    if( $scope == '' )
        report_error( RegDBAuth::reporErrorHtml( "search scope can't be empty" ));
} else {
    report_error( RegDBAuth::reporErrorHtml( "no valid scope to earch in user accounts" ));
}

/* Proceed with the operation
 */
try {

    $pattern_help = 'a substring to be search in user account names or user names';

    $con = new RegDBHtml( 0, 0, 400, 65 );
    echo $con

        ->value      (   0,  3, '<b>Search for user</b>' )
        ->value_input( 105,  0, 'accounts_pattern', $string2search, $pattern_help, 8 )

        ->value      ( 210,  3, '<b>in</b>' )
        ->radio_input( 240,  3, 'scope', 'uid_and_name', $scope == 'uid_and_name' )->label( 260,   3, '<b>UID & name</b>', false )
        ->radio_input( 240, 23, 'scope', 'uid',          $scope == 'uid'          )->label( 260,  23, '<b>UID</b>',        false )
        ->radio_input( 240, 43, 'scope', 'name',         $scope == 'name'         )->label( 260,  43, '<b>name</b>',       false )

        ->button     ( 102, 35, 'accounts_filter_button', 'Search', 'update the table using this filter' )
        ->html();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>