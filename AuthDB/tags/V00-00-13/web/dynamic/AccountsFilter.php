<?php

require_once('AuthDB/AuthDB.inc.php');

/*
 * This script will generate a module with input elements for the filter form
 * which is used as afilter for user accounts.
 */
if( isset( $_GET['string2search'] )) {
    $string2search = trim( $_GET['string2search'] );
} else {
    print( RegDBAuth::reporErrorHtml( "no valid string to search in user accounts" ));
    exit;
}
if( isset( $_GET['scope'] )) {
    $scope = trim( $_GET['scope'] );
    if( $scope == '' ) {
        print( RegDBAuth::reporErrorHtml( "search scope can't be empty" ));
        exit;
    }
} else {
    print( RegDBAuth::reporErrorHtml( "no valid scope to earch in user accounts" ));
    exit;
}

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB;
    $authdb->begin();

    $pattern_help = 'a substring to be search in user account names or user names';

    /* Proceed to the operation
     */
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 400, 60 );
    echo $con

        ->value      (   0,  3, '<b>Search for user</b>' )
        ->value_input( 105,  0, 'accounts_pattern', $string2search, $pattern_help, 8 )

        ->value      ( 210,  3, '<b>in</b>' )
        ->radio_input( 240,  3, 'scope', 'uid_and_name', $scope == 'uid_and_name' )->label( 260,   3, '<b>UID & name</b>', false )
        ->radio_input( 240, 23, 'scope', 'uid',          $scope == 'uid'          )->label( 260,  23, '<b>UID</b>',        false )
        ->radio_input( 240, 43, 'scope', 'name',         $scope == 'name'         )->label( 260,  43, '<b>name</b>',       false )

        ->button     ( 365,  0, 'accounts_filter_button', 'Go', 'update the table using this filter' )
        ->html();

    $authdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>