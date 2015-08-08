<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;

/*
 * This script will generate a module with input elements for the filter form
 * which is used as a filter for groups the current user is authorized to manage.
 */
header( 'Content-type: text/html' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print $msg;
    exit;
}

try {
    RegDB::instance()->begin();

    /* The list of groups will depend on privileges of a user who's
     * made the request.
     */
    $groups = array();
    foreach( array_keys( RegDB::instance()->experiment_specific_groups()) as $g ) {
        if( RegDBAuth::instance()->canManageLDAPGroup( $g ))
            array_push( $groups, $g );
    }
    sort( $groups );

    $con = new RegDBHtml( 0, 0, 400, 65 );
    echo $con
        ->value       (   0,  3, '<b>Select group to manage</b>' )
        ->select_input( 160,  0, 'group', $groups, ( count( $groups ) ? $groups[0] : '' ), "groups_filter_input", 'javascript:apply_select_group(this);' )
        ->button      ( 157, 35, 'groups_filter_button', 'Refresh', 'refresh the table using this filter' )
        ->html();

    RegDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>