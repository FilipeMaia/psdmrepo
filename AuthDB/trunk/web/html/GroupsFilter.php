<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDBException;

use RegDB\RegDB;
use RegDB\RegDBHtml;

/*
 * This script will generate a module with input elements for the filter form
 * which is used as a filter for groups the current user is authorized to manage.
 */

/* Proceed with the operation
 */
try {
    $regdb = new RegDB;
    $regdb->begin();

    /* The list of groups will depend on privileges of a user who's
     * made the request.
     */
    $groups = array();
    foreach( array_keys( $regdb->experiment_specific_groups()) as $g ) {
        if( RegDBAuth::instance()->canManageLDAPGroup( $g ))
            array_push( $groups, $g );
    }
    sort( $groups );
    
    /* Proceed to the operation
     */
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 400, 65 );
    echo $con
        ->value       (   0,  3, '<b>Select group to manage</b>' )
        ->select_input( 160,  0, 'group', $groups, ( count( $groups ) ? $groups[0] : '' ), "groups_filter_input", 'javascript:apply_select_group(this);' )
        ->button      ( 157, 35, 'groups_filter_button', 'Refresh', 'refresh the table using this filter' )
        ->html();

    $regdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>