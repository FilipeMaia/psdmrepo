<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;

/*
 * This script will process a request for creating a new experiment
 * in the database.
 */
function report_error($msg) {
    print $msg;
    exit;
}

/* Process parameters and get user account name and a list of groups.
 */
if( !array_key_exists( 'user:uid', $_POST )) report_error( 'no user account information found in the request' );
$uid = $_POST['user:uid'];

$requested_groups = array();
foreach( array_keys( $_POST ) as $k )
    if( !strncasecmp( $k, 'gid:', 4 ))
        $requested_groups[$_POST[$k]] = True;

if( isset( $_POST['actionSuccess'] )) $actionSuccess = trim( $_POST['actionSuccess'] );

/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    /* Get special groups which we're going to deal with. All group operations
     * will be restricted to those only!
     */
    $restricted_groups = RegDB::instance()->experiment_specific_groups();

    /* Get all groups the account is member of. The account must exist.
     */
    $account = RegDB::instance()->find_user_account( $uid );
    if( is_null( $account ))
        report_error( RegDBAuth::reporErrorHtml('No such account known to LDAP server: '.$uid ));

    $registered_groups = array();
    foreach( $account['groups'] as $g )
    	$registered_groups[$g] = True;

    /* Go to LDAP and remove registered groups which aren't
     * in the request, and add those which are in the request but
     * not registered yet.
     * 
     * WARNING: Due to a nature of the POST data for the list of
     *          requested groups, the algorithm will first remove
     *          a user from all groups not mentioned in the input request,
     *          and then (on the second step) it will readd the user back
     *          to those groups. THIS NEEDS TO BE EITHER FIXED OR THE WHOLE
     *          FUNCTION BE COMPLETELLY REMOVED!!!!
     */
    foreach( array_keys( $registered_groups ) as $g ) {
    	if( !array_key_exists( $g, $requested_groups )) {
    		if( array_key_exists( $g, $restricted_groups )) {
    			if( RegDBAuth::instance()->canManageLDAPGroup( $g )) {
                    RegDB::instance()->remove_user_from_posix_group( $uid, $g );
    			}
    		}
    	}
    }
    foreach( array_keys( $requested_groups ) as $g ) {
    	if( !array_key_exists( $g, $registered_groups )) {
    		if( array_key_exists( $g, $restricted_groups )) {
    			if( RegDBAuth::instance()->canManageLDAPGroup( $g )) {
    			    RegDB::instance()->add_user_to_posix_group( $uid, $g );
    			}
    		}
    	}
    }
    
    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home'        ) header('Location: ../index.php');
        elseif ($actionSuccess == 'view_account') header('Location: ../index.php?action=view_account&uid='.$uid);
    }
    RegDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>