<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;
use LusiTime\LusiTime;

/*
 * This script will process requests for managing members of POSIX groups.
 * Depending on parameters to the script (see below), there may be three
 * kinds of requests:
 * - obtaining a list of members of a group
 * - including a user into a group and returning an updated list of members
 * - excluding a user from a group and returning an updated list of members
 *
 * Upon its successfull completion the script will always return a JSOB object
 * encoding a list of group members and a status of the operation. In case
 * of a (controlled and detected) failure only a status will be returned.
 * It's up to a caller to analyze the status value to determine if the list
 * of users is present in the returned object.
 */
header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

/* Package the error message into a JAON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error_and_return( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "status": {$status_encoded},
  "ResultSet": {
    "Status": {$status_encoded},
    "Message": {$msg_encoded}
  }
}
HERE;
    exit;
}

/* Analyze input parameters.
 */
if( isset( $_GET['group'] )) {
    $groupname = trim( $_GET['group'] );
    if( $groupname == '' )
        report_error_and_return( "group name can't be empty" );
} else {
    report_error_and_return( "no valid group name" );
}

$known_actions = array( 'include' => 0, 'exclude' => 1 );
if( isset( $_GET['action'] )) {
    $action = trim( $_GET['action'] );
    if( $action == '' ) {
        report_error_and_return( "action name can't be empty" );
    } else if( array_key_exists( $action, $known_actions )) {
    	if( isset( $_GET['uid'] )) {
            $uid = trim( $_GET['uid'] );
            if( $uid == '' )
                report_error_and_return( "UID can't be empty in the request" );
        } else {
            report_error_and_return( "no valid UID found among parameters of the request" );
        }
    } else {
        report_error_and_return( "unsupported action: '{$action}'" );
    }
}

/* This optional parameter can be used to specify if a simple JSON array
 * of triplets (uid,name,email) is to be returned instead of HTML
 * decorated entries.
 */
$simple = isset( $_GET['simple'] );

function member2json( $member ) {
    return json_encode(
        array (
            "uid"   => "<a href=\"javascript:view_account('".$member['uid']."')\">".$member['uid']."</a> ",
            "name"  => $member['gecos'],
            "email" => $member['email']
        )
    );
}
function member2json_simple( $member ) {

    return json_encode(
        array (
            "uid_link" => "<a href=\"javascript:view_account('".$member['uid']."')\">".$member['uid']."</a> ",
            "uid"      => $member['uid'],
            "name"     => substr( $member['gecos'], 0, 32 ),
            "email"    => $member['email']
        )
    );
}

/*
 * Return JSON objects with a list of groups.
 */
try {
    RegDB::instance()->begin();

    /* Make sure the group exists
     */
    if( !RegDB::instance()->is_known_posix_group( $groupname )) {
        report_error_and_return( "The group doen't exist" );
    }

    /* The following section is executed only for modification requests. Special
     * constrains will be also evalueatd before making any changes to the group.
     */
    if( isset( $action )) {

        /* Make sure the group can be managed in the local realm
         */
    	if( !array_key_exists( $groupname, RegDB::instance()->experiment_specific_groups())) {
            report_error_and_return( "The group can't be managed by this application" );
        }
    	
        /* Check for authorization.
         */
    	if( !RegDBAuth::instance()->canManageLDAPGroup( $groupname )) {
                report_error_and_return( 'You are not authorized to mamage the group members' );
        }

        /* Proceed to the requested operation
         */
        if(      $action == 'include' ) RegDB::instance()->add_user_to_posix_group     ( $uid, $groupname );
    	else if( $action == 'exclude' ) RegDB::instance()->remove_user_from_posix_group( $uid, $groupname );
    }
    $members = RegDB::instance()->posix_group_members( $groupname );

    $status_encoded  = json_encode( "success" );
    $updated_encoded = json_encode( LusiTime::now()->toStringShort());
    print <<< HERE
{
  "status":  {$status_encoded},
  "updated": {$updated_encoded},
  "ResultSet": {
    "Status": {$status_encoded},
    "Result": [
HERE;
    $first = true;
    foreach( $members as $m ) {
    	$entry = $simple ? member2json_simple( $m ) : member2json( $m );
        if( $first ) {
            $first = false;
            echo "\n".$entry;
        } else {
            echo ",\n".$entry;
        }
    	    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();

} catch (RegDBException $e) { report_error_and_return( $e->toHtml()); }
  catch (Exception      $e) { report_error_and_return( $e );}

?>
