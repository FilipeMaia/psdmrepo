<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying user accounts.
 */
if( !RegDBAuth::instance()->canRead()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to read the users info'));
    exit;
}

/*
 * The script has two parameters:
 * - a substring to search in UIDs or names of users. If the parameter is empty
 *   then all known users will be returned regardless of the scope.
 * - a scope of the search operation
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

/* This optional parameter can be used to specify if a simple JSON array
 * of triplets (uid,name,email) is to be returned instead of HTML
 * decorated entries.
 */
$simple = isset( $_GET['simple'] );


function account2json( $account ) {
    $groups_str = '';
    $first = true;
    foreach( $account['groups'] as $g ) {
        if( $first) $first = false;
        else $groups_str .= "<br>";
        $groups_str .= "<a href=\"javascript:view_group('".$g."')\">".$g."</a> ";
    }

    return json_encode(
        array (
            "uid"    => "<a href=\"javascript:view_account('".$account['uid']."')\">".$account['uid']."</a> ",
            "name"   => $account['gecos'],
            "email"  => $account['email'],
            "groups" => $groups_str
        )
    );
}

function account2json_simple( $account ) {

    return json_encode(
        array (
            "uid_link" => "<a href=\"javascript:view_account('".$account['uid']."')\">".$account['uid']."</a> ",
            "uid"      => $account['uid'],
            "name"     => substr( $account['gecos'], 0, 32 ),
            "email"    => $account['email']
        )
    );
}

/*
 * Return JSON objects with a list of groups.
 */
try {
    RegDB::instance()->begin();

    if( $string2search == '' ) {
    	$accounts = RegDB::instance()->user_accounts();
    } else {
    	$search_in_scope = array( "uid" => true, "gecos" => true );
    	if(      $scope == "uid"  ) $search_in_scope["gecos"] = false;
    	else if( $scope == "name" ) $search_in_scope["uid"]   = false;
    	$accounts = RegDB::instance()->find_user_accounts( $string2search, $search_in_scope );
    }

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "status": "success",
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $accounts as $a ) {
    	$entry = $simple ? account2json_simple( $a ) : account2json( $a );
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

} catch( RegDBException $e ) { print $e->toHtml(); }

?>
