<?php

require_once('RegDB/RegDB.inc.php');

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

/*
 * Return JSON objects with a list of groups.
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    if( $string2search == '' ) {
    	$accounts = $regdb->user_accounts();
    } else {
    	$search_in_scope = array( "uid" => true, "gecos" => true );
    	if(      $scope == "uid"  ) $search_in_scope["gecos"] = false;
    	else if( $scope == "name" ) $search_in_scope["uid"]   = false;
    	$accounts = $regdb->find_user_accounts( $string2search, $search_in_scope );
    }

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $accounts as $a ) {
        if( $first ) {
            $first = false;
            echo "\n".account2json( $a );
        } else {
            echo ",\n".account2json( $a );
        }
    }
    print <<< HERE
 ] } }
HERE;

    $regdb->commit();

} catch( regdbException $e ) {
    print $e->toHtml();
}

?>
