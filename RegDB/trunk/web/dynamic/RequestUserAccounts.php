<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying user accounts.
 */
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
            "uid"    => $account['uid'],
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

    $accounts = $regdb->user_accounts();

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
