<?php

/*
 * This script is used by autocompletion algorithms to find user
 * accounts containing the specified substring.
 */
require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBException;

$string2search = trim( $_GET[ 'term' ] );
if( !isset( $string2search )) die( 'missing or empty mandatory parameter' );

try {
    RegDB::instance()->begin();

    $search_in_scope = array( "uid" => true, "gecos" => true );
    $accounts = RegDB::instance()->find_user_accounts( $string2search, $search_in_scope );

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
[
HERE;
    $first = true;
    foreach( $accounts as $a ) {
    	$entry = json_encode( '{ name: "'.$a['gecos'].'", uid: "'.$a['uid'].'", email: "'.$a['email'].'" }' );
        if( $first ) {
            $first = false;
            echo "\n".$entry;
        } else {
            echo ",\n".$entry;
        }
    }
    print <<< HERE

]
HERE;

    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>
