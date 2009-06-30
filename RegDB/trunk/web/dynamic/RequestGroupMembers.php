<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying members of a POSIX group.
 */
if( isset( $_GET['name'] )) {
    $name = trim( $_GET['name'] );
    if( $name == '' )
        die( "group name can't be empty" );
} else
    die( "no valid group name" );

function member2json( $member ) {
    return json_encode(
        array (
            "uid"   => $member['uid'],
            "name"  => $member['gecos'],
            "email" => $member['email']
        )
    );
}

/*
 * Return JSON objects with a list of groups.
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $members = $regdb->posix_group_members( $name );

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $members as $m ) {
        if( $first ) {
            $first = false;
            echo "\n".member2json( $m );
        } else {
            echo ",\n".member2json( $m );
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
