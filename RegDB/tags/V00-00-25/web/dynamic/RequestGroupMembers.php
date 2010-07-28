<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying members of a POSIX group.
 */

if( !RegDBAuth::instance()->canRead()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to read the group members info'));
    exit;
}

if( isset( $_GET['name'] )) {
    $name = trim( $_GET['name'] );
    if( $name == '' )
        die( "group name can't be empty" );
} else {
    die( "no valid group name" );
}

/* This optional parameter can be used to specify if a simple JSON array
 * of triplets (uid,name,email) is to be returned instead of HTML
 * decorated entries.
 */
$simple = isset( $_GET['simple'] );


function member2json( $member ) {
    return json_encode(
        array (
            "uid"   => "<a href=\"javascript:view_account('".$member['uid']."')\">".$member['uid']."</a> ",//$member['uid'],
            "name"  => $member['gecos'],
            "email" => $member['email']
        )
    );
}
function member2json_simple( $member ) {

    return json_encode(
        array (
            "uid"   => $member['uid'],
            "name"  => substr( $member['gecos'], 0, 32 ),
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

    $regdb->commit();

} catch( regdbException $e ) {
    print $e->toHtml();
}

?>
