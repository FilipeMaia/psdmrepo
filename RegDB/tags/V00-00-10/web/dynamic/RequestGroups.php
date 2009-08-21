<?php

require_once('RegDB/RegDB.inc.php');

function group2json( $group ) {
    $group_url = "<a href=\"javascript:view_group('".$group."')\">".$group.'</a>';
    return json_encode(
        array ( "group" => $group_url )
    );
}

/*
 * Return JSON objects with a list of groups.
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $groups = $regdb->posix_groups();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $groups as $g ) {
        if( $first ) {
            $first = false;
            echo "\n".group2json( $g );
        } else {
            echo ",\n".group2json( $g );
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
