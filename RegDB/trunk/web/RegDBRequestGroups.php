<?php

require_once('RegDB.inc.php');

function group2json( $group, $num_members ) {
    $group_url = "<a href=\"javascript:view_group('".$group."')\">".$group.'</a>';
    return json_encode(
        array (
            "group"   => $group_url,
            "members" => $num_members
        )
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
        $members = $regdb->posix_group_members( $g );
        if( $first ) {
            $first = false;
            echo "\n".group2json( $g, count( $members ));
        } else {
            echo ",\n".group2json( $g, count( $members ));
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
