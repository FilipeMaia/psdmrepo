<?php
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
// basic sequence with LDAP is connect, bind, search, interpret search
// result, close connection

echo "<h3>Find Posix groups from LDAP</h3>";

$ds = ldap_connect( "localhost" )
    or die( "failed to connect to the server" );

$r = ldap_bind( $ds )
    or die( "failed to bibd to the server" );

$sr = ldap_search( $ds, "ou=Group,dc=reg,o=slac", "cn=*" )
    or die( "search operation failed" );

echo "<br>Found: <b>".ldap_count_entries( $ds, $sr ) . "</b> groups";

$info = ldap_get_entries( $ds, $sr );
for( $i = 0; $i < $info["count"]; $i++ ) {
    echo "<br><hr>Name: <b>" . $info[$i]["cn"][0]."</b>";
    for( $j = 0; $j < $info[$i]["objectclass"]["count"]; $j++ )
        echo "<br>&nbsp;&nbsp;Type: <b>" . $info[$i]["objectclass"][$j]."</b>";
    for( $j = 0; $j < $info[$i]["gidnumber"]["count"]; $j++ )
        echo "<br>&nbsp;&nbsp;GID: <b>" . $info[$i]["gidnumber"][$j]."</b>";
    echo "<br>&nbsp;&nbsp;Members: ";
    for( $j = 0; $j < $info[$i]["memberuid"]["count"]; $j++ ) {
        $full_name = "";
        $passwd = posix_getpwnam($info[$i]["memberuid"][$j]);
        if( $passwd ) $full_name = "(".$passwd["gecos"].")";
        echo "<br>&nbsp;&nbsp;&nbsp;&nbsp;<b>" . $info[$i]["memberuid"][$j]."</b>&nbsp;".$full_name;
    }
    echo "<br>";
}
ldap_close( $ds );
?>
