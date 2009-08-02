<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for displaying members of a POSIX group.
 */
if( isset( $_GET['name'] )) {
    $name = trim( $_GET['name'] );
    if( $name == '' )
        die( "POSIX group name can't be empty" );
} else
    die( "no valid POSIX group name" );

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $members = $regdb->posix_group_members( $name )
        or die( "no such instrument" );
?>
<!--
The page for isplaying parameters of amembers of a POSIX group.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>POSIX Group: <?php echo $name; ?></title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>POSIX Group:</b>&nbsp;<?php echo $name; ?></p>
        <div id="test_container">
            <table cellpadding="0" border="0" >
                <thead style="color:#0071bc;">
                    <th align="left" style="width:6em;">
                        <b>UID</b></th>
                    <th align="left" style="width:10em;">
                        <b>Full Name</b></th>
                    <th align="left">
                        <b>E-mail Address</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <?php
                    foreach( $members as $member ) {
                        echo <<< HERE
                    <tr>
                        <td align="left" valign="top">
                            <b>{$member['uid']}</b></td>
                        <td valign="top">
                            {$member['gecos']}</td>
                        <td valign="top">
                            {$member['email']}</td>
                    </tr>
HERE;
                    }
                    ?>
                </tbody>
            </table>
        </div>
    </body>
</html>
<?php

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>