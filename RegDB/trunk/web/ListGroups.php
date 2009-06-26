<!--
The page for creating displaying all POSIX groups.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Display POSIX Groups</title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <style>
    table_cell {
        width:72em;
    }
    </style>
    <body>
        <p id="title"><b>POSIX Groups</b></p>
        <div id="test_container">
            <table cellpadding="3" border="0" >
                <thead style="color:#0071bc;">
                    <th align="left">
                        <b>Name</b></th>
                    <th align="left">
                        <b># members</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <?php
                    require_once('RegDB/RegDB.inc.php');
                    try {
                        $regdb = new RegDB();
                        $regdb->begin();
                        $groups = $regdb->posix_groups();
                        foreach( $groups as $g ) {
                            $num_members = count( $regdb->posix_group_members( $g ));
                            echo <<< HERE
                    <tr>
                        <td align="left" valign="top">
                            <a href="DisplayGroup.php?name={$g}"><b>{$g}</b></a></td>
                        <td class="table_cell">
                            {$num_members}</td>
                    </tr>
HERE;
                        }
                        $regdb->commit();

                    } catch ( RegDBException $e ) {
                        print( $e->toHtml());
                    }
                    ?>
                </tbody>
            </table>
        </div>
    </body>
</html>

