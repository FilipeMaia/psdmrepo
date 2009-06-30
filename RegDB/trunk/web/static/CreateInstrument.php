<?php
require_once('RegDB/RegDB.inc.php');

try {
    $regdb = new RegDB();
    $regdb->begin();
?>
<!--
The page for creating a new instrument.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Create Instrument</title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <p id="title"><b>Create Instrument</b></p>
        <div id="test_container">
        <form enctype="multipart/form-data" action="ProcessCreateInstrument.php" method="POST" style="margin-left:2em;">
            <table cellpadding="3" border="0" >
                <tbody>
                    <tr>
                        <td align="left" valign="top" style="color:#0071bc;">
                            <b>Name</b></td>
                        <td valign="top" >
                            <input align="left" size="24" type="text" name="instrument_name" value="" /></td></tr>
                    <tr>
                        <td></td>
                        <td><br></td>
                        <td></td></tr>
                </tbody>
            </table>
            <br>
            <table>
                <thead>
                    <th style="color:#0071bc;"><b>Instrument Description</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <textarea name="description" rows="12" cols="72"></textarea></td></tr>
                </tbody>
            </table>
            <br>
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
        </div>
    </body>
</html>
<?php
    $regdb->commit();

} catch( RegDBException $e ) {
    echo $e->toHtml();
}
?>
