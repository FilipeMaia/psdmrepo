<?php
require_once('RegDB.inc.php');

try {
    $regdb = new RegDB();
    $regdb->begin();
    $instrument_names = $regdb->instrument_names();
    $posix_groups = $regdb->posix_groups();
?>
<!--
The page for creating a new experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Create Experiment</title>
    </head>
    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />
    <body>
        <?php
        $now = new DateTime();
        $now_str = $now->format(DateTime::ISO8601);
        $now_str[10] = ' ';  // get rid of date-time separator 'T'
        ?>
        <p id="title"><b>Create Experiment</b></p>
        <div id="test_container">
        <form enctype="multipart/form-data" action="ProcessCreateExperiment.php" method="POST" style="margin-left:2em;">

            <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                <tbody>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                        <td align="center" width="65%"><span class="table_header_cell"><b>Description</b></span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Experiment</b></span></td>
                                <td width="60%"><input type="text" name="experiment_name" value=""/></td>
                            </table>
                        </td>
                        <td align="left" width="65%" rowspan="5">
                            <textarea style="width:100%; height:100%;" name="description"></textarea></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Instrument</b></span></td>
                                <td width="60%">
                                    <select align="center" type="text" name="instrument_name" >
                                    <?php foreach( $instrument_names as $i ) echo "<option> $i </option>"; ?>
                                    </select></td>
                            </table>
                        </td>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Begin Time</b></span></td>
                                <td width="60%"><input type="text" name="begin_time" value="<?php echo $now_str; ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>End Time</b></span></td>
                                <td width="60%"><input type="text" name="end_time" value="<?php echo $now_str; ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%"><span>&nbsp;</span></td>
                        <td align="center" width="65%"><span class="table_header_cell"><b>Contact Info</b></span></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>POSIX Group</b></span></td>
                                <td width="60%">
                                    <select align="center" type="text" name="group" >
                                    <?php foreach( $posix_groups as $g ) echo "<option> $g </option>"; ?>
                                    </select></td>
                            </table>
                        </td>
                        <td align="left" width="65%" rowspan="2">
                            <textarea style="width:100%; height:100%;" name="contact"></textarea></td>
                    </tr>
                    <tr>
                        <td align="left" width="35%">
                            <table border="0" width="100%" align="left">
                                <td width="40%"><span class="table_header_cell"><b>Leader</b></span></td>
                                <td width="60%"><input type="text" name="leader" value="<?php echo $_SERVER['WEBAUTH_USER']; ?>"/></td>
                            </table>
                        </td>
                    </tr>
                    <tr>
                    </tr>
                </tbody>
            </table>
            <br>
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
