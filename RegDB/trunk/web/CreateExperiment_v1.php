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
            <table cellpadding="3" border="0" >
                <tbody>
                    <tr>
                        <td align="left" valign="top" style="color:#0071bc;">
                            <b>Name</b></td>
                        <td valign="top" >
                            <input align="left" size="24" type="text" name="experiment_name" value="" /></td></tr>
                    <tr>
                        <td align="left" style="color:#0071bc;">
                            <b>Instrument</b></td>
                        <td>
                            <select align="center" type="text" name="instrument_name" >
                            <?php foreach( $instrument_names as $i ) echo "<option> $i </option>"; ?>
                            </select></td></tr>
                    <tr>
                        <td></td>
                        <td><br></td>
                        <td></td></tr>
                    <tr>
                        <td align="left" style="color:#0071bc;">
                            <b>Begin Time</b></td>
                        <td>
                            <input align="left" size="20" type="text" name="begin_time" value="<?php echo $now_str; ?>" /></td>
                        <td>
                            YYYY-MM-DD hh:mm:ss-zzzz</td></tr>
                    <tr>
                        <td align="left" style="color:#0071bc;">
                            <b>End Time</b></td><td>
                            <input align="left" size="20" type="text" name="end_time" value="<?php echo $now_str; ?>" /></td>
                        <td>
                            YYYY-MM-DD hh:mm:ss-zzzz</td></tr>
                    <tr>
                        <td></td>
                        <td><br></td>
                        <td></td></tr>
                    <tr>
                        <td align="left" style="color:#0071bc;">
                            <b>POSIX Group</b></td>
                        <td>
                            <select align="center" type="text" name="group" >
                            <?php foreach( $posix_groups as $g ) echo "<option> $g </option>"; ?>
                            </select><td></tr>
                    <tr>
                        <td align="left" valign="top" style="color:#0071bc;">
                            <b>Leader</b></td>
                        <td valign="top" >
                            <input align="left" size="8" type="text" name="leader" value="<?php echo $_SERVER['WEBAUTH_USER']; ?>" /></td>
                        <td style="width:20em;">
                            UNIX account name of the leader. The account must be
                            a member of the selected POSIX group.</td></tr>
                </tbody>
            </table>
            <br>
            <table>
                <thead>
                    <th style="color:#0071bc;"><b>Contact Info</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <textarea name="contact" rows="3" cols="72"></textarea></td></tr>
                </tbody>
            </table>
            <br>
            <table>
                <thead>
                    <th style="color:#0071bc;"><b>Experiment Description</b></th>
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
