<!--
The page for creating a new run.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title></title>
    </head>
    <body>
        <?php
        $now = new DateTime();
        $now_str = $now->format(DateTime::ISO8601);
        $now_str[10] = ' ';  // get rid of date-time separator 'T'
        ?>
        <h1>Add new run :</h1>
        <form action="ProcessCreateRun.php" method="POST" style="margin-left:2em;">
            <table cellpadding="3"  border="0" >
                <thead style="color:#0071bc;">
                    <th align="right">
                        &nbsp;<b>Attribute</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Value</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Format</b>&nbsp;</th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <tr>
                        <td align="right" style="width:6em;">
                            &nbsp;<b>Number</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="num" value=" <autogenerate>" />&nbsp;</td>
                        <td>
                            &nbsp;1,2,3... [or leave default to autogenerate]</td>
                    </tr>
                    <tr>
                        <td align="right" style="width:6em;">
                            &nbsp;<b>Experiment</b>&nbsp;</td>
                        <td>
                            &nbsp;<select align="center" type="text" name="experiment_name" ><?php
                            require_once('LogBook.inc.php');
                            $host     = "localhost";
                            $user     = "gapon";
                            $password = "";
                            $database = "logbook";
                            $logbook = new LogBook( $host, $user, $password, $database );
                            $experiments = $logbook->experiments()
                                or die("failed to find experiments" );
                            foreach( $experiments as $e)
                                echo '<option> '.$e->attr['name'].' </option>';
                            ?></select>&nbsp;
                        <td>
                            &nbsp;</td>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Begin Time</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="32" type="text" name="begin_time" value="<?php echo ' '.$now_str; ?>" />&nbsp;</td>
                        <td>
                            &nbsp;YYYY-MM-DD hh:mm:ss-zzzz</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>End Time</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="32" type="text" name="end_time" value=" <unknown>" />&nbsp;</td>
                        <td>
                            &nbsp;YYYY-MM-DD hh:mm:ss-zzzz [optional] </td>
                    </tr>
                </tbody>
            </table>
            <br>
            <br>
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
    </body>
</html>

