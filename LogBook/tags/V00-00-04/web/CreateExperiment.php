<!--
To change this template, choose Tools | Templates
and open the template in the editor.
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
        <h1>Register new experiment :</h1>
        <form action="ProcessCreateExperiment.php" method="POST" style="margin-left:2em;">
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
                        <td><hr></td></tr>
                    <tr>
                        <td align="right" style="width:6em;">
                            &nbsp;<b>Name</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="48" type="text" name="name" value="" />&nbsp;</td>
                        <td>
                            &nbsp;Max.Len. 255</td>
                    </tr>
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
                            &nbsp;<input align="left" size="32" type="text" name="end_time" value="" />&nbsp;</td>
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
