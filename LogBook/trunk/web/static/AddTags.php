<!--
The page for adding tags to an existing free-form.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Add Tags to Free-Form Entry</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <?php
        $now = new DateTime();
        $now_str = $now->format(DateTime::ISO8601);
        $now_str[10] = ' ';  // get rid of date-time separator 'T'
        ?>
        <p id="title"><b>Add Tags to Free-Form Entry</b></p>
        <form enctype="multipart/form-data" action="ProcessAddTags.php" method="POST" style="margin-left:2em;">
            <table cellpadding="3" border="0" >
                <thead style="color:#0071bc;">
                    <th align="right">
                        &nbsp;<b>Attribute</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Value</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Notes</b>&nbsp;</th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Entry Id</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="entry_id" value="" />&nbsp;</td>
                        <td>
                            &nbsp;Leave blank to assume the last entry of the experiment</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Experiment</b>&nbsp;</td>
                        <td>
                            &nbsp;<select align="center" type="text" name="experiment_name" ><?php
                            require_once('LogBook/LogBook.inc.php');
                            $logbook = new LogBook();
                            $logbook->begin();
                            $experiments = $logbook->experiments()
                                or die("failed to find experiments" );
                            foreach( $experiments as $e)
                                echo '<option> '.$e->attr['name'].' </option>';
                            $logbook->commit();
                            ?></select>&nbsp;
                        <td>
                            &nbsp;</td>
                    </tr>
                </tbody>
            </table>
            <br>
            <br>
            <table ellpadding="3" border="0" >
                <thead style="color:#0071bc;">
                    <th align="left">
                        &nbsp;<b>Tag</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Value (optional)</b>&nbsp;</th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<input align="left" size="16" type="text" name="tag_name1" value="" />&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="24" type="text" name="tag_value1" value="" />&nbsp;</td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<input align="left" size="16" type="text" name="tag_name2" value="" />&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="24" type="text" name="tag_value2" value="" />&nbsp;</td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<input align="left" size="16" type="text" name="tag_name3" value="" />&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="24" type="text" name="tag_value3" value="" />&nbsp;</td>
                    </tr>
                </tbody>
            </table>
            <br>
            <br>
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
    </body>
</html>

