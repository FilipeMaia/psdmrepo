<!--
The page for creating a new free-form.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Create Free-form Entry</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <?php
        $now = new DateTime();
        $now_str = $now->format(DateTime::ISO8601);
        $now_str[10] = ' ';  // get rid of date-time separator 'T'
        ?>
        <p id="title"><b>Create Free-Form Entry</b></p>
        <form enctype="multipart/form-data" action="ProcessCreateFFEntry.php" method="POST" style="margin-left:2em;">
            <table cellpadding="3" border="0" >
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
                        <td align="right">
                            &nbsp;<b>Parent Entry Id</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="parent_entry_id" value="" />&nbsp;</td>
                        <td>
                            &nbsp;Leave blank to start new discussion</td>
                    </tr>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
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
                    <tr>
                        <td align="right">
                            &nbsp;<b>Shift Id</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="shift_id" value="" />&nbsp;</td>
                        <td>
                            &nbsp;[optional]</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Run Number</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="runnum" value="" />&nbsp;</td>
                        <td>
                            &nbsp;[optional]</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Relevance Time</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="32" type="text" name="relevance_time" value="<?php echo ' '.$now_str; ?>" />&nbsp;</td>
                        <td>
                            &nbsp;YYYY-MM-DD hh:mm:ss-zzzz [optional]</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td><br></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Author</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="author" value=" <?php echo $_SERVER['WEBAUTH_USER']; ?> " />&nbsp;</td>
                        <td>
                            &nbsp;UNIX account name</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Content Type</b>&nbsp;</td>
                        <td>
                            &nbsp;<select align="center" type="text" name="content_type" >
                            <option> TEXT </option>
                            <option> HTML </option>
                            </select>&nbsp;
                        <td>
                            &nbsp;</td>
                    </tr>
                </tbody>
            </table>
            <br>
            <textarea name="content" rows="12" cols="72"></textarea>
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
            <table ellpadding="3" border="0" >
                <thead style="color:#0071bc;">
                    <th align="left">
                        &nbsp;<b>#</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>File to attach</b>&nbsp;</th>
                    <th align="left">
                        &nbsp;<b>Description (optional)</b>&nbsp;</th>
                </thead>
                <tbody>
                    <tr>
                        <td style="width:4em;"><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<b>1</b>&nbsp;</td>
                        <td>
                            &nbsp;<input type="hidden" name="MAX_FILE_SIZE" value="1000000">
                            <input type="file" name="file1">&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="48" type="text" name="description1" value="" />&nbsp;</td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<b>2</b>&nbsp;</td>
                        <td>
                            &nbsp;<input type="hidden" name="MAX_FILE_SIZE" value="1000000">
                            <input type="file" name="file2">&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="48" type="text" name="description2" value="" />&nbsp;</td>
                    </tr>
                    <tr>
                        <td align="left">
                            &nbsp;<b>3</b>&nbsp;</td>
                        <td>
                            &nbsp;<input type="hidden" name="MAX_FILE_SIZE" value="1000000">
                            <input type="file" name="file3">&nbsp;</td>
                        <td align="left">
                            &nbsp;<input align="left" size="48" type="text" name="description3" value="" />&nbsp;</td>
                    </tr>
                </tbody>
            </table>
            <br>
            <br>
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
    </body>
</html>

