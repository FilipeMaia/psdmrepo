<!--
The page for creating a new free-form.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Create a new shift</title>
    </head>
    <body>
        <?php
        $now = new DateTime();
        $now_str = $now->format(DateTime::ISO8601);
        $now_str[10] = ' ';  // get rid of date-time separator 'T'
        ?>
        <h1>Create new free-form entry :</h1>
        <form action="ProcessCreateFFEntry.php" method="POST" style="margin-left:2em;">
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
                        <td align="right">
                            &nbsp;<b>Experiment</b>&nbsp;</td>
                        <td>
                            &nbsp;<select align="center" type="text" name="experiment_name" ><?php
                            require_once('LogBook.inc.php');
                            $logbook = new LogBook();
                            $experiments = $logbook->experiments()
                                or die("failed to find experiments" );
                            foreach( $experiments as $e)
                                echo '<option> '.$e->attr['name'].' </option>';
                            ?></select>&nbsp;
                        <td>
                            &nbsp;</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Relevance Time</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="32" type="text" name="relevance_time" value="<?php echo ' '.$now_str; ?>" />&nbsp;</td>
                        <td>
                            &nbsp;YYYY-MM-DD hh:mm:ss-zzzz</td>
                    </tr>
                    <tr>
                        <td align="right">
                            &nbsp;<b>Author</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="author" value="" />&nbsp;</td>
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
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
    </body>
</html>

