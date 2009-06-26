<!--
The page for displaying values of parameters of an experiment in
either one specified run or in all runs of that experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title></title>
    </head>
    <body>
        <h1>List Values of Run Parameters :</h1>
        <form action="ProcessListRunParamValues.php" method="POST" style="margin-left:2em;">
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
                            &nbsp;<b>Run Number</b>&nbsp;</td>
                        <td>
                            &nbsp;<input align="left" size="16" type="text" name="num" value="" />&nbsp;</td>
                        <td>
                            &nbsp;1,2,3... [or leave blank to see all runs]</td>
                    </tr>
                    <tr>
                        <td align="right" style="width:6em;">
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
            <input type="submit" value="Submit" name="submit_button" /><br>
        </form>
    </body>
</html>

