<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>LogBook Login Page</title>
    </head>
    <body>
        <?php
        /* This context is provided by WebAuth authentication
         */
        $user    = $_SERVER['WEBAUTH_USER'];
        $created = (int)$_SERVER['WEBAUTH_TOKEN_CREATION'];
        $now     = mktime();
        $expired = (int)$_SERVER['WEBAUTH_TOKEN_EXPIRATION'];

        $seconds = $expired - $now;
        $hours_left   = (int)($seconds / 3600);
        $minutes_left = (int)((int)($seconds % 3600) / 60);
        $seconds_left = (int)((int)($seconds % 3600) % 60);
        ?>
        <h1 style="margin-left:1em">WebAuth Authorization Context</h1>
        <table style="margin-left:4em">
            <tbody>
                <tr>
                    <td><b>&nbsp;Loged as : &nbsp;</b></td>
                    <td> <?php echo $user; ?> </td>
                </tr>
                <tr>
                    <td><b>&nbsp;WebAuth token creation time : &nbsp;</b></td>
                    <td> <?php echo $created; ?> </td>
                </tr>
                <tr>
                    <td><b>&nbsp;Current time : &nbsp;</b></td>
                    <td> <?php echo $now; ?> </td>
                </tr>
                <tr>
                    <td><b>&nbsp;WebAuth token expiration time : &nbsp;</b></td>
                    <td> <?php echo $expired; ?> </td>
                </tr>
                <tr>
                    <td><b>&nbsp;Token will expire in : &nbsp;</b></td>
                    <td> <?php echo "{$hours_left} hours, {$minutes_left} minutes, {$seconds_left}"; ?> </td>
                </tr>
            </tbody>
        </table>
        <br>
        <div style="margin-left:3em">
        Use browser's <b>REFRESH</b> button to seee how much time is left before
        token's expiration.
        </div>
    </body>
</html>
