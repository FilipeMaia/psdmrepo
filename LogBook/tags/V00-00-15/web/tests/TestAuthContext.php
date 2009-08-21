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
        echo 'REMOTE_USER: '.$_SERVER['REMOTE_USER'].'<br>';
        echo 'AUTH_TYPE: '.$_SERVER['AUTH_TYPE'].'<br>';
        //echo 'Server:<br>';
        //print_r( $_SERVER );
        //echo 'Session:<br>';
        echo 'SERVER:<br>';
        print_r( $_SERVER );
        ?>
    </body>
</html>
