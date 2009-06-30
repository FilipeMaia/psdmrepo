<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>File Uploading Test</title>
    </head>
    <body>
        <h1>File Uploading Test</h1>
        <!--
        next comes the form, you must set the enctype to "multipart/form-data"
        and use an input type "file"
        -->
        <form name="newad" method="POST" enctype="multipart/form-data" action="ProcessUploadFile.php">
            <table>
                <tr>
                    <td>
                        <input type="hidden" name="MAX_FILE_SIZE" value="1000000">
                        <input type="file" name="file">
                    </td>
                </tr>
                <tr>
                    <td>
                        <input type="submit" name="upload_file" value="Upload">
                    </td>
                </tr>
            </table>
        </form>
    </body>
</html>
