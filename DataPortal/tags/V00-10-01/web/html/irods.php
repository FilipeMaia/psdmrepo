<?php
require_once 'filemgr/filemgr.inc.php';
use \FileMgr\FileMgrIrodsWs;

foreach(FileMgrIrodsWs::all_runs('AMO','amo74213','xtc') as $run) {
    echo "<br><b>run</b>: {$run->run}<br>";
    echo '<pre>'.print_r($run, true).'</pre>';
    echo '<br>';
}
?>
