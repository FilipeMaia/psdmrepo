<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once( 'filemgr/filemgr.inc.php' );

use FileMgr\RestRequest;

function list_files($path) {

    $resource = "/files{$path}";
    $request = new RestRequest($resource, 'GET');
    $request->execute();

    print '<pre>'.print_r($request->getResponseInfo(), true).'</pre>';
    print '<pre>'.print_r($request, true).'</pre>';

    return json_decode($request->getResponseBody());
}
function delete_file($path, $replica=0) {

    $resource = "/files{$path}";
    $request = new RestRequest($resource, 'DELETE', array('replica' => $replica ));
    $request->execute();

    print '<pre>'.print_r($request->getResponseInfo(), true).'</pre>';
    print '<pre>'.print_r($request, true).'</pre>';

    return json_decode($request->getResponseBody());
}

function restore_file($path) {

    $resource = "/files{$path}";
    $request = new RestRequest(
        $resource,
        'POST',
        array(
            'src_resource' => 'hpss-resc',
            'dst_resource' => 'lustre-resc' ),
        true  /* to package parameters into the POST body */
    );
    $request->execute();

    print '<pre>'.print_r($request->getResponseInfo(), true).'</pre>';
    print '<pre>'.print_r($request, true).'</pre>';

    return json_decode($request->getResponseBody());
}

function file_restore_queue() {
    $resource = "/queue";
    $request = new RestRequest($resource, 'GET');
    $request->execute();

    print '<pre>'.print_r($request->getResponseInfo(), true).'</pre>';
    print '<pre>'.print_r($request, true).'</pre>';

    return json_decode($request->getResponseBody());
}

try {

//    $catalog = '/psdm-zone/psdm/XPP/xppdaq12/xtc';
//    $files = list_files($catalog);
//    echo '<pre>'.print_r( $files, true).'</pre>'; 

//    $file = '/psdm-zone/psdm/XPP/xppdaq12/hdf5/xppdaq12-r0001.h5';
//    $replica = 2;
//    delete_file($file, $replica);

    $requests = file_restore_queue();
    echo '<pre>'.print_r( $requests, true).'</pre>'; 

//    $file = '/psdm-zone/psdm/CXI/cxi80410/xtc/e55-r0699-s00-c01.xtc';
//    $request = restore_file($file);
//    echo '<pre>'.print_r( $request, true).'</pre>'; 
//
//    $requests = file_restore_queue();
//    echo '<pre>'.print_r( $requests, true).'</pre>'; 

} catch(Exception $e) { 
    print 'Failed<br>';
    print $e;
}

?>
