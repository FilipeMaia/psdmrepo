<?php

namespace FileMgr ;

require_once 'filemgr.inc.php' ;

/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit", "64M") ;

/*
 * The helper utility class to deal with iRODS Web Servive
 */
class FileMgrIfaceCtrlWs1 {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $CONN_PARAMS = array (
        'STANDARD' => array (
            'HOST'       => FILEMGR_ICWS_STANDARD_HOST ,
            'PATH'       => FILEMGR_ICWS_STANDARD_PATH ,
            'BASIC_AUTH' => FILEMGR_ICWS_STANDARD_BASIC_AUTH
        ) ,
        'MONITORING' => array (
            'HOST'       => FILEMGR_ICWS_MONITORING_HOST ,
            'PATH'       => FILEMGR_ICWS_MONITORING_PATH ,
            'BASIC_AUTH' => FILEMGR_ICWS_MONITORING_BASIC_AUTH
        )
    ) ;
    private static $instance = null ;

    /**
     * Singleton to simplify certain operations.
     *
     * @return FileMgrIfaceCtrlWs1
     */
    public static function instance ($service_name='STANDARD') {
        if (is_null(FileMgrIfaceCtrlWs1::$instance)) FileMgrIfaceCtrlWs1::$instance = array() ;
        if (!array_key_exists($service_name, FileMgrIfaceCtrlWs1::$instance))
            FileMgrIfaceCtrlWs1::$instance[$service_name] =
                new FileMgrIfaceCtrlWs1 (
                    FileMgrIfaceCtrlWs1::$CONN_PARAMS[$service_name]['HOST'] ,
                    FileMgrIfaceCtrlWs1::$CONN_PARAMS[$service_name]['PATH'] ,
                    FileMgrIfaceCtrlWs1::$CONN_PARAMS[$service_name]['BASIC_AUTH']) ;
        return FileMgrIfaceCtrlWs1::$instance[$service_name] ;
    }

    // --------------------------
    // --- INSTANCE INTERFACE ---
    // --------------------------

    private $host = null ;
    private $path = null ;
    private $basic_auth = null ;

    
    /**
     * Constructor
     *
     * @param {String} $host
     * @param {String} $path
     * @param {String} $basic_auth 
     */
    public function __construct ($host, $path, $basic_auth) {
        $this->host = $host ;
        $this->path = $path ;
        $this->basic_auth = $basic_auth ;
    }

    /**
     * Return the list of objects describing controller state for every host
     * in the system. The method shall return null if no controllers exist
     * in the system.
     *
     * @return array( [<system_info_object>[,...]] )
     */
    public function systems () {
        return $this->request_json('/system') ;
    }

    /**
     * Return an object describing thhe state of the specified controller.
     * The method should return null if no such object exists.
     *
     * @param Integer $id
     * @return <system_info_object>
     */
    public function system ($id) {
        return $this->request_json ("/system/{$id}") ;
    }

    /**
     * Return the list of experiments known to the translation service.
     * in the system. The method shall return null if no experiments exist.
     *
     * @return array( [<experiment_info_object>[,...]] )
     */
    public function experiments () {
        return $this->request_json ('/exp') ;
    }

    /**
     * Return a list of translation requests. The requests will be sorted by their identifiers.
     * If the latest_only flag is set to true then only latest requests for each run will be
     * returned. Note that "latest" means the most recent change in a request status, which
     * progresses over three timestamps:
     * 
     *   create - when the request was made
     *   begin  - when the translation began
     *   end    - when the translation finished/failed
     *
     * @param $instrument - name
     * @param $experiment - name
     * @param $latest_only - boolean flag
     * @return array
     */
    public function experiment_requests ($instrument, $experiment, $latest_only=false) {

    	$info = $this->request_json("/exp/{$instrument}/{$experiment}") ;

        if (!$latest_only) return $info ;


        /* First turn an array into a dictionary. The dictionary will have two
         * keys:
         *
         *   [run][id]
         */
        $run2id2request = array() ;
        foreach ($info as $req) {
            $run2id2request[$req->run][$req->id] = $req ;
        }

        function latest_timestamp ($req) {
            if (isset($req->stopped) && ($req->stopped != '')) return $req->stopped ;
            if (isset($req->started) && ($req->started != '')) return $req->started ;
            return $req->created ;
        }
        $out = array() ;

        $runs = array_keys($run2id2request) ;
        sort($runs) ;
        foreach ($runs as $run) {
            $latest = null ;
            $id = null ;
            foreach ($run2id2request[$run] as $req) {
                $time = latest_timestamp($req) ;
                if (is_null($latest) || ($time > $latest)) {
                    $latest = $time ;
                    $id = $req->id ;
                }
            }
            array_push ($out, $run2id2request[$run][$id]) ;
        }
        return $out ;
    }

    /**
     * Get the request by its identifier. Return null if nothing found.
     *
     * @param $id
     * @return object
     */
    public function request_by_id ($id) {
        return $this->request_json("/request/{$id}") ;
    }

    /** Request the contents of the specified log file by its URI.
     * 
     * NOTE: The log file URI is expected to include the full path, including
     *       Web service authentication method, the service name as awell as specific
     *       request with the log file path.
     *
     * @param $uri
     * @return string
     */
    public function log ($uri) {
    	return $this->request_plain_text($uri, true) ;
    }

    /**
     * 
     * @param String $instrument
     * @param String $experiment
     * @param Number $run
     * @throws FileMgrException
     */
    public function create_request ($instrument, $experiment, $run) {

    	$uri = $this->build_uri('/request', false) ;

    	$response = http_post_fields (
            $uri ,
            array (                             // the body of the request
                'instrument' => $instrument ,
                'experiment' => $experiment ,
                'runs'       => $run ,
                'force'      => 1
            ) ,
            null ,                              // no files submitted with ths request
            $this->opts() ,
            $info) ;

        if( $info['response_code'] != 200 ) {
            throw new FileMgrException (
                __METHOD__ ,
                "Web service request failed: {$uri}, eror code: ".$info['response_code']) ;
        }
        $response_parsed = http_parse_message($response) ;
        $str = ''.$response_parsed->body ;      // This is needed because the body is returned
                                                // as stdClass which can't be JSON decoded.
        return json_decode($str) ;
    }

    public function delete_request ($id) {

    	$uri = $this->build_uri('/request/'.$id, false) ;

    	$response = http_request (
            HTTP_METH_DELETE ,
            $uri ,
            null ,                              // no body for this request
            $this->opts() ,
            $info) ;

        if ($info['response_code'] != 200) {
            throw new FileMgrException(
                __METHOD__,
                "Web service request failed: {$uri}, eror code: ".$info['response_code']) ;
        }
    }

    public function set_request_priority ($id, $priority) {

    	$uri = $this->build_uri('/request/'.$id, false) ;

    	/* ATTENTION: Using this OO interface is the only way to make PUT to work
    	 * with the current implementation of PHP and HTTP module.
    	 */
        $http = new \HttpRequest($uri, \HttpRequest::METH_PUT) ;
        $http->setOptions    ($this->opts()) ;
        $http->setContentType("application/x-www-form-urlencoded; charset=utf-8") ;
        $http->addPutData    ("priority={$priority}") ;
        try {
            return json_decode($http->send()->getBody()) ;
        } catch (\HttpException $ex) {
            throw new FileMgrException (
                __METHOD__  ,
                "Web service request failed: {$uri}, due to: ".$ex) ;
        }
    }

    // --------------------------------
    // Helpers for making HTTP requests
    // --------------------------------

    private function build_uri ($uri_or_service, $is_uri) {
        return $is_uri ? $this->host.$uri_or_service : $this->host.$this->path.$uri_or_service ;
    }

    private function opts () {
    	return array (
            "timeout"      => 10,
    	    "httpauthtype" => HTTP_AUTH_BASIC,
    	    "httpauth"     => $this->basic_auth) ;
    }
    private function request_plain_text ($uri_or_service, $is_uri=false) {
    	return $this->request_('text', $uri_or_service, $is_uri) ;
    }
    private  function request_json ($uri_or_service, $is_uri=false) {
        return $this->request_('json', $uri_or_service, $is_uri) ;
    }
    private function request_ ($type, $uri_or_service, $is_uri) {

        $uri = $this->build_uri($uri_or_service, $is_uri) ;
    	$response = http_get($uri, $this->opts(), $info) ;

        if ($info['response_code'] != 200) {

            // Special case for 404: 'File Not Found'
            //
            if ($info['response_code'] == 404) return null ;

            throw new FileMgrException (
                __METHOD__ ,
                "Web service request failed: {$uri}, eror code: ".$info['response_code']) ;
        }
        $response_parsed = http_parse_message($response) ;

        // Promote result to the requested type.
        //
        switch ($type) {

        case 'text' :
            return $response_parsed->body ;

        case 'json' :
            $str = ''.$response_parsed->body ;  // This is needed because the body is returned
                                                // as stdClass which can't be JSON decoded.
            return json_decode($str) ;

        default :
            throw new FileMgrException (
                __METHOD__ ,
                "Internal implementation error. Unknown data type requested: {$type}, request uri: {$uri}") ;
        }
    }
}


/* -------------------------------
 * Here follows the unit test code
 * -------------------------------
 *

require_once 'filemgr/filemgr.inc.php' ;

function par2html ($name, $value) {
    return $name.': <b>'.$value.'</b><br>' ;
}

function dump_system ($ctrl, $info) {
    if (is_null($info)) return ;
    echo par2html('id',      $info->id) ;
    echo par2html('host',    $info->host) ;
    echo par2html('pid',     $info->pid) ;
    echo par2html('status',  $info->status) ;
    echo par2html('started', $info->started) ;
    echo par2html('stopped', $info->stopped) ;
    echo par2html('log',     $info->log) ;
    echo par2html('log_url', $info->log_url) ;
    if (isset($info) && ($info->log_url != '')) {
        echo '<h3>LogFile contents:</h3>' ;
        echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">'.$ctrl->log($info->log_url).'</div>' ;
    }
}

function dump_system_array ($ctrl, $info) {
    if (is_null($info)) return ;
    foreach ($info as $i) {
        dump_system($ctrl, $i) ;
        echo "<br>" ;
    }
}

function dump_request ($ctrl, $info) {
    if (is_null($info)) return ;
    echo par2html('id',         $info->id) ;
    echo par2html('instrument', $info->instrument) ;
    echo par2html('experiment', $info->experiment) ;
    echo par2html('run',        $info->run) ;
    echo par2html('status',     $info->status) ;
    echo par2html('created',    $info->created) ;
    echo par2html('started',    $info->started) ;
    echo par2html('stopped',    $info->stopped) ;
    echo par2html('priority',   $info->priority) ;
    echo 'xtc_files:' ;
    if (isset($info))
        foreach ($info->xtc_files as $f) echo ' <b>'.$f.'</b>' ;
    echo '<br>' ;
    echo 'hdf_files:' ;
    if (isset($info))
        foreach ($info->hdf_files as $f) echo ' <b>'.$f.'</b>' ;
    echo '<br>' ;
    echo par2html('log',       $info->log) ;
    echo par2html('log_url',   $info->log_url) ;
    echo par2html('url',       $info->url) ;

    //if (isset($info) && ($info->log_url != '')) {
    //    echo '<h3>LogFile contents:</h3>' ;
    //    echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">'.$ctrl->log($info->log_url).'</div>' ;
    //}
}

function dump_request_array ($ctrl, $info) {
    if (is_null($info)) return ;
    foreach($info as $i) {
        dump_request($ctrl, $i) ;
        echo "<br>" ;
    }
}

function dump_experiment ($ctrl, $info) {
    if (is_null($info)) return ;
    echo par2html('instrument', $info->instrument) ;
    echo par2html('experiment', $info->experiment) ;
    echo par2html('url',        $info->url) ;
    echo '<h3>Requests:</h3>' ;
    echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">' ;
    dump_request_array($ctrl, $ctrl->experiment_requests($info->instrument, $info->experiment)) ;
    echo '</div>' ;
}

function dump_experiment_array ($ctrl, $info) {
    if (is_null($info)) return ;
    foreach ($info as $i) {
        dump_experiment($ctrl, $i) ;
        echo "<br>" ;
    }
}

try {

    // Interface controllers
 
    $service = 'MONITORING' ;
    $ctrl = FileMgrIfaceCtrlWs1::instance($service) ;
    
    echo "<h1>Interface Controller: {$service}</h1>" ;

    echo "<h2>systems</h2>" ;
    dump_system_array ($ctrl, $ctrl->systems()) ;

    echo "<h2>system(203)</h2>" ;
    dump_system($ctrl, $ctrl->system(203)) ;

    echo "<h2>system(0)</h2>" ;
    dump_system($ctrl, $ctrl->system(0)) ;

    // Known experiments and relevant requests.

    echo "<h2>experiments()</h2>" ;
    dump_experiment_array($ctrl, $ctrl->experiments()) ;

} catch (\Exception $e) {
    echo $e ;
}

*
* ----------------
* End Of Unit test
* ----------------
*/
?>
