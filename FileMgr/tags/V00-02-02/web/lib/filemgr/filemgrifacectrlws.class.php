<?php

namespace FileMgr;

require_once( 'filemgr.inc.php' );

/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit","64M");

/*
 * The helper utility class to deal with iRODS Web Servive
 */
class FileMgrIfaceCtrlWs {

    /**
     * Return the list of objects describing controller state for every host
     * in the system. The method shall return null if no controllers exist
     * in the system.
     *
     * @return array( [<system_info_object>[,...]] )
     */
    public static function systems() {
    	$info = null;
        FileMgrIfaceCtrlWs::request_json( $info, '/system' );
        return $info;
    }

    /**
     * Return an object describing thhe state of the specified controller.
     * The method should return null if no such object exists.
     *
     * @param Integer $id
     * @return <system_info_object>
     */
    public static function system( $id ) {
    	$info = null;
        FileMgrIfaceCtrlWs::request_json( $info, '/system/'.$id );
        return $info;
    }

    /**
     * Return the list of experiments known to the translation service.
     * in the system. The method shall return null if no experiments exist.
     *
     * @return array( [<experiment_info_object>[,...]] )
     */
    public static function experiments() {
    	$info = null;
        FileMgrIfaceCtrlWs::request_json( $info, '/exp' );
        return $info;
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
    public static function  experiment_requests( $instrument, $experiment, $latest_only=false ) {

    	$info = null;
        FileMgrIfaceCtrlWs::request_json( $info, "/exp/{$instrument}/{$experiment}" );

        if( !$latest_only ) return $info;


        /* First turn an array into a dictionary. The dictionary will have two
         * keys:
         *
         *   [run][id]
         */
        $run2id2request = array();
        foreach( $info as $req ) {
            $run2id2request[$req->run][$req->id] = $req;
        }

        function latest_timestamp( $req ) {
            if( $req->stopped != '' ) return $req->stopped;
            if( $req->started != '' ) return $req->started;
            return $req->created;
        }
        $out = array();

        $runs = array_keys( $run2id2request );
        sort( $runs );
        foreach( $runs as $run ) {
            $latest = null;
            $id = null;
            foreach( $run2id2request[$run] as $req ) {
                $time = latest_timestamp( $req );
                if( is_null( $latest ) || ( $time > $latest )) {
                    $latest = $time;
                    $id = $req->id;
                }
            }
            array_push( $out, $run2id2request[$run][$id] );
        }
        return $out;
    }

    /**
     * Get the request by its identifier. Return null if nothing found.
     *
     * @param $id
     * @return object
     */
    public static function request_by_id( $id ) {
    	$info = null;
        FileMgrIfaceCtrlWs::request_json( $info, "/request/{$id}" );
    	return $info;
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
    public static function log( $uri ) {
    	$result = null;
    	FileMgrIfaceCtrlWs::request_plain_text( $result, $uri, true );
    	return $result;
    }

    // ----------
    // Operations
    // ----------

    public static function create_request( &$result, $instrument, $experiment, $run ) {

    	$uri = FileMgrIfaceCtrlWs::build_uri( '/request', false );

    	$response = http_post_fields (
    		$uri,
    		array(
				'instrument' => $instrument,
				'experiment' => $experiment,
				'runs'       => $run,
				'force'      => 1
			),
			null,							/* no files submitted with ths request */
			FileMgrIfaceCtrlWs::opts(),
			$info );

        if( $info['response_code'] != 200 ) {
            throw new FileMgrException(
                __METHOD__,
                "Web service request failed: {$uri}, eror code: ".$info['response_code'] );
        }
        $response_parsed = http_parse_message( $response );
        $str = ''.$response_parsed->body;  // This is needed because the body is returned
                                           // as stdClass which can't be JSON decoded.
        $result = json_decode( $str );
    }

    public static function delete_request( $id ) {

    	$uri = FileMgrIfaceCtrlWs::build_uri( '/request/'.$id, false );

    	$response = http_request (
    		HTTP_METH_DELETE,
    		$uri,
    		null,	// body
			FileMgrIfaceCtrlWs::opts(),
			$info );

        if( $info['response_code'] != 200 ) {
            throw new FileMgrException(
                __METHOD__,
                "Web service request failed: {$uri}, eror code: ".$info['response_code'] );
        }
    }

    public static function set_request_priority( &$result, $id, $priority ) {

    	$uri = FileMgrIfaceCtrlWs::build_uri( '/request/'.$id, false );

    	/* ATTENTION: Using this OO interface is the only way to make PUT to work
    	 * with the current implementation of PHP and HTTP module.
    	 */
		$http = new \HttpRequest( $uri, \HttpRequest::METH_PUT );
		$http->setOptions( FileMgrIfaceCtrlWs::opts());
		$http->setContentType( "application/x-www-form-urlencoded; charset=utf-8" );
		$http->addPutData( "priority={$priority}" );
		try {
    		$result = json_decode( $http->send()->getBody());
		} catch ( \HttpException $ex ) {
			throw new FileMgrException(
    			__METHOD__,
        		"Web service request failed: {$uri}, due to: ".$ex );
		}
    }

    // --------------------------------
    // Helpers for making HTTP requests
    // --------------------------------

    private static function build_uri( $uri_or_service, $is_uri ) {
        return FILEMGR_ICWS_HOST.( $is_uri ? '' : '/ws-auth/icws' ).$uri_or_service;
    	
    }

    private static function opts() {
    	return array(
            "timeout"      => 10,
    	    "httpauthtype" => HTTP_AUTH_BASIC,
    	    "httpauth"     => FILEMGR_ICWS_BASIC_AUTH
    	);
    }
    
    private static function request_plain_text( &$result, $uri_or_service, $is_uri=false ) {
    	FileMgrIfaceCtrlWs::request_( $result, 'text', $uri_or_service, $is_uri );
    }

    private static function request_json( &$result, $uri_or_service, $is_uri=false ) {
        FileMgrIfaceCtrlWs::request_( $result, 'json', $uri_or_service, $is_uri );
    }

    private static function request_( &$result, $type, $uri_or_service, $is_uri ) {

    	$uri = FileMgrIfaceCtrlWs::build_uri( $uri_or_service, $is_uri);
    	$response = http_get( $uri, FileMgrIfaceCtrlWs::opts(), $info );

        if( $info['response_code'] != 200 ) {
        	if( $info['response_code'] == 404 ) {
        		
        		// Special case for 404: 'File Not Found'
        		//
        		$result = null;
        		return;
            }
            throw new FileMgrException(
                __METHOD__,
                "Web service request failed: {$uri}, eror code: ".$info['response_code'] );
        }
        $response_parsed = http_parse_message( $response );

        // Promote result to the requested type.
        //
        if( $type == 'text' ) {
        	$result = $response_parsed->body;
        } else if( $type == 'json' ) {
            $str = ''.$response_parsed->body;  // This is needed because the body is returned
                                               // as stdClass which can't be JSON decoded.
            $result = json_decode( $str );
        } else {
        	throw new FileMgrException(
                __METHOD__,
                "Internal implementation error. Unknown data type requested: {$type}, request uri: {$uri}" );
        }
    }
}


/* -------------------------------
 * Here follows the unit test code
 * -------------------------------
 *

require_once( 'filemgr/filemgr.inc.php' );

function par2html( $name, $value ) {
	return $name.': <b>'.$value.'</b><br>';
}

function dump_system( $info ) {
	if( is_null( $info )) return;
	echo par2html( 'id',      $info->id );
	echo par2html( 'host',    $info->host );
	echo par2html( 'pid',     $info->pid );
	echo par2html( 'status',  $info->status );
	echo par2html( 'started', $info->started );
	echo par2html( 'stopped', $info->stopped );
	echo par2html( 'log',     $info->log );
	echo par2html( 'log_url', $info->log_url );
	if( isset( $info ) && ( $info->log_url != '' )) {
	    echo '<h2>LogFile contents:</h2>';
	    echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">'.FileMgrIfaceCtrlWs::log( $info->log_url ).'</div>';
    }
}

function dump_system_array( $info ) {
	if( is_null( $info )) return;
	foreach( $info as $i ) {
        dump_system( $i );
        echo "<br>";
    }
}


function dump_request( $info ) {
	if( is_null( $info )) return;
	echo par2html( 'id',         $info->id );
	echo par2html( 'instrument', $info->instrument );
	echo par2html( 'experiment', $info->experiment );
	echo par2html( 'run',        $info->run );
	echo par2html( 'status',     $info->status );
	echo par2html( 'created',    $info->created );
	echo par2html( 'started',    $info->started );
	echo par2html( 'stopped',    $info->stopped );
	echo par2html( 'priority',   $info->priority );
	echo 'xtc_files:';
	if( isset( $info ))
	    foreach( $info->xtc_files as $f ) echo ' <b>'.$f.'</b>';
	echo '<br>';
	echo 'hdf_files:';
	if( isset( $info ))
	    foreach( $info->hdf_files as $f ) echo ' <b>'.$f.'</b>';
	echo '<br>';
	echo par2html( 'log',        $info->log );
	echo par2html( 'log_url',    $info->log_url );
	echo par2html( 'url',        $info->url );

	//if( isset( $info ) && ( $info->log_url != '' )) {
	//    echo '<h2>LogFile contents:</h2>';
	//    echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">'.FileMgrIfaceCtrlWs::log( $info->log_url ).'</div>';
    //}
}

function dump_request_array( $info ) {
	if( is_null( $info )) return;
	foreach( $info as $i ) {
        dump_request( $i );
        echo "<br>";
    }
}

function dump_experiment( $info ) {
	if( is_null( $info )) return;
	echo par2html( 'instrument', $info->instrument );
	echo par2html( 'experiment', $info->experiment );
	echo par2html( 'url',        $info->url );
    echo '<h2>Requests:</h2>';
	echo '<div style="margin-left:20; width:800; height:200; overflow: auto;">';
	dump_request_array( FileMgrIfaceCtrlWs::experiment_requests( $info->instrument, $info->experiment ));
	echo '</div>';
}

function dump_experiment_array( $info ) {
	if( is_null( $info )) return;
	foreach( $info as $i ) {
        dump_experiment( $i );
        echo "<br>";
    }
}

try {

    // Interface controllers

	echo "<h1>systems</h1>";
    dump_system_array( FileMgrIfaceCtrlWs::systems());

    echo "<h1>system(203)</h1>";
    dump_system( FileMgrIfaceCtrlWs::system(203));

    echo "<h1>system(0)</h1>";
    dump_system( FileMgrIfaceCtrlWs::system(0));

    // Known experiments and relevant requests.

    echo "<h1>experiments()</h1>";
    dump_experiment_array( FileMgrIfaceCtrlWs::experiments());
	
	
	
} catch( FileMgrException $e ) {
	echo $e->toHtml();
}

*
* ----------------
* End Of Unit test
* ----------------
*/
?>
