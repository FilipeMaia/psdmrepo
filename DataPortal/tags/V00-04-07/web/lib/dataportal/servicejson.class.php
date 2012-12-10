<?php

namespace DataPortal ;

require_once 'dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime;

class ServiceJSON {

    public static function run_handler ($method, $body, $options=array()) {
        try {
            $svc = new ServiceJSON ($method, $options) ;
            $body($svc) ;
        } catch( \Exception $e ) {
            ServiceJSON::report_error (
                $e.'<pre>'.print_r($e->getTrace(), true).'</pre>' ,
                array () ,
                $options
            ) ;
        }
    }

    // ----------------
    //   Data members
    // ----------------

    private $method      = null ;
    private $method_var  = null ;
    private $options     = null ;
    private $authdb      = null ;
    private $regdb       = null ;
    private $logbook     = null ;
    private $logbookauth = null ;
    private $configdb    = null ;
    private $irodsdb     = null ;
    private $neocaptar   = null ;
    private $irep        = null ;
    private $exptimemon  = null ;
    private $sysmon      = null ;

    public function __construct ($method, $options=array()) {
        $this->method = strtoupper(trim($method)) ;
        switch ($this->method) {
            case 'GET':  $this->method_var = $_GET ;  break ;
            case 'POST': $this->method_var = $_POST ; break ;
            default:
                throw new DataPortalException (
                    __CLASS__.'::'.__METHOD__, "illegal method: {$this->method}") ;
        }
        $this->options = $options ;
    }

    public final function run () {
        try {
            $this->handler ($this) ;
        } catch( \Exception $e ) {
            ServiceJSON::report_error (
                $e.'<pre>'.print_r($e->getTrace(), true).'</pre>' ,
                array () ,
                $this->options
            ) ;
        }
    }

    /**
     * User defined function to be overloaded if using object-oriented approach.
     */
    protected function handler ($SVC) {}

    /**
     * The method accepts a numerioc error code on its input and returns back
     * a text with human readable interpretation (if available) of the code.
     *
     * @param number $errcode
     * @return string
     */
    private static function upload_err2string ($errcode) {
        switch( $errcode ) {
            case UPLOAD_ERR_OK:         return "There is no error, the file uploaded with success." ;
            case UPLOAD_ERR_INI_SIZE:   return "The uploaded file exceeds the maximum of ".get_ini("upload_max_filesize")." in this Web server configuration." ;
            case UPLOAD_ERR_FORM_SIZE:  return "The uploaded file exceeds the maximum of ".$_POST["MAX_FILE_SIZE"]." that was specified in the sender's HTML form." ;
            case UPLOAD_ERR_PARTIAL:    return "The uploaded file was only partially uploaded." ;
            case UPLOAD_ERR_NO_FILE:    return "No file was uploaded." ;
            case UPLOAD_ERR_NO_TMP_DIR: return "Missing a temporary folder in this Web server installation." ;
            case UPLOAD_ERR_CANT_WRITE: return "Failed to write file to disk at this Web server installation." ;
            case UPLOAD_ERR_EXTENSION:  return "A PHP extension stopped the file upload." ;
        }
        return "Unknown error code: {$errorcode}" ;
    }

    // ----------------------
    //   Parameters parsers
    // ----------------------

    private function parse ($name, $required=true, $has_value=true, $allow_empty_value=false) {
        $name = trim($name) ;
        if (isset($this->method_var[$name])) {
            if ($has_value) {
                $val = trim($this->method_var[$name]) ;
                if ($allow_empty_value || ($val !== '')) return array (true, $val) ;
                throw new DataPortalException (
                    __CLASS__.'::'.__METHOD__, "parameter '{$name}' requires a non-empty value") ;
            }
            return array (true, null) ;
        } else {
            if (!$required) return array (false, null) ;
            throw new DataPortalException (
                __CLASS__.'::'.__METHOD__, "required parameter '{$name}' is missing") ;
            
        }
    }

    public function required_str ($name) {
        $result = $this->parse ($name, true, true, true) ;
        return $result[1] ;
    }

    public function optional_str ($name, $default) {
        $result = $this->parse ($name, false, true, true) ;
        if ($result[0]) return $result[1] ;
        return $default ;
    }

    public function required_int ($name) {
        $result = $this->parse ($name, true, true) ;
        return intval ($result[1]) ;
    }

    public function optional_int ($name, $default) {
        $result = $this->parse ($name, false, true, true) ;
        if ($result[0] && ($result[1] != '')) return intval ($result[1]) ;
        return $default ;
    }

    public function required_bool ($name) {
        $result = $this->parse ($name, true, true) ;
        $val = strtolower($result[1]) ;
        switch ($val) {
            case 'false' : return false ;
            case 'true'  : return true ;
        }
        return (boolean) $val ;
    }

    public function optional_bool ($name, $default) {
        $result = $this->parse ($name, false, true) ;
        if (!$result[0] || ($result[1] == '')) return $default ;
        $val = strtolower($result[1]) ;
        switch ($val) {
            case 'false' : return false ;
            case 'true'  : return true ;
        }
        return (boolean) $val ;
    }

    public function optional_flag ($name) {
        $result = $this->parse ($name, false, false) ;
        return $result[0] ;
    }

    public function required_time_32 ($name) {
        require_once 'lusitime/lusitime.inc.php' ;
        return new \LusiTime\LusiTime ($this->required_int ($name)) ;
    }

    public function optional_time_32 ($name, $default) {
        $time32 = $this->optional_int ($name, null) ;
        if (is_null($time32)) return $default ;
        require_once 'lusitime/lusitime.inc.php' ;
        return new \LusiTime\LusiTime ($time32) ;
    }

    public function required_time_64 ($name) {
        require_once 'lusitime/lusitime.inc.php' ;
        return \LusiTime\LusiTime::from64 ($this->required_int ($name)) ;
    }

    public function optional_time_64 ($name, $default) {
        $time64 = $this->optional_int ($name, null) ;
        if (is_null($time64)) return $default ;
        require_once 'lusitime/lusitime.inc.php' ;
        return \LusiTime\LusiTime::from64 ($time64) ;
    }

    public function required_time ($name) {
        $result = $this->parse ($name, true, true) ;
        require_once 'lusitime/lusitime.inc.php' ;
        $time = \LusiTime\LusiTime::parse ($result[1]) ;
        if (!is_null($time)) return $time ;
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;
    }

    public function optional_time ($name, $default) {
        $result = $this->parse ($name, false, true, true) ;
        if (!$result[0] || ($result[1] == '')) return $default ;
        require_once 'lusitime/lusitime.inc.php' ;
        $time = \LusiTime\LusiTime::parse ($result[1]) ;
        if (!is_null($time)) return $time ;
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;
    }

    public function required_file () {
        $files = $this->optional_files () ;
        $num_files = count($files) ;
        if ($num_files == 0) $this->abort('required file attachment is missing') ;
        if ($num_files != 1) $this->abort('too many file attachments instdead of one') ;
        return $files[0] ;
    }

    public function optional_files () {
        if ($this->method != 'POST')
            $this->abort("can't process file uploading with method {$this->method} (POST is required)") ;

        $files = array () ;
        foreach ( array_keys($_FILES) as $file_key) {

            $name  = $_FILES[$file_key]['name'] ;
            $error = $_FILES[$file_key]['error'] ;

            if ($error != UPLOAD_ERR_OK) {
                if ($error == UPLOAD_ERR_NO_FILE) continue ;
                $this->abort("Attachment '{$name}' couldn't be uploaded because of the following problem: '".ServiceJSON::upload_err2string($error)."'.") ;
            }
            if ($name != '') {

                // Read file contents into a local variable
                //
                $location = $_FILES[$file_key]['tmp_name'] ;
                $size = filesize($location) ;
                $fd = fopen ($location, 'r') or $this->abort("failed to open file: {$location}") ;
                $contents = fread($fd, $size) ;
                fclose($fd) ;

                // Get its description. If none is present then use the original
                // name of the file at client's side.
                //
                $description = $name ;
                if (isset($_POST[$file_key])) {
                    $str = trim($_POST[$file_key]) ;
                    if ($str != '') $description = $str ;
                }
                array_push (
                    $files ,
                    array (
                        'type'        => $_FILES[$file_key]['type'] ,
                        'description' => $description ,
                        'contents'    => $contents ,
                        'size'        => $size
                    )
                ) ;
            }
        }
        return $files ;
    }

    // ------------------------
    //   Database connections
    // ------------------------

    public function authdb () {
        if (is_null($this->authdb)) {
            require_once 'authdb/authdb.inc.php' ;
            $this->authdb = \AuthDB\AuthDB::instance() ;
            $this->authdb->begin() ;
        }
        return $this->authdb ;
    }
    public function regdb () {
        if (is_null($this->regdb)) {
            require_once 'regdb/regdb.inc.php' ;
            $this->regdb = \RegDB\RegDB::instance() ;
            $this->regdb->begin() ;
        }
        return $this->regdb ;
    }
    public function logbook () {
        if (is_null($this->logbook)) {
            require_once 'logbook/logbook.inc.php' ;
            $this->logbook = \LogBook\LogBook::instance() ;
            $this->logbook->begin() ;
        }
        return $this->logbook ;
    }
    public function logbookauth () {
        if (is_null($this->logbookauth)) {
            require_once 'logbook/logbook.inc.php' ;
            $this->logbookauth = \LogBook\LogBookAuth::instance() ;
        }
        return $this->logbookauth ;
    }
    public function configdb () {
        if (is_null($this->configdb)) {
            require_once 'dataportal/dataportal.inc.php' ;
            $this->configdb = \DataPortal\Config::instance() ;
            $this->configdb->begin() ;
        }
        return $this->configdb ;
    }
    public function irodsdb () {
        if (is_null($this->irodsdb)) {
            require_once 'filemgr/filemgr.inc.php' ;
            $this->irodsdb = \FileMgr\FileMgrIrodsDb::instance() ;
            $this->irodsdb->begin() ;
        }
        return $this->irodsdb ;
    }
    public function neocaptar () {
        if (is_null($this->neocaptar)) {
            require_once 'neocaptar/neocaptar.inc.php' ;
            $this->neocaptar = \NeoCaptar\NeoCaptar::instance() ;
            $this->neocaptar->begin() ;
        }
        return $this->neocaptar ;
    }
    public function irep () {
        if (is_null($this->irep)) {
            require_once 'irep/irep.inc.php' ;
            $this->irep = \Irep\Irep::instance() ;
            $this->irep->begin() ;
        }
        return $this->irep ;
    }
    public function exptimemon () {
        if (is_null($this->exptimemon)) {
            require_once 'dataportal/dataportal.inc.php' ;
            $this->exptimemon = \DataPortal\ExpTimeMon::instance() ;
            $this->exptimemon->begin() ;
        }
        return $this->exptimemon ;
    }
    public function sysmon () {
        if (is_null($this->sysmon)) {
            require_once 'sysmon/sysmon.inc.php' ;
            $this->sysmon = \SysMon\SysMon::instance() ;
            $this->sysmon->begin() ;
        }
        return $this->sysmon ;
    }

    // -------------
    //  Finalizers
    // -------------

    public function abort ($message, $parameters=array()) {
        ServiceJSON::report_error ($message, $parameters, $this->options) ;
    }
    public function finish ($parameters=array()) {
        if (!is_null($this->authdb     )) $this->authdb    ->commit() ;
        if (!is_null($this->regdb      )) $this->regdb     ->commit() ;
        if (!is_null($this->logbook    )) $this->logbook   ->commit() ;
        if (!is_null($this->logbookauth)) ;
        if (!is_null($this->configdb   )) $this->configdb  ->commit() ;
        if (!is_null($this->irodsdb    )) $this->irodsdb   ->commit() ;
        if (!is_null($this->neocaptar  )) $this->neocaptar ->commit() ;
        if (!is_null($this->irep       )) $this->irep      ->commit() ;
        if (!is_null($this->exptimemon )) $this->exptimemon->commit() ;
        if (!is_null($this->sysmon     )) $this->sysmon    ->commit() ;
        ServiceJSON::report_success ($parameters, $this->options) ;
    }

    // ------------------
    //   Report results
    // ------------------

    private static function report_error ($message, $parameters, $options) {
        ServiceJSON::report_ (
            array (
                'status' => 'error' ,
                'message' => $message
            ) ,
            $parameters ,
            $options
        ) ;
    }
    private static function report_success ($parameters, $options) {
        ServiceJSON::report_ (
            array (
                'status'  => 'success' ,
                'updated' => LusiTime::now()->toStringShort()
            ) ,
            $parameters ,
            $options
        ) ;
    }
    private static function report_ ($status, $parameters, $options) {

        $response = json_encode (
            array_merge (
                $status ,
                is_array($parameters) ? $parameters : array()
            )
        ) ;

        /* Wrapping may be requiested by a client which uses JQuery AJAX Form plug-in to upload
         * files w/o reloading the current page. In that case we can not return JSON MIME type 
         * because of the following issue: http://jquery.malsup.com/form/#file-upload
         */
        if (in_array('wrap_in_textarea', $options) && $options['wrap_in_textarea']) {
            print "<textarea>{$response}</textarea>" ;
        } else {
            header( 'Content-type: application/json' );
            header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
            header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

            print $response ;
        }
        exit ;
    }
}

?>

