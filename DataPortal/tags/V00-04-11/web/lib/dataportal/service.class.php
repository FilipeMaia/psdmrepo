<?php

namespace DataPortal ;

require_once 'dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime;

/*
 * This function doesn't exist prior PHP 5.4
 */
if (!function_exists('http_response_code')) {
    function http_response_code ($code=NULL) {

        if ($code !== NULL) {

            switch ($code) {
                case 100: $text = 'Continue'; break;
                case 101: $text = 'Switching Protocols'; break;
                case 200: $text = 'OK'; break;
                case 201: $text = 'Created'; break;
                case 202: $text = 'Accepted'; break;
                case 203: $text = 'Non-Authoritative Information'; break;
                case 204: $text = 'No Content'; break;
                case 205: $text = 'Reset Content'; break;
                case 206: $text = 'Partial Content'; break;
                case 300: $text = 'Multiple Choices'; break;
                case 301: $text = 'Moved Permanently'; break;
                case 302: $text = 'Moved Temporarily'; break;
                case 303: $text = 'See Other'; break;
                case 304: $text = 'Not Modified'; break;
                case 305: $text = 'Use Proxy'; break;
                case 400: $text = 'Bad Request'; break;
                case 401: $text = 'Unauthorized'; break;
                case 402: $text = 'Payment Required'; break;
                case 403: $text = 'Forbidden'; break;
                case 404: $text = 'Not Found'; break;
                case 405: $text = 'Method Not Allowed'; break;
                case 406: $text = 'Not Acceptable'; break;
                case 407: $text = 'Proxy Authentication Required'; break;
                case 408: $text = 'Request Time-out'; break;
                case 409: $text = 'Conflict'; break;
                case 410: $text = 'Gone'; break;
                case 411: $text = 'Length Required'; break;
                case 412: $text = 'Precondition Failed'; break;
                case 413: $text = 'Request Entity Too Large'; break;
                case 414: $text = 'Request-URI Too Large'; break;
                case 415: $text = 'Unsupported Media Type'; break;
                case 500: $text = 'Internal Server Error'; break;
                case 501: $text = 'Not Implemented'; break;
                case 502: $text = 'Bad Gateway'; break;
                case 503: $text = 'Service Unavailable'; break;
                case 504: $text = 'Gateway Time-out'; break;
                case 505: $text = 'HTTP Version not supported'; break;
                default:
                    exit('Unknown http status code "' . htmlentities($code) . '"');
                break;
            }

            $protocol = (isset($_SERVER['SERVER_PROTOCOL']) ? $_SERVER['SERVER_PROTOCOL'] : 'HTTP/1.0');

            header($protocol . ' ' . $code . ' ' . $text);

            $GLOBALS['http_response_code'] = $code;

        } else {
            $code = (isset($GLOBALS['http_response_code']) ? $GLOBALS['http_response_code'] : 200);
        }
        return $code;
    }
}

class Service {

    public static function run_handler ($method, $body) {

        try {
            $svc = new Service ($method) ;
            $body($svc) ;
        } catch( DataPortalException $e ) { Service::report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>', 400) ; }
          catch( \Exception $e ) { Service::report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>', 409) ; }
    }

    // ----------------
    //   Data members
    // ----------------

    private $method_var  = null ;
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

    public function __construct ($method) {
        switch (strtoupper(trim($method))) {
            case 'GET':  $this->method_var = $_GET ;  break ;
            case 'POST': $this->method_var = $_POST ; break ;
            default: throw new DataPortalException (__CLASS__.'::'.__METHOD__, 'illegal method parameter') ;
        }
    }

    public final function run () {
        try {
            $this->handler ($this) ;
        } catch( DataPortalException $e ) { $this->report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>', 400) ; }
          catch( \Exception $e ) { $this->report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>', 409) ; }
    }

    /**
     * User defined function to be overloaded if using object-oriented approach.
     */
    protected function handler ($SVC) {}


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
        return new LusiTime ($this->required_int ($name)) ;
    }
    public function optional_time_32 ($name, $default) {
        $time32 = $this->optional_int ($name, null) ;
        if (is_null($time32)) return $default ;
        return new LusiTime ($time32) ;
    }

    public function required_time_64 ($name) {
        return LusiTime::from64 ($this->required_int ($name)) ;
    }
    public function optional_time_64 ($name, $default) {
        $time64 = $this->optional_int ($name, null) ;
        if (is_null($time64)) return $default ;
        return LusiTime::from64 ($time64) ;
    }

    public function required_time ($name) {
        $result = $this->parse ($name, true, true) ;
        $time = LusiTime::parse ($result[1]) ;
        if (!is_null($time)) return $time ;
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;
    }
    public function optional_time ($name, $default) {
        $result = $this->parse ($name, false, true, true) ;
        if (!$result[0] || ($result[1] == '')) return $default ;
        $time = LusiTime::parse ($result[1]) ;
        if (!is_null($time)) return $time ;
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;
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

    public function abort ($message, $http_code=null) {
        Service::report_error ($message, $http_code) ;
    }
    public function finish () {
        $this->commit_transactions() ;
        Service::report_success() ;
    }
    protected function commit_transactions () {
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
    }
    // -------------------
    //   Rests reporting
    // -------------------

    private static function report_error ($message, $http_code=null) {
        if (is_null($http_code)) {
            error_log($message) ;
            print $message ;
        } else {
            error_log($message) ;
            http_response_code($http_code) ;
        }
        exit ;
    }
    private static function report_success () {
        exit ;
    }
}

?>
