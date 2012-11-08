<?php

namespace DataPortal ;

require_once 'dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime;

class ServiceJSON {

    public static function run_handler ($method, $body) {

        header( 'Content-type: application/json' );
        header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
        header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

        try {
            $svc = new ServiceJSON ($method) ;
            $body($svc) ;
        } catch( \Exception $e ) { ServiceJSON::report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>') ; }
    }

    // ----------------
    //   Data members
    // ----------------

    private $method_var = null ;
    private $authdb     = null ;
    private $regdb      = null ;
    private $logbook    = null ;
    private $configdb   = null ;
    private $irodsdb    = null ;
    private $neocaptar  = null ;
    private $irep       = null ;
    private $exptimemon = null ;
    private $sysmon     = null ;

    public function __construct ($method) {
        switch (strtoupper(trim($method))) {
            case 'GET':  $this->method_var = $_GET ;  break ;
            case 'POST': $this->method_var = $_POST ; break ;
            default: throw new DataPortalException (__CLASS__.'::'.__METHOD__, 'illegal method parameter') ;
        }
    }

    public final function run () {

        header( 'Content-type: application/json' );
        header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
        header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

        try {
            $this->handler ($this) ;
        } catch( \Exception $e ) { $this->report_error ($e.'<pre>'.print_r($e->getTrace(), true).'</pre>') ; }
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
        ServiceJSON::report_error ($message, $parameters) ;
    }
    public function finish ($parameters=array()) {
        if (!is_null($this->authdb    )) $this->authdb    ->commit() ;
        if (!is_null($this->regdb     )) $this->regdb     ->commit() ;
        if (!is_null($this->logbook   )) $this->logbook   ->commit() ;
        if (!is_null($this->configdb  )) $this->configdb  ->commit() ;
        if (!is_null($this->irodsdb   )) $this->irodsdb   ->commit() ;
        if (!is_null($this->neocaptar )) $this->neocaptar ->commit() ;
        if (!is_null($this->irep      )) $this->irep      ->commit() ;
        if (!is_null($this->exptimemon)) $this->exptimemon->commit() ;
        if (!is_null($this->sysmon    )) $this->sysmon    ->commit() ;
        ServiceJSON::report_success ($parameters) ;
    }

    // -------------------
    //   Rests reporting
    // -------------------

    private static function report_error ($message, $parameters=array()) {
        print json_encode (
            array_merge (
                array (
                    'status' => 'error' ,
                    'message' => $message
                ) ,
                $parameters
            )
        ) ;
        exit ;
    }
    private static function report_success ($parameters=array()) {
        print json_encode (
            array_merge (
                array (
                    'status'  => 'success' ,
                    'updated' => LusiTime::now()->toStringShort()
                ) ,
                $parameters
            )
        ) ;
        exit ;
    }
}

?>

