<?php

namespace DataPortal ;

require_once 'dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime;

class ServiceJSON {

    public static function run_handler ($method, $body, $options=array()) {
        try {
            $svc = new ServiceJSON ($method, $options) ;
            $svc->finish($body($svc)) ;     // in case if users won't call $SVC->finish()
                                            // or chose to return result using 'return'.
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
    private $regdbauth   = null ;
    private $logbook     = null ;
    private $logbookauth = null ;
    private $configdb    = null ;
    private $irodsdb     = null ;
    private $neocaptar   = null ;
    private $irep        = null ;
    private $exptimemon  = null ;
    private $sysmon      = null ;
    private $shiftmgr    = null ;
    private $ifacectrldb = null ;   // multiple controllers in the dictonary
    private $ifacectrlws = null ;   // multiple controllers in the dictonary

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

    /**
     * Case-controlled version of the standard 'in_array()'
     *
     * @param  string  $needle      - a string value to search in the array
     * @param  array   $haystack    - an input array of strings where to search
     * @param  boolean $ignore_case - set to 'true' for case-insensitive comparision
     * @return boolean
     */
    private static function _in_array_case ($needle, $haystack, $ignore_case=false) {
        foreach ($haystack as $v) {
            if ($ignore_case ?
                strtolower($needle) === strtolower($v) :
                $needle             === $v) return true ;
        }
        return false ;
    }

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
    public function required_JSON ($name) {
        $result = json_decode($this->required_str ($name)) ;
        if (!is_null($result)) return $result ;
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__, "required parameter '{$name}' isn't a JSON object") ;
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

    public function required_time_any ($name) {
        return $this->_parse_time_any($name, $this->required_str ($name)) ;
    }
    public function optional_time_any ($name, $default) {
        $str = $this->optional_str ($name, null) ;
        if (is_null($str)) return $default ;
        return $this->_parse_time_any($name, $str) ;
    }
    private function _parse_time_any ($name, $str) {

        require_once 'lusitime/lusitime.inc.php' ;

        // Try ISO-style timestamp

        $time = \LusiTime\LusiTime::parse ($str) ;
        if (!is_null($time)) return $time ;

        // Try 64-bit second/nanosecond variant if the number
        // is larger than the maximum 32-bit unsigned integer.
        // Otherwise assume UNIX timestamp.
        //
        // NOTE: This logic still won't allow to resolve 
        //       ambiguty between a very small 64-bit number
        //       made of nanoseconds versus a UNIX timestamp.

        $num = intval($str) ;
        if ($num < pow(2,32)) return new \LusiTime\LusiTime($num) ;

        return \LusiTime\LusiTime::from64($str) ;
    }

    public static $_ENUM_OPTIONS = array (
        'ignore_case'     => false ,
        'convert'         => 'none'
    ) ;
    public function required_enum (
        $name ,
        $allowed_values ,
        $options = null) {

        return $this->_parse_enum (
            $this->required_str($name) ,
            $name ,
            $allowed_values ,
            $options
        ) ;
    }
    public function optional_enum (
        $name ,
        $allowed_values ,
        $default ,
        $options = null) {

        $str = $this->optional_str($name, null) ;
        if (is_null($str)) return $default ;

        return $this->_parse_enum (
            $this->required_str($name) ,
            $name ,
            $allowed_values ,
            $options
        ) ;
    }
    private function _parse_enum (
        $str ,
        $name ,
        $allowed_values ,
        $options) {

        $options = is_null($options) ? self::$_ENUM_OPTIONS : $options ;

        $ignore_case = array_key_exists('ignore_case', $options) && $options['ignore_case'] ;

        $convert = array_key_exists('convert', $options) ?  $options['convert'] : 'none' ;
        if (!in_array($convert, array('none', 'toupper', 'tolower')))
            throw new DataPortalException (
                    __CLASS__.'::'.__METHOD__, "invalid value of the string case conversion option: {$convert}'") ;

        // Note that we this check has to be done first before applying
        // the optional case conversion rule

        if (!ServiceJSON::_in_array_case($str, $allowed_values, $ignore_case))
            throw new DataPortalException (
                __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;

        // (Optional) case conversion has to be made before placing result
        // in case if duplicates aren't allowed.

        switch ($convert) {
            case 'none'    : break ;
            case 'toupper' : $str = strtoupper($str) ; break ;
            case 'tolower' : $str = strtolower($str) ; break ;
        }
        return $str ;
    }
        
    public static $_LIST_OPTIONS = array (
        'ignore_case'     => false ,
        'convert'         => 'none' ,
        'skip_duplicates' => false
    ) ;

    public function required_list (
        $name ,
        $allowed_values ,
        $options = null) {

        return $this->_parse_list (
            $this->required_str($name) ,
            $name ,
            $allowed_values ,
            $options) ;
    }
    public function optional_list (
        $name ,
        $allowed_values ,
        $default ,
        $options = null) {

        $str = $this->optional_str($name, null) ;
        if (is_null($str)) return $default ;

        return $this->_parse_list (
            $str ,
            $name ,
            $allowed_values ,
            $options) ;
    }

    private function _parse_list (
        $str ,
        $name ,
        $allowed_values ,
        $options) {

        $options = is_null($options) ? self::$_LIST_OPTIONS : $options ;

        $skip_duplicates = array_key_exists('skip_duplicates', $options) && $options['skip_duplicates'] ;
        $ignore_case     = array_key_exists('ignore_case',     $options) && $options['ignore_case'] ;

        $convert = array_key_exists('convert',         $options) ? $options['convert'] : 'none' ;
        if (!in_array($convert, array('none', 'toupper', 'tolower')))
            throw new DataPortalException (
                    __CLASS__.'::'.__METHOD__, "invalid value of the string case conversion option: {$convert}'") ;
        
        $values = array() ;
        foreach (explode(',', $str) as $v) {

            // Note that we this check has to be done first before applying
            // the optional case conversion rule

            if (!ServiceJSON::_in_array_case($v, $allowed_values, $ignore_case))
                throw new DataPortalException (
                    __CLASS__.'::'.__METHOD__, "invalid value of parameter '{$name}'") ;

            // (Optional) case conversion has to be made before placing result
            // in case if duplicates aren't allowed.

            switch ($convert) {
                case 'none'    : break ;
                case 'toupper' : $v = strtoupper($v) ; break ;
                case 'tolower' : $v = strtolower($v) ; break ;
            }
            if ($skip_duplicates && ServiceJSON::_in_array_case($v, $values, $ignore_case)) continue ;
            array_push($values, $v) ;
        }
        return $values ;
    }

    public function required_range ($name) {

        $str = $this->required_str($name, null) ;

        $values = explode('-', $str);
        switch (count($values)) {

        case 0:
            return array('min' => null, 'max' => null) ;
            
        case 1:
            $min = intval($values[0]) ;
            if ($min <= 0) break ;
            return array('min' => $min, 'max' => $min);

        case 2:
            $min = $runs[0] ? intval($runs[0]) : null ;
            $max = $runs[1] ? intval($runs[1]) : null ;
            if ((!is_null($min) && ($min <= 0)) || (!is_null($max) && ($max <= 0)) || (($min && $max) && ($min > $max))) break ;
            return array('min' => $min, 'max' => $max);
        }
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__ ,
            "invalid value of parameter '{$name}'. Please use simple range like: '13', '1-20', '100-', '-200'") ;
    }

    public function optional_range ($name, $default_range) {

        $str = $this->optional_str($name, null) ;
        if (is_null($str)) return $default_range ;

        $values = explode('-', $str);
        switch (count($values)) {

        case 0:
            return array('min' => null, 'max' => null) ;
            
        case 1:
            $min = intval($values[0]) ;
            if ($min <= 0) break ;
            return array('min' => $min, 'max' => $min);

        case 2:
            $min = $values[0] ? intval($values[0]) : null ;
            $max = $values[1] ? intval($values[1]) : null ;
            if ((!is_null($min) && ($min <= 0)) || (!is_null($max) && ($max <= 0)) || (($min && $max) && ($min > $max))) break ;
            return array('min' => $min, 'max' => $max);
        }
        throw new DataPortalException (
            __CLASS__.'::'.__METHOD__ ,
            "invalid value of parameter '{$name}'. Please use simple range like: '13', '1-20', '100-', '-200'") ;
    }

    public function required_file () {
        $files = $this->optional_files () ;
        $num_files = count($files) ;
        if ($num_files == 0) $this->abort('required file attachment is missing') ;
        if ($num_files != 1) $this->abort('too many file attachments instdead of one') ;
        return $files[0] ;
    }
    public function optional_files () {

        $this->assert ($this->method == 'POST' ,
                       "can't process file uploading with method {$this->method} (POST is required)") ;

        $files = array () ;
        foreach (array_keys($_FILES) as $file_key) {

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

    public function optional_tags () {

        $this->assert ($this->method == 'POST' ,
                       "can't process tags uploading with method {$this->method} (POST is required)") ;

        $tags = array();
        if (isset($_POST['num_tags'])) {
            
            $num_tags = 0 ;
            $this->assert (sscanf(trim($_POST['num_tags']), "%d", $num_tags) == 1 ,
                           'not a number where a number of tags was expected') ;

            for ($i = 0; $i < $num_tags; $i++) {
                $tag_name_key  = 'tag_name_'.$i ;
                if (isset($_POST[$tag_name_key])) {

                    $tag = trim($_POST[$tag_name_key]) ;
                    if ($tag) {

                        $tag_value_key = 'tag_value_'.$i ;
                        $this->assert (isset($_POST[$tag_value_key]) ,
                                       "No valid value for tag {$tag_name_key}") ;

                        $value = trim($_POST[$tag_value_key]) ;

                        array_push (
                            $tags ,
                            array (
                                'tag'   => $tag ,
                                'value' => $value)) ;
                    }
                }
            }
        }
        return $tags ;
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
    public function regdbauth () {
        if (is_null($this->regdbauth)) {
            require_once 'regdb/regdb.inc.php' ;
            $this->regdbauth = \RegDB\RegDBAuth::instance() ;
        }
        return $this->regdbauth ;
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
    public function shiftmgr () {
        if (is_null($this->shiftmgr)) {
            require_once 'shiftmgr/shiftmgr.inc.php' ;
            $this->shiftmgr = \ShiftMgr\ShiftMgr::instance() ;
            $this->shiftmgr->begin() ;
        }
        return $this->shiftmgr ;
    }
    public function ifacectrldb ($service_name='STANDARD') {
        if (is_null($this->ifacectrldb)) {
            require_once 'filemgr/filemgr.inc.php' ;
            $this->ifacectrldb = array() ;
        }
        if (!array_key_exists($service_name, $this->ifacectrldb)) {
            $this->ifacectrldb[$service_name] = \FileMgr\IfaceCtrlDb::instance($service_name) ;
            $this->ifacectrldb[$service_name]->begin() ;
        }
        return $this->ifacectrldb[$service_name] ;
    }
    public function ifacectrlws ($service_name='STANDARD') {
        if (is_null($this->ifacectrlws)) {
            require_once 'filemgr/filemgr.inc.php' ;
            $this->ifacectrlws = array() ;
        }
        if (!array_key_exists($service_name, $this->ifacectrlws)) {
            $this->ifacectrlws[$service_name] = \FileMgr\FileMgrIfaceCtrlWs1::instance($service_name) ;
        }
        return $this->ifacectrlws[$service_name] ;
    }
        
    // -----------------------
    //  Convenience functions
    // -----------------------

    /**
     * Sanity check for an input parameter to be sure it's set and it's not null
     *
     * @param mixed $in
     * @param string $msg
     * @param array $parameters
     * @return mixed
     */
    public function safe_assign ($in, $msg, $parameters=array()) {
        $this->assert(isset($in) || !is_null($in), $msg, $parameters) ;
        return $in ;
    }

    /**
     * Assert that the input parameter evaluates as TRUE
     *
     * @param mixed $condition
     * @param string $msg
     * @param array $parameters
     */
    public function assert ($condition, $msg, $parameters=array()) {
        if (!$condition) $this->abort($msg ? $msg : 'Web service failed', $parameters) ;
    }

    // -------------
    //  Finalizers
    // -------------

    public function abort ($message, $parameters=array()) {
        ServiceJSON::report_error ($message, $parameters, $this->options) ;
    }
    public function finish ($parameters) {
        if (!$parameters) { $parameters = array() ; }
        if (!is_null($this->authdb     )) { $this->authdb    ->commit() ; }
        if (!is_null($this->regdb      )) { $this->regdb     ->commit() ; }
        if (!is_null($this->regdbauth  )) { }
        if (!is_null($this->logbook    )) { $this->logbook   ->commit() ; }
        if (!is_null($this->logbookauth)) { }
        if (!is_null($this->configdb   )) { $this->configdb  ->commit() ; }
        if (!is_null($this->irodsdb    )) { $this->irodsdb   ->commit() ; }
        if (!is_null($this->neocaptar  )) { $this->neocaptar ->commit() ; }
        if (!is_null($this->irep       )) { $this->irep      ->commit() ; }
        if (!is_null($this->exptimemon )) { $this->exptimemon->commit() ; }
        if (!is_null($this->sysmon     )) { $this->sysmon    ->commit() ; }
        if (!is_null($this->shiftmgr   )) { $this->shiftmgr  ->commit() ; }
        if (!is_null($this->ifacectrldb)) { foreach ($this->ifacectrldb as $ctrl) { $ctrl->commit() ; }}
        if (!is_null($this->ifacectrlws)) { }
        ServiceJSON::report_success ($parameters, $this->options) ;
    }

    // ------------------
    //   Report results
    // ------------------

    private static function report_error ($message, $parameters, $options) {
        $status = 'error' ;
        ServiceJSON::report_ (
            array (
                'status'  => $status ,
                'message' => $message ,
                'Status'  => $status ,      // -- backward compatibility with older clients 
                'Message' => $message       // -- backward compatibility with older clients 
            ) ,
            $parameters ,
            $options
        ) ;
    }
    private static function report_success ($parameters, $options) {
        $status  = 'success' ;
        $now_str = LusiTime::now()->toStringShort() ;
        ServiceJSON::report_ (
            array (
                'status'  => $status ,
                'updated' => $now_str ,
                'Status'  => $status ,      // -- backward compatibility with older clients 
                'Updated' => $now_str       // -- backward compatibility with older clients 
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

