<?php

namespace FileMgr ;

require_once 'filemgr.inc.php' ;

use FileMgr\FileMgrException ;
use FileMgr\DbConnection ;


/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit","256M") ;

use \stdClass ;

/*
 * The helper utility class to deal with Interface Controller Database
 */
class IfaceCtrlDb extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $CONN_PARAMS = array (
        'STANDARD' => array (
            'HOST'     => IFACECTRL_STANDARD_DEFAULT_HOST ,
            'USER'     => IFACECTRL_STANDARD_DEFAULT_USER ,
            'PASSWORD' => IFACECTRL_STANDARD_DEFAULT_PASSWORD ,
            'DATABASE' => IFACECTRL_STANDARD_DEFAULT_DATABASE
        ) ,
        'MONITORING' => array (
            'HOST'     => IFACECTRL_MONITORING_DEFAULT_HOST ,
            'USER'     => IFACECTRL_MONITORING_DEFAULT_USER ,
            'PASSWORD' => IFACECTRL_MONITORING_DEFAULT_PASSWORD ,
            'DATABASE' => IFACECTRL_MONITORING_DEFAULT_DATABASE
        )
    ) ;
    private static $instance = null ;

    public static $AUTO_TRANSLATE_HDF5 = array (
        'STANDARD'   => 'AUTO_TRANSLATE_HDF5' ,
        'MONITORING' => 'FFB_AUTO_TRANSLATE_HDF5'
    ) ;

    /**
     * Singleton to simplify certain operations.
     *
     * @return IfaceCtrlDb
     */
    public static function instance ($service_name='STANDARD') {
        if (is_null(IfaceCtrlDb::$instance)) IfaceCtrlDb::$instance = array() ;
        if (!array_key_exists($service_name, IfaceCtrlDb::$instance))
            IfaceCtrlDb::$instance[$service_name] =
                new IfaceCtrlDb (
                    IfaceCtrlDb::$CONN_PARAMS[$service_name]['HOST'] ,
                    IfaceCtrlDb::$CONN_PARAMS[$service_name]['USER'] ,
                    IfaceCtrlDb::$CONN_PARAMS[$service_name]['PASSWORD'] ,
                    IfaceCtrlDb::$CONN_PARAMS[$service_name]['DATABASE']) ;
        return IfaceCtrlDb::$instance[$service_name] ;
    }

    /**
     * Constructor
     *
     * @param {String} $host
     * @param {String} $user
     * @param {String} $password
     * @param {String} $database 
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct ($host, $user, $password, $database) ;
    }

    private function prepare_SELECT_option ($param, $name, $default_test){
        if (is_null($param) || trim($param) === '') return "{$name} {$default_test}" ;
        return "{$name} = '{$this->escape_string(trim($param))}'" ;
    }
    private function prepare_INSERT_option ($param, $default) {
        if (is_null($param) || trim($param) === '') return "{$default}" ;
        return "'{$this->escape_string(trim($param))}'" ;
    }
    private function prepare_INSERT_option_Integer ($param, $default) {
        if (is_null($param)) return $default ;
        return $param ;
    }
    private function prepare_INSERT_option_Float ($param, $default) {
        if (is_null($param)) return $default ;
        return $param ;
    }
    private function prepare_UPDATE_option ($param, $name) {
        if (is_null($param) || trim($param) === '')
            throw new FileMgrException(__METHOD__, "the value of parameter '{$name}' can't be empty") ;
        return "{$name} = '{$this->escape_string(trim($param))}'" ;
    }
    private function prepare_UPDATE_option_Integer ($param, $name) {
        if (is_null($param))
            throw new FileMgrException(__METHOD__, "the value of parameter '{$name}' can't be empty") ;
        return "{$name} = {$param}" ;
    }
    private function prepare_UPDATE_option_Float ($param, $name) {
        if (is_null($param))
            throw new FileMgrException(__METHOD__, "the value of parameter '{$name}' can't be empty") ;
        return "{$name} = {$param}" ;
    }

    /**
     * Get a value of a configuration parameter or null
     *
     * @param {String} $section
     * @param {String} $param
     * @param {String} $instrument
     * @param {String} $experiment
     * @return {String}
     * @throws FileMgrException
     */
    public function get_config_param_val (
        $section ,
        $param ,
        $instrument ,
        $experiment)
    {
        $sql = "SELECT value FROM {$this->database}.config_def ".
               "WHERE ".
               " ".$this->prepare_SELECT_option($section,    'section',    " = ''"   )." AND ".
               " ".$this->prepare_SELECT_option($param,      'param',      " = ''"   )." AND ".
               " ".$this->prepare_SELECT_option($instrument, 'instrument', " IS NULL")." AND ".
               " ".$this->prepare_SELECT_option($experiment, 'experiment', " IS NULL") ;
        
        $result = $this->query ($sql);
        $nrows  = mysql_numrows($result);
        if (!$nrows) return null ;
        if ($nrows != 1) throw new FileMgrException(__METHOD__, "duplicate entries for query {$sql}). Database may be corrupted.") ;

        $row = mysql_fetch_array($result, MYSQL_ASSOC) ;
        
        return $row['value'] ;
    }

    /**
     * Set a new value to configuration parameter
     *
     * @param {String} $section
     * @param {String} $param
     * @param {String} $instrument
     * @param {String} $experiment
     * @param {String} $value
     * @param {String} $description
     */
    public function set_config_param_val (
        $section ,
        $param ,
        $instrument ,
        $experiment ,
        $value ,
        $description ,
        $type = 'String')
    {
        if (is_null (
            $this->get_config_param_val (
                $section ,
                $param ,
                $instrument ,
                $experiment))) {

            switch ($type) {
                case 'Integer': $value_option = $this->prepare_INSERT_option_Integer($value, 0) ;    break ;
                case 'Float':   $value_option = $this->prepare_INSERT_option_Float  ($value, 0.0) ;  break ;
                case 'String':  $value_option = $this->prepare_INSERT_option        ($value, "''") ; break ;
                default:        throw new FileMgrException(__METHOD__, "unsupported type: '{$type}'") ;
            }
            $sql =
                "INSERT INTO {$this->database}.config_def ".
                   "VALUES (".
                   " ".$this->prepare_INSERT_option($section,     "''"  ).", ".
                   " ".$this->prepare_INSERT_option($param,       "''"  ).", ".
                   " {$value_option}, ".
                   " '{$type}', ".
                   " ".$this->prepare_INSERT_option($description, "''"  ).", ".
                   " ".$this->prepare_INSERT_option($instrument,  'NULL').", ".
                   " ".$this->prepare_INSERT_option($experiment,  'NULL').")" ;

        } else {

            switch ($type) {
                case 'Integer': $value_option = $this->prepare_UPDATE_option_Integer($value, 'value') ; break ;
                case 'Float':   $value_option = $this->prepare_UPDATE_option_Float  ($value, 'value') ; break ;
                case 'String':  $value_option = $this->prepare_UPDATE_option        ($value, 'value') ; break ;
                default:        throw new FileMgrException(__METHOD__, "unsupported type: '{$type}'") ;
            }
            $sql =
                "UPDATE {$this->database}.config_def ".
                    "SET ".
                    " {$value_option} ".
                    "WHERE ".
                    " ".$this->prepare_SELECT_option($section,    'section',    " = ''"   )." AND ".
                    " ".$this->prepare_SELECT_option($param,      'param',      " = ''"   )." AND ".
                    " ".$this->prepare_SELECT_option($instrument, 'instrument', " IS NULL")." AND ".
                    " ".$this->prepare_SELECT_option($experiment, 'experiment', " IS NULL") ;
        }
        $this->query($sql) ;
    }
    
    /**
     * Remove a configuration arameter (if any)
     *
     * @param {String} $section
     * @param {String} $param
     * @param {String} $instrument
     * @param {String} $experiment
     */
    public function remove_config_param (
        $section ,
        $param ,
        $instrument ,
        $experiment)
    {
        $sql =
            "DELETE FROM {$this->database}.config_def ".
            "WHERE ".
            " ".$this->prepare_SELECT_option($section,    'section',    " = ''"   )." AND ".
            " ".$this->prepare_SELECT_option($param,      'param',      " = ''"   )." AND ".
            " ".$this->prepare_SELECT_option($instrument, 'instrument', " IS NULL")." AND ".
            " ".$this->prepare_SELECT_option($experiment, 'experiment', " IS NULL") ;
        
        $this->query($sql) ;
    }
    
    /**
     * Recursively find a value of the parameter in any scope (experiment, instrument or global)
     *
     * @param String $section
     * @param String $param
     * @param String $instrument
     * @param String $experiment
     * @return String || Null
     */
    public function get_config_param_val_r (
        $section ,
        $param ,
        $instrument ,
        $experiment)
    {
        $result = $this->get_config_param_val($section, $param, $instrument, $experiment) ;
        if (is_null($result)) {
            $result = $this->get_config_param_val($section, $param, $instrument, '') ;
            if (is_null($result)) {
                $result = $this->get_config_param_val($section, $param, '', '') ;
            }
        }
        return $result ;
    }
}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *
try {
    $ifacectrldb = IfaceCtrlDb::instance() ;
    $ifacectrldb->begin() ;

    $instrument = 'AMO' ;
    $experiment = 'amodaq09' ;
    $section    = '' ;
    $param      = 'release' ;

    print <<<HERE
<div style="padding-left:10px;" >
  <h3>The current value of the parameter</h3>
HERE;
    $val        = $ifacectrldb->get_config_param_val($section, $param, $instrument, $experiment) ;
    if (is_null($val)) $val = '&nbsp;' ;
    print <<< HERE
  <div style="padding-left:10px;" >
    <table><tbody>
      <tr><td align="right" ><b>Section :</b>    </td><td>{$section}</td></tr>
      <tr><td align="right" ><b>Param :</b>      </td><td>{$param}</td></tr>
      <tr><td align="right" ><b>Instrument :</b> </td><td>{$instrument}</td></tr>
      <tr><td align="right" ><b>Experiment :</b> </td><td>{$experiment}</td></tr>
      <tr><td align="right" ><b>Value :</b>      </td><td>{$val}</td></tr>
    </tbody></table>
  </div>
HERE;

    $val         = '/reg/g/psdm/sw/releases/ana-current/' ;
    $description = 'release diretory from where to run the Translator' ;

    print <<<HERE
  <h3>Setting a new value of the parameter</h3>
  <div style="padding-left:10px;" >
    <table><tbody>
      <tr><td align="right" ><b>Value :</b>       </td><td>{$val}</td></tr>
      <tr><td align="right" ><b>Description :</b> </td><td>{$description}</td></tr>
    </tbody></table>
  </div>
HERE;
    $ifacectrldb->set_config_param_val($section, $param, $instrument, $experiment, $val, $description) ;

    print <<<HERE
<div style="padding-left:10px;" >
  <h3>The updated value of the parameter</h3>
HERE;
    $val        = $ifacectrldb->get_config_param_val($section, $param, $instrument, $experiment) ;
    if (is_null($val)) $val = '&nbsp;' ;
    print <<< HERE
  <div style="padding-left:10px;" >
    <table><tbody>
      <tr><td align="right" ><b>Section :</b>    </td><td>{$section}</td></tr>
      <tr><td align="right" ><b>Param :</b>      </td><td>{$param}</td></tr>
      <tr><td align="right" ><b>Instrument :</b> </td><td>{$instrument}</td></tr>
      <tr><td align="right" ><b>Experiment :</b> </td><td>{$experiment}</td></tr>
      <tr><td align="right" ><b>Value :</b>      </td><td>{$val}</td></tr>
    </tbody></table>
  </div>
HERE;

    print <<<HERE
  <h3>Removing the parameter</h3>
HERE;
    $ifacectrldb->remove_config_param($section, $param, $instrument, $experiment) ;

    print <<<HERE
</div>
HERE;

    $ifacectrldb->commit() ;

} catch (Exception        $e) { print $e ; }
  catch (FileMgrException $e) { print $e->toHtml() ; }
*/
?>
