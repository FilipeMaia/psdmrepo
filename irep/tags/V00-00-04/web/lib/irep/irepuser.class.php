<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepUser is an abstraction for users.
 *
 * @author gapon
 */
class IrepUser {

    /* Data members
     */
    private $irep ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($irep, $attr) {
        $this->irep = $irep ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function irep             () { return $this->irep ; }
    public function uid              () { return trim($this->attr['uid' ]) ; }
    public function role             () { return trim($this->attr['role']) ; }
    public function name             () { return trim($this->attr['name']) ; }
    public function added_time       () { return LusiTime::from64( trim($this->attr['added_time'])) ; }
    public function added_uid        () { return trim($this->attr['added_uid']) ; }
    public function last_active_time () { return trim($this->attr['last_active_time']) == '' ? '' : LusiTime::from64( trim($this->attr['last_active_time'])) ; }

    /* Helper functions
     */
    public function is_administrator () { return 'ADMINISTRATOR' == $this->role() ; }
    public function is_editor        () { return 'EDITOR'        == $this->role() ; }
    public function is_other         () { return 'OTHER'         == $this->role() ; }

    /* Fine grain privileges
     */
    public function has_dict_priv () {
        if ($this->is_administrator()) return true ;
        $val = $this->get_priv('dict_priv') ;
        return !is_null($val) && ($val == 'YES') ;
    }
    public function set_dict_priv ($on) { $this->set_priv('dict_priv',$on) ; }

    private function set_priv ($name, $on) {
        $uid_escaped  = $this->irep()->escape_string($this->uid()) ;
        $name = strtolower(trim($name)) ;
        $on_option = $on  ? 'YES' : 'NO' ;
        $val = $this->get_priv($name) ;
        $result = $this->irep()->query (
            is_null($val)
                ? "INSERT INTO {$this->irep()->database}.user_priv VALUES('{$uid_escaped}','{$on_option}')"
                : "UPDATE {$this->irep()->database}.user_priv SET {$name}='{$on_option}' WHERE uid='{$uid_escaped}'"
        ) ;
    }
    private function get_priv ($name) {
        $uid_escaped  = $this->irep()->escape_string($this->uid()) ;
        $name = strtolower(trim($name)) ;
        $result = $this->irep()->query("SELECT {$name} FROM {$this->irep()->database}.user_priv WHERE uid='{$uid_escaped}'") ;
        $nrows = mysql_numrows( $result ) ;
        if( $nrows == 0 ) return null ;
        if( $nrows != 1 )
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." ) ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
        return strtoupper($attr[$name]) ;
    }
}
?>
