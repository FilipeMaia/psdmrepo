<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use AuthDB\AuthDB ;

use LusiTime\LusiTime ;

/**
 * Class IrepManufacturer is an abstraction for manufacturers.
 *
 * @author gapon
 */
class IrepManufacturer {

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
    public function irep         () { return $this->irep ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function description  () { return                  trim($this->attr['description']) ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }

    /* Models
     */
    public function models () {
        $list = array () ;
        $result = $this->irep()->query("SELECT * FROM {$this->irep()->database}.dict_model WHERE manufacturer_id={$this->id()} ORDER BY name") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++)
            array_push (
                $list,
                new IrepModel (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_model ($name, $description='') {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        $description_escaped = $this->irep()->escape_string(trim($description)) ;
        $created_time = LusiTime::now()->to64() ;
        $created_uid_escaped = $this->irep()->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->irep()->database}.dict_model VALUES(NULL,{$this->id()},'{$name_escaped}','{$description_escaped}',{$created_time},'{$created_uid_escaped}')" ;
        $this->irep()->query($sql) ;
        return $this->find_model_by_('id=(SELECT LAST_INSERT_ID())') ;
    }
    public function find_model_by_name ($name) {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        return $this->find_model_by_("name='{$name_escaped}'") ;
    }
    private function find_model_by_ ($condition='') {
        $conditions_opt = $condition ? " AND {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->irep()->database}.dict_model WHERE manufacturer_id={$this->id()} {$conditions_opt}" ;
        $result = $this->irep()->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepModel (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    
    /* Operations
     */
    public function update_description ($description) {
        $description = trim($description) ;
        $description_escaped = $this->irep()->escape_string($description) ;
        $sql = "UPDATE {$this->irep()->database}.dict_manufacturer SET description='{$description_escaped}' WHERE id={$this->id()}" ;
        $this->irep()->query($sql) ;
        $this->attr['description'] = $description ;
    }
}
?>
