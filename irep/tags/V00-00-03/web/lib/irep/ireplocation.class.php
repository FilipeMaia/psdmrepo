<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'authdb/authdb.inc.php' ;

use LusiTime\LusiTime ;

use \AuthDB\AuthDB ;

/**
 * Class IrepLocation is an abstraction for locations.
 *
 * @author gapon
 */
class IrepLocation {

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
    public function irep         () { return                       $this->irep ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }

    /* Rooms
     */
    public function rooms () {
        $list = array () ;
        $result = $this->irep()->query("SELECT * FROM {$this->irep()->database}.dict_room WHERE location_id={$this->id()} ORDER BY name") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++)
            array_push (
                $list,
                new IrepRoom (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_room ($name) {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        $created_time = LusiTime::now()->to64() ;
        $created_uid_escaped = $this->irep()->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->irep()->database}.dict_room VALUES(NULL,{$this->id()},'{$name_escaped}',{$created_time},'{$created_uid_escaped}')" ;
        $this->irep()->query($sql) ;
        return $this->find_room_by_('id=(SELECT LAST_INSERT_ID())') ;
    }
    public function find_room_by_name ($name) {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        return $this->find_room_by_("name='{$name_escaped}'") ;
    }
    private function find_room_by_ ($condition='') {
        $conditions_opt = $condition ? " AND {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->irep()->database}.dict_room WHERE location_id={$this->id()} {$conditions_opt}" ;
        $result = $this->irep()->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepRoom (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
}
?>
