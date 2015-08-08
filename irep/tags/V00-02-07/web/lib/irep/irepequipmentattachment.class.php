<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepEquipmentAttachment is an abstraction for equipment attachments.
 *
 * @author gapon
 */
class IrepEquipmentAttachment {

    /* Data members
     */
    private $equipment ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($equipment, $attr) {
        $this->equipment = $equipment ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function equipment () { return $this->equipment ; }

    public function id   () { return intval($this->attr['id']) ; }
    public function name () { return   trim($this->attr['name']) ; }

    public function document_type () { return   trim($this->attr['document_type']) ; }
    public function document_size () { return intval($this->attr['document_size']) ; }

    public function create_time () { return LusiTime::from64(intval($this->attr['create_time'])) ; }
    public function create_uid  () { return                    trim($this->attr['create_uid']) ; }

    /* The payload and preview image can be very large so we will load them on demand
     */
    public function document () {
        $sql = "SELECT document FROM {$this->equipment()->irep()->database}.equipment_attachment WHERE id={$this->id()}" ;
        $result = $this->equipment()->irep()->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
        return $attr['document'] ;
    }
    public function document_preview () {
        $sql = "SELECT document_preview FROM {$this->equipment()->irep()->database}.equipment_attachment WHERE id={$this->id()}" ;
        $result = $this->equipment()->irep()->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
        return $attr['document_preview'] ;
    }
    
    public function update_document_preview ($stringdata=null) {
        $stringdata_value = is_null($stringdata) ? 'NULL' : "'".$this->equipment()->irep()->escape_string($stringdata)."'" ;
        $sql = "UPDATE {$this->equipment()->irep()->database}.equipment_attachment SET document_preview={$stringdata_value} WHERE id={$this->id()}" ;
        $this->equipment()->irep()->query($sql) ;
    }
}
?>
