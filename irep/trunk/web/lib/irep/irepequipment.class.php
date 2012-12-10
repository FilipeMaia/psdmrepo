<?php

namespace Irep ;

require_once 'irep.inc.php' ;

/**
 * Class IrepEquipment is an abstraction for equipment.
 *
 * @author gapon
 */
class IrepEquipment {

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
    public function irep         () { return        $this->irep ; }
    public function id           () { return intval($this->attr['id']) ; }
    public function status       () { return   trim($this->attr['status']) ; }
    public function status2      () { return   trim($this->attr['status2']) ; }
    public function manufacturer () { return   trim($this->attr['manufacturer']) ; }
    public function model        () { return   trim($this->attr['model']) ; }
    public function serial       () { return   trim($this->attr['serial']) ; }
    public function description  () { return   trim($this->attr['description']) ; }
    public function slacid       () { return intval($this->attr['slacid']) ; }
    public function pc           () { return   trim($this->attr['pc']) ; }
    public function location     () { return   trim($this->attr['location']) ; }
    public function custodian    () { return   trim($this->attr['custodian']) ; }
    

    public function property ($name) {
        if (!array_key_exists($name, $this->attr))
            throw new IrepException (
                __METHOD__, "unknown equipment property: {$name}") ;
        return $this->attr[$name] ;
    }

    /* Operations
     */
    public function attachments () {
        $list = array () ;
        $sql = "SELECT id,equipment_id,name,document_type,document_size,create_time,create_uid FROM {$this->irep()->database}.equipment_attachment WHERE equipment_id={$this->id()} ORDER BY create_time" ;
        $result = $this->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepEquipmentAttachment (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function last_history_event () {
        return $this->irep()->last_history_event($this->id()) ;
    }
    public function history () {
        $list = array () ;
        $sql = "SELECT * FROM {$this->irep()->database}.equipment_history WHERE equipment_id={$this->id()} ORDER BY event_time" ;
        $result = $this->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepEquipmentHistoryEvent (
                    $this->irep() ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    
    public function update($properties2update) {
        $sql_options = '' ;
        foreach ($properties2update as $property => $value) {
            $sql_options .= $sql_options == '' ? '' : ',' ;
            switch ($property) {
                case 'status':
                case 'status2':
                case 'manufacturer':
                case 'model':
                case 'serial':
                case 'description':
                case 'pc':
                case 'location':
                case 'custodian':
                    $value_escaped = $this->irep()->escape_string(trim($value)) ;
                    $sql_options .= "{$property}='{$value_escaped}'" ;
                    break ;
                case 'slacid':
                    $value_int = intval($value) ;
                    $sql_options .= "{$property}={$value}" ;
                    break ;
                default:
                    throw new IrepException (
                        __METHOD__, "unknown equipment property: {$property}") ;
             }
         }
         if ($sql_options != '')
            $this->irep()->query("UPDATE {$this->irep()->database}.equipment SET {$sql_options} WHERE id={$this->id()}") ;
    }
}
?>
