<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'authdb/authdb.inc.php' ;

use LusiTime\LusiTime ;

use \AuthDB\AuthDB ;


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
    public function parent_id    () { return intval($this->attr['parent_id']) ; }
    public function status       () { return   trim($this->attr['status']) ; }
    public function status2      () { return   trim($this->attr['status2']) ; }
    public function manufacturer () { return   trim($this->attr['manufacturer']) ; }
    public function model        () { return   trim($this->attr['model']) ; }
    public function serial       () { return   trim($this->attr['serial']) ; }
    public function description  () { return   trim($this->attr['description']) ; }
    public function slacid       () { return intval($this->attr['slacid']) ; }
    public function pc           () { return   trim($this->attr['pc']) ; }
    public function custodian    () { return   trim($this->attr['custodian']) ; }
    public function location     () { return   trim($this->attr['location']) ; }
    public function room         () { return   trim($this->attr['room']) ; }
    public function rack         () { return   trim($this->attr['rack']) ; }
    public function elevation    () { return   trim($this->attr['elevation']) ; }

    /**
     * Return the parent equipment object in a composition (if any). Return null otherwise.
     *
     * @return IrepEquipment
     */
    public function parent () {
        return
            $this->parent_id() ?
            $this->irep()->find_equipment_by_id($this->parent_id()) :
            null ;
    }
    /**
     * Connectt/disconnect to/from the parent object.
     *
     * @param IrepEquipment $parent
     */
    public function set_parent ($parent=null) {
        if (is_null($parent)) {
            $prev_parent_id = $this->parent_id() ;
            $this->update(array('parent_id' => 'NULL')) ;
            $this->attr['parent_id'] = '' ;
            $this->irep()->add_history_event($this->id(), 'Modified', array ("Disconnected from parent: {$prev_parent_id}")) ;
        } else {
            $this->update(array('parent_id' => $parent->id())) ;
            $this->attr['parent_id'] = $parent->id() ;
            $this->irep()->add_history_event($this->id(), 'Modified', array ("Connected to parent: {$parent->id()}")) ;
        }
    }

    /**
     * Return direct child equipment objects in a composition (if any).
     * 
     * @return array() in which each element has type IrepEquipment
     */
    public function children () {
        return $this->irep->find_equipment_many_by_("parent_id={$this->id()}") ;
    }

    /* --------------
     *   Properties
     * --------------
     */
    public function property ($name) {
        if (!array_key_exists($name, $this->attr))
            throw new IrepException (
                __METHOD__, "unknown equipment property: {$name}") ;
        return $this->attr[$name] ;
    }
    public function update($properties2update) {
        $sql_options = '' ;
        foreach ($properties2update as $property => $value) {
            $sql_options .= $sql_options == '' ? '' : ',' ;
            switch ($property) {
                case 'slacid':
                    $value_int = intval($value) ;
                    $sql_options .= "{$property}={$value_int}" ;
                    break ;
                case 'parent_id':
                    $value_opt = $value == 'NULL' ? $value : intval($value) ;
                    $sql_options .= "{$property}={$value_opt}" ;
                    break ;
                default:
                    $value_escaped = $this->irep()->escape_string(trim($value)) ;
                    $sql_options .= "{$property}='{$value_escaped}'" ;
                    break ;
             }
         }
         if ($sql_options != '')
            $this->irep()->query("UPDATE {$this->irep()->database}.equipment SET {$sql_options} WHERE id={$this->id()}") ;
    }

    /* --------------
     *   Atachments
     * --------------
     */
    public function attachments () {
        $list = array () ;
        $sql = "SELECT id,equipment_id,name,document_type,document_size,create_time,create_uid FROM {$this->irep()->database}.equipment_attachment WHERE equipment_id={$this->id()} ORDER BY create_time" ;
        $result = $this->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepEquipmentAttachment (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_attachment ($file, $uid) {
        $name_escaped     = $this->irep()->escape_string(trim($file['description'])) ;
        $type_escaped     = $this->irep()->escape_string(trim($file['type'])) ;
        $size             = intval($file['size']) ;
        $document_escaped = $this->irep()->escape_string($file['contents']) ;
        $now_64           = LusiTime::now()->to64() ;
        $uid_escaped      = $this->irep()->escape_string(trim($uid)) ;
        $sql =<<<HERE
INSERT INTO {$this->irep()->database}.equipment_attachment
  VALUES (
    NULL ,
    {$this->id()} ,
    '{$name_escaped}' ,
    '{$type_escaped}' ,
    {$size} ,
    '{$document_escaped}' ,
    NULL ,
    {$now_64} ,
    '{$uid_escaped}'
  )
HERE;
        $this->irep()->query($sql) ;
        $attachment = $this->find_attachment_by_('id IN (SELECT LAST_INSERT_ID())') ;
        $this->irep()->add_history_event($this->id(), 'Modified', array ("Add attachment: {$attachment->name()}")) ;
        return $attachment ;
    }
    public function find_attachment_by_id ($id) {
        $id = intval($id) ;
        return $this->find_attachment_by_("id={$id}") ;
    }
    private function find_attachment_by_ ($condition) {
        $list = array () ;
        $sql = "SELECT id,equipment_id,name,document_type,document_size,create_time,create_uid FROM {$this->irep()->database}.equipment_attachment WHERE equipment_id={$this->id()} AND {$condition}" ;
        $result = $this->irep()->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if (0 == $nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepEquipmentAttachment (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function delete_attachment ($id) {
        $id = intval($id) ;
        $attachment = $this->find_attachment_by_id($id) ;
        if (is_null($attachment)) return ;
        $sql = "DELETE FROM {$this->irep()->database}.equipment_attachment WHERE id={$id}" ;
        $this->irep()->query($sql) ;
        $this->irep()->add_history_event($this->id(), 'Modified', array ("Delete attachment: {$attachment->name()}")) ;
    }

    /* --------
     *   Tags
     * --------
     */
    public function tags () {
        $list = array () ;
        $sql = "SELECT * FROM {$this->irep()->database}.equipment_tag WHERE equipment_id={$this->id()} ORDER BY create_time" ;
        $result = $this->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepEquipmentTag (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function find_tag_by_id ($id) {
        $id = intval($id) ;
        return $this->find_tag_by_("id={$id}") ;
    }
    public function find_tag_by_name ($name) {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        return $this->find_tag_by_("name='{$name_escaped}'") ;
    }
    public function add_tag ($name) {
        $name_escaped = $this->irep()->escape_string(trim($name)) ;
        $create_time = LusiTime::now()->to64() ;
        $create_uid_escaped = $this->irep()->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->irep()->database}.equipment_tag VALUES(NULL,{$this->id()},'{$name_escaped}',{$create_time},'{$create_uid_escaped}')" ;
        $this->irep()->query($sql) ;
        $tag = $this->find_tag_by_('id=(SELECT LAST_INSERT_ID())') ;
        $this->irep()->add_history_event($this->id(), 'Modified', array ("Add tag: {$tag->name()}")) ;
        return $tag ;
    }
    public function delete_tag_by_id ($id) {
        $tag = $this->find_tag_by_id($id) ;
        if (is_null($tag)) return ;
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->irep()->database}.equipment_tag WHERE id={$id} AND equipment_id={$this->id()}" ;
        $this->irep()->query($sql) ;
        $this->irep()->add_history_event($this->id(), 'Modified', array ("Delete tag: {$tag->name()}")) ;
    }
    private function find_tag_by_ ($condition='') {
        $conditions_opt = $condition ? " AND {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->irep()->database}.equipment_tag WHERE equipment_id={$this->id()} {$conditions_opt}" ;
        $result = $this->irep()->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (0 == $nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepEquipmentTag (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }    

    /* -----------
     *   History
     * -----------
     */
    public function modified_time () {
        $event = $this->last_history_event() ;
        return $event ? $event->event_time() : null ;
    }
    public function modified_uid () {
        $event = $this->last_history_event() ;
        return $event ? $event->event_uid() : null ;
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

    /* -----------
     *   Issues
     * -----------
     */
    public function issues () {
        $list = array () ;
        $sql = "SELECT * FROM {$this->irep()->database}.issue WHERE equipment_id={$this->id()} ORDER BY open_time" ;
        $result = $this->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepIssue (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
}
?>
