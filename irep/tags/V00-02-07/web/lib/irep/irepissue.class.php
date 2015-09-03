<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'authdb/authdb.inc.php' ;

use LusiTime\LusiTime ;

use \AuthDB\AuthDB ;


/**
 * Class IrepAction is an abstraction for equipment issue actions.
 *
 * @author gapon
 */
class IrepAction {

    /* Data members
     */
    private $issue ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($issue, $attr) {
        $this->issue = $issue ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function issue       () { return        $this->issue ; }
    public function id          () { return intval($this->attr['id']) ; }
    public function description () { return   trim($this->attr['description']) ; }
    public function action      () { return        $this->attr['action'] ; }

    /* ---------------
     *   Attachments
     * ---------------
     */
    public function attachments () {
        $irep = $this->issue()->equipment()->irep() ;
        $list = array () ;
        $sql = "SELECT id,action_id,name,document_type,document_size,create_time,create_uid FROM {$irep->database}.issue_action_attachment WHERE action_id={$this->id()} ORDER BY create_time" ;
        $result = $irep->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepActionAttachment (
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
INSERT INTO {$this->issue()->equipment()->irep()->database}.issue_action_attachment
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
        $this->issue()->issue()->equipment()->irep()->query($sql) ;
        return $this->find_attachment_by_('id IN (SELECT LAST_INSERT_ID())') ;
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
