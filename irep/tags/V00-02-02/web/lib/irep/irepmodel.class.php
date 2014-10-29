<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepModel is an abstraction for models.
 *
 * @author gapon
 */
class IrepModel {

    /* Data members
     */
    private $manufacturer ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($manufacturer, $attr) {
        $this->manufacturer = $manufacturer ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function manufacturer () { return $this->manufacturer ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function description  () { return                  trim($this->attr['description']) ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }
    
    /* Operations
     */
    public function update_description ($description) {
        $description = trim($description) ;
        $description_escaped = $this->manufacturer()->irep()->escape_string($description) ;
        $sql = "UPDATE {$this->manufacturer()->irep()->database}.dict_model SET description='{$description_escaped}' WHERE id={$this->id()}" ;
        $this->manufacturer()->irep()->query($sql) ;
        $this->attr['description'] = $description ;
    }
    public function attachments () {
        $list = array () ;
        $sql = "SELECT id,model_id,rank,name,document_type,document_size,create_time,create_uid FROM {$this->manufacturer()->irep()->database}.dict_model_attachment WHERE model_id={$this->id()} ORDER BY rank DESC, create_time" ;
        $result = $this->manufacturer()->irep()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepModelAttachment (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function default_attachment () {
        $sql = "SELECT id,model_id,rank,name,document_type,document_size,create_time,create_uid FROM {$this->manufacturer()->irep()->database}.dict_model_attachment WHERE model_id={$this->id()} ORDER BY rank DESC, create_time LIMIT 1" ;
        $result = $this->manufacturer()->irep()->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepModelAttachment (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function add_default_attachment ($file, $uid) {
        $attachment       = $this->default_attachment() ;
        $rank             = is_null($attachment) ? 0 : $attachment->rank() + 1 ;
        $name_escaped     = $this->manufacturer()->irep()->escape_string(trim($file['description'])) ;
        $type_escaped     = $this->manufacturer()->irep()->escape_string(trim($file['type'])) ;
        $size             = intval($file['size']) ;
        $document_escaped = $this->manufacturer()->irep()->escape_string($file['contents']) ;
        $now_64           = LusiTime::now()->to64() ;
        $uid_escaped      = $this->manufacturer()->irep()->escape_string(trim($uid)) ;
        $sql =<<<HERE
INSERT INTO {$this->manufacturer()->irep()->database}.dict_model_attachment
  VALUES (
    NULL ,
    {$this->id()} ,
    {$rank} ,
    '{$name_escaped}' ,
    '{$type_escaped}' ,
    {$size} ,
    '{$document_escaped}' ,
    NULL ,
    {$now_64} ,
    '{$uid_escaped}'
  )
HERE;
        $this->manufacturer()->irep()->query($sql) ;
        return $this->default_attachment() ;
    }
    public function delete_attachment ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->manufacturer()->irep()->database}.dict_model_attachment WHERE id={$id}" ;
        $this->manufacturer()->irep()->query($sql) ;
    }
}
?>
