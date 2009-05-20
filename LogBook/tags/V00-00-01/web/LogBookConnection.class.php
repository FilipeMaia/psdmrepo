<?php
class LogBookConnection {
    private $connected = FALSE;
    private $host;
    private $user;
    private $password;
    private $database;
    public function __construct($host, $user, $password, $database) {
        $this->host = $host;
        $this->user = $user;
        $this->password = $password;
        $this->database = $database;
    }
    public function __destruct() {
        mysql_close();
    }
    public function query( $sql ) {
        $this->connect();
        $result = mysql_query( $sql )
            or die('query failed: '.mysql_error().', in query: '.$sql);
        return $result;
    }
    private function connect() {
        if( !$this->connected ) {
            mysql_connect( $this->host, $this->user, $this->password )
                or die( "failed to connect to server" );
            mysql_select_db( $this->database )
                or die( "failed to select the database" );
            mysql_query( "SET SESSION SQL_MODE='ANSI'" )
                or die( "query failed: ".mysql_error());
            $this->connected = TRUE;
        }
    }
}
?>
