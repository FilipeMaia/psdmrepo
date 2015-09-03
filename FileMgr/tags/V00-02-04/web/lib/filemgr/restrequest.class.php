<?php

/**
 * Class RestRequest is designed to make RESTful calls from PHP.
 * The code is based on ideas found at:
 *   http://www.gen-x-design.com/archives/making-restful-requests-in-php/
 */

namespace FileMgr;

require_once( 'filemgr.inc.php' );

/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit","64M");

class RestRequest {

    protected $base_url = 'https://pswww.slac.stanford.edu/ws-auth/irodsws';
    protected $url;
    protected $verb;
    protected $requestBody;
    protected $requestLength;
    protected $username = 'irodsws';
    protected $password = 'pcds';
    protected $acceptType;
    protected $responseBody;
    protected $responseInfo;

    public function __construct ($resource = null, $verb = 'GET', $requestBody = null, $postBody = false) {
        $this->url           = $this->base_url.$resource;
        $this->verb          = $verb;
        $this->requestLength = 0;
        $this->acceptType    = 'application/json';
        $this->responseBody  = null;
        $this->responseInfo  = null;

        if ($requestBody !== null) {
            if( $postBody ) {
                $this->requestBody = $requestBody;
                $this->buildPostBody();
            } else {
                $this->url .= '?'.http_build_query($requestBody, '', '&');
            }
        }
    }
    public function flush () {
        $this->requestBody   = null;
        $this->requestLength = 0;
        $this->verb          = 'GET';
        $this->responseBody  = null;
        $this->responseInfo  = null;
    }
    public function execute () {
        $ch = curl_init();
        $this->setAuth($ch);

        try {
            switch (strtoupper($this->verb)) {
                case 'GET'   : $this->executeGet   ($ch); break;
                case 'POST'  : $this->executePost  ($ch); break;
                case 'PUT'   : $this->executePut   ($ch); break;
                case 'DELETE': $this->executeDelete($ch); break;
                default:
                    throw new InvalidArgumentException('Current verb (' . $this->verb . ') is an invalid REST verb.');
            }
        } catch (InvalidArgumentException $e) { curl_close($ch); throw $e; }
          catch (Exception $e               ) { curl_close($ch); throw $e; }
    }
    public function buildPostBody ($data = null) {
        $data = ($data !== null) ? $data : $this->requestBody;

        if (!is_array($data))
            throw new InvalidArgumentException('Invalid data input for postBody.  Array expected');

        $data = http_build_query($data, '', '&');
        $this->requestBody = $data;
    }
    protected function executeGet  ($ch) { $this->doExecute($ch); }
    protected function executePost ($ch) {
        if (!is_string($this->requestBody)) $this->buildPostBody();

        curl_setopt($ch, CURLOPT_POSTFIELDS, $this->requestBody);
        curl_setopt($ch, CURLOPT_POST, 1);

        $this->doExecute($ch);    
    }
    protected function executePut ($ch) {
        if (!is_string($this->requestBody)) $this->buildPostBody();

        $this->requestLength = strlen($this->requestBody);

        $fh = fopen('php://memory', 'rw');
        fwrite($fh, $this->requestBody);
        rewind($fh);

        curl_setopt($ch, CURLOPT_INFILE, $fh);
        curl_setopt($ch, CURLOPT_INFILESIZE, $this->requestLength);
        curl_setopt($ch, CURLOPT_PUT, true);

        $this->doExecute($ch);

        fclose($fh);
    }
    protected function executeDelete ($ch) {
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, 'DELETE');
        $this->doExecute($ch);
    }
    protected function doExecute (&$curlHandle) {
        $this->setCurlOpts($curlHandle);
        $this->responseBody = curl_exec($curlHandle);
        $this->responseInfo = curl_getinfo($curlHandle);

        curl_close($curlHandle);
    }
    protected function setCurlOpts (&$curlHandle) {
        curl_setopt($curlHandle, CURLOPT_TIMEOUT, 10);
        curl_setopt($curlHandle, CURLOPT_URL, $this->url);
        curl_setopt($curlHandle, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($curlHandle, CURLOPT_HTTPHEADER, array ('Accept: ' . $this->acceptType));
    }
    protected function setAuth (&$curlHandle) {
        if ($this->username !== null && $this->password !== null) {
            curl_setopt($curlHandle, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
            curl_setopt($curlHandle, CURLOPT_USERPWD, $this->username . ':' . $this->password);
            curl_setopt($curlHandle, CURLOPT_SSL_VERIFYHOST, 2);
        }
    }
    public function getAcceptType   ()            { return $this->acceptType;                 }
    public function setAcceptType   ($acceptType) {        $this->acceptType   = $acceptType; }
    public function getPassword     ()            { return $this->password;                   }
    public function setPassword     ($password)   {        $this->password     = $password;   }
    public function getResponseBody ()            { return $this->responseBody;               }
    public function getResponseInfo ()            { return $this->responseInfo;               }
    public function getUrl          ()            { return $this->url;                        }
    public function setUrl          ($url)        {        $this->url          = $url;        }
    public function getUsername     ()            { return $this->username;                   }
    public function setUsername     ($username)   {        $this->username     = $username;   }
    public function getVerb         ()            { return $this->verb;                       }
    public function setVerb         ($verb)       {        $this->verb         = $verb;       }
}
?>
