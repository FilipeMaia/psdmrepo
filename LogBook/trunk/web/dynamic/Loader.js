/* The function returning an AJAX loader supported by the browser.
 */
function PageLoader() {

  var xmlHttp;
  try {
    xmlHttp=new XMLHttpRequest(); // Firefox, Opera 8.0+, Safari
    return xmlHttp;
  } catch (e) {
    try {
      xmlHttp=new ActiveXObject("Msxml2.XMLHTTP"); // Internet Explorer
      return xmlHttp;
    } catch (e) {
      try {
        xmlHttp=new ActiveXObject("Microsoft.XMLHTTP");
        return xmlHttp;
      } catch (e){
        alert("Your browser does not support AJAX!");
        return false;
      }
    }
  }
}

/* The function to be called to initate loading the requested (by URL)
 * page into a div specified via its identifier.
 */
function load(url,id){
  if(url==null) {
    document.getElementById(id).innerHTML='Loading...';
    return;
  }
  var ldr=PageLoader();
  ldr.onreadystatechange=function() {
    if(ldr.readyState==4)
      document.getElementById(id).innerHTML=ldr.responseText;
  }
  ldr.open('GET', url, true);
  ldr.send(null);
}

/* This is an extended version of the regular 'load()' function.
 * This one would also call a user defined action when the loading is done.
 * This may be needed to trigger additional actiions.
 */
function load_and_call(url,id, onSuccess){
  if(url==null) {
    document.getElementById(id).innerHTML='Loading...';
    return;
  }
  var ldr=PageLoader();
  ldr.onreadystatechange=function() {
    if(ldr.readyState==4) {
      document.getElementById(id).innerHTML=ldr.responseText;
      onSuccess();
    }
  }
  ldr.open('GET', url, true);
  ldr.send(null);
}

/* The function to be called to initate loading the requested (by URL)
 * page and if successfull - calling the specified function and passing the
 * results into it as a parameter. The result will be presented as
 * a JavaScript object (array).
 */
function load_then_call(url, onSuccess, onError){
  var ldr=PageLoader();
  ldr.onreadystatechange=function() {
    if(ldr.readyState==4) {
        if( ldr.status == 200)
            onSuccess( eval( "("+ldr.responseText+")" ));
        else
            onError( ldr.status );
    }
  }
  ldr.open('GET', url, true);
  ldr.send(null);
}
