define({
load: function (url) {
    var img = document.createElement("img") ;
    img.type = "image/png" ;
    img.href = url+"?bust="+new Date().getTime() ;
    document.getElementsByTagName("body")[0].appendChild(img) ;
}
}) ;



