/**
 * Web services API
 */
define ([] ,

function () {

    function _report_error (msg) {
        console.log(msg) ;
    }
    function _GET (url, params, on_success, on_failure) {
        var jqXHR = $.get(url, params, function (data) {
            if (on_success) on_success(data) ;
        },
        'JSON').error(function () {
            var msg = 'WebService.GET: request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(msg) ;
            else            _report_error(msg) ;
        }) ;
    }
    
    return {
        GET: _GET
    } ;
}) ;