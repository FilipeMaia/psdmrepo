/**
 * A coolection of helper functions to interact with Web services.
 */
function web_service_GET (url, params, on_success, on_failure) {
    var jqXHR = $.get(url, params, function (data) {
        if (data.status != 'success') {
            if (on_failure) on_failure(data.message) ;
            else            report_error(data.message, null) ;
            return ;
        }
        if (on_success) on_success(data) ;
    },
    'JSON').error(function () {
        var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
        if (on_failure) on_failure(message) ;
        else            report_error(message, null) ;
    }) ;
} ;

function web_service_POST (url, params, on_success, on_failure) {
    var jqXHR = $.post(url, params, function (data) {
        if (data.status != 'success') {
            if (on_failure) on_failure(data.message) ;
            else            report_error(data.message, null) ;
            return ;
        }
        if (on_success) on_success(data) ;
    },
    'JSON').error(function () {
        var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
        if (on_failure) on_failure(message) ;
        else            report_error(message, null) ;
    }) ;
} ;


