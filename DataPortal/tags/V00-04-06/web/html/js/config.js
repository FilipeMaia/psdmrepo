/*
 * An interface providing persistent support for run-time configurations
 * of Web applications.
 */
function config_handler_create (application_config, scope, parameter) {

    this.application_config = application_config ;

    this.scope     = scope ;
    this.parameter = parameter ;

    this.load = function (on_found, on_not_found) {
        this.application_config.load(this.scope, this.parameter, on_found, on_not_found) ;
    } ;
    this.save = function (value) {
        this.application_config.save(this.scope, this.parameter, value) ;
    } ;
}

function config_create (application) {

    this.application = application ;

    this.handler = function (scope, parameter) {
        return new config_handler_create(this, scope, parameter) ;
    } ;

    this.load = function (scope, parameter, on_found, on_not_found) {
        var url = '../portal/ws/config_load.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter
        } ;
        var jqXHR = $.get(url, params, function(data) {
            var result = eval(data) ;
            if (result.status != 'success') { report_error(result.message, null) ; return ; }
            if (result.found) on_found(eval('('+result.value.value+')')) ;
            else              on_not_found() ;
        } ,
        'JSON').error(function () {
            report_error('configuration loading failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;
    
    this.save = function (scope, parameter, value) {
        var url = '../portal/ws/config_save.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter ,
            value      : JSON.stringify(value)
        } ;
        var jqXHR = $.post(url, params, function(data) {
            var result = eval(data) ;
            if (result.status != 'success') { report_error(result.message, null) ; return ; }
        } ,
        'JSON').error(function () {
            report_error('configuration saving failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;
}


