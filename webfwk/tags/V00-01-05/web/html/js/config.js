/*
 * An interface providing persistent support for run-time configurations
 * of Web applications.
 */
function config_handler_create (application_config, scope, parameter) {

    this.application_config = application_config ;

    this.scope     = scope ;
    this.parameter = parameter ;

    this.cached_value = null ;

    this.set_cached_value = function (value) {
        //alert('set:'+this.application_config.application+'/'+this.scope+'/'+this.parameter+':'+$.toJSON(value));
        this.cached_value = value ;
    } ;
    this.load = function (on_found, on_not_found) {
        if (this.cached_value != null) {
            //alert('load:'+this.application_config.application+'/'+this.scope+'/'+this.parameter+':'+$.toJSON(this.cached_value));
            return this.cached_value ;
        }
        this.application_config.load(this.scope, this.parameter, on_found, on_not_found, this) ;
        return null ;
    } ;
    this.save = function (value) {
        //alert('save:'+this.application_config.application+'/'+this.scope+'/'+this.parameter+':'+$.toJSON(value));
        this.cached_value = value ;
        this.application_config.save(this.scope, this.parameter, value) ;
    } ;
}

function config_create (application) {

    this.application = application ;

    this.cached_handlers = {} ;
    
    this.handler = function (scope, parameter) {
        if (!(scope     in this.cached_handlers))        this.cached_handlers[scope] = {} ;
        if (!(parameter in this.cached_handlers[scope])) this.cached_handlers[scope][parameter] = new config_handler_create(this, scope, parameter) ;
        return this.cached_handlers[scope][parameter] ;
    } ;

    this.load = function (scope, parameter, on_found, on_not_found, handler2update_on_found) {
        var url = '../webfwk/ws/config_load.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter
        } ;
        var jqXHR = $.get(url, params, function(data) {
            var result = eval(data) ;
            if (result.status != 'success') { report_error(result.message, null) ; return ; }
            if (result.found) {
                var value = eval('('+result.value.value+')') ;
                on_found(value) ;
                if (handler2update_on_found) handler2update_on_found.set_cached_value(value) ;
            } else {
                on_not_found() ;
            }
        } ,
        'JSON').error(function () {
            report_error('configuration loading failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;
    
    this.save = function (scope, parameter, value) {
        var url = '../webfwk/ws/config_save.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter ,
            value      : $.toJSON(value) //JSON.stringify(value)
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


