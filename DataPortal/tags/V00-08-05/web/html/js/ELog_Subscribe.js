define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ELog_Subscribe.css') ;

    /**
     * The application for viewing and managing subscriptions for events posted in the experimental e-Log
     *
     * @returns {ELog_Subscribe}
     */
    function ELog_Subscribe (experiment, access_list) {

        var that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.on_update() ;
        } ;

        this.on_deactivate = function() {
            this.init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this.init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this.is_initialized = false ;

        this.wa = null ;

        this.init = function () {

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html('<div id="elog-subscribe"></div>') ;
            this.wa = this.container.find('div#elog-subscribe') ;

            if (!this.access_list.elog.read_messages) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div class="elog-subscribe-user" id="subscribed" style="display:none;">' +
'  <h3>Your subscription:</h3>' +
'  <div style="padding-left:10px;">' +
'    <p align="justify">Your SLAC UNIX account <b>'+this.access_list.user.uid+'</b> is already subscribed to receive automated e-mail' +
'       notifications on various e-log events of this experiment. The notifications are sent' +
'       onto your SLAC e-mail address:</p>' +
'    <div style="padding-left: 10px;">' +
'      <b>'+this.access_list.user.email+'</b>' +
'      <button class="control-button" style="margin-left:10px;" id="unsubscribe" title="stop receiving automatic notifications">Unsubscribe</button>' +
'    </div>' +
'    <p align="justify">You may subscribe or unsubscribe at any time. You\'ll receive a confirmation' +
'       message shortly after unsubscribing.</p>' +
'  </div>' +
'</div>' +
'<div class="elog-subscribe-user" id="unsubscribed" style="display:none;">' +
'  <h3>Your subscription:</h3>' +
'  <div style="padding-left:10px;">' +
'    <p align="justify">At the moment, your SLAC UNIX account <b>'+this.access_list.user.uid+'</b> is not subscribed to receive automated' +
'       e-mail notifications on various e-log events of this experiment. If you choose to do so' +
'       then notifications will be sent onto your SLAC e-mail address:</p>' +
'    <div style="padding-left: 10px;">' +
'      <b>'+this.access_list.user.email+'</b>' +
'      <button class="control-button" style="margin-left:10px;" id="subscribe" title="start receiving automatic notifications">Subscribe</button>' +
'    </div>' +
'    <p align="justify">You may subscribe or unsubscribe at any time. You\'ll receive a confirmation' +
'       message shortly after subscribing. If your primary e-mail address differs from' +
'       the one mentioned above then make sure you set proper e-mail forwarding from' +
'       SLAC to your primary address. Also check if your SPAM filter won\'t be blocking' +
'       messages with the following properties:</p>' +
'    <div>' +
'      <table><tbody>' +
'        <tr><td class="table_cell table_cell_left" >From</td>' +
'            <td class="table_cell table_cell_right">LCLS E-Log [apache@slac.stanford.edu]</td></tr>' +
'        <tr><td class="table_cell table_cell_left  table_cell_bottom">Subject</td>' +
'            <td class="table_cell table_cell_right table_cell_bottom">[ '+this.experiment.instrument.name+' / '+this.experiment.name+' ]</td></tr>' +
'      </tbody></table>' +
'    </div>' +
'    <p align="justify">And here is the final remark: do not try to reply to e-log messages! Injecting' +
'       replies into e-Log stream via e-mail transport is not presently implemented.' +
'      We\'re still debating whether this would be a useful feature to have in the Portal.</p>' +
'  </div>' +
'</div>' +
'<div class="elog-subscribe-all"></div>' ;
            this.wa.html(html) ;

            this.wa.find('button#subscribe'  ).button().click(function () { that.subscription('SUBSCRIBE',   null) ; }) ;
            this.wa.find('button#unsubscribe').button().click(function () { that.subscription('UNSUBSCRIBE', null) ; }) ;

            this.subscription('CHECK', null) ;
        } ;

        this.subscription = function(operation, id) {
            var params = {exper_id: this.experiment.id, operation: operation} ;
            if (id) params.id = id ;
            Fwk.web_service_GET (
                '../logbook/ws/subscribe_check.php' ,
                params ,
                function (data) {
                    that.wa.find('#subscribed'  ).css('display', data.Subscribed ? 'block' : 'none' ) ;
                    that.wa.find('#unsubscribed').css('display', data.Subscribed ? 'none'  : 'block') ;
                    var html = '' ;
                    var all_subscriptions = data.AllSubscriptions ;
                    for (var i=0; i < all_subscriptions.length; i++) {
                        var s = all_subscriptions[i] ;
                        var extra_class = (i == all_subscriptions.length-1 ? 'table_cell_bottom' : '') ;
                        html +=
'  <tr><td class="table_cell '+extra_class+' table_cell_left" >'+s.address        +'</td>' +
'      <td class="table_cell '+extra_class+' "                >'+s.subscriber     +'</td>' +
'      <td class="table_cell '+extra_class+' "                >'+s.subscribed_host+'</td>' +
'      <td class="table_cell '+extra_class+' "                >'+s.subscribed_time+'</td>' +
'      <td class="table_cell '+extra_class+' table_cell_right"><button class="control-button" title="Stop receiving automated notifications" value='+s.id+'>Unsubscribe</button></td></tr>' ;
                }
                if (html !== '') {
                    html =
'<h3>All subscriptions for this e-Log:</h3>' +
'<table style="padding-left:10px;"><tbody>' +
'  <tr><td class="table_hdr">Recipient</td>' +
'      <td class="table_hdr">Subscribed by</td>' +
'      <td class="table_hdr">From host</td>' +
'      <td class="table_hdr">Date</td>' +
'      <td class="table_hdr">Actions</td></tr>' + html +
'</tbody></table>' ;
                }
                that.wa.find('.elog-subscribe-all').html(html) ;
                that.wa.find('.elog-subscribe-all').find('button').button().click(function () {
                    that.subscription('UNSUBSCRIBE', $(this).val()) ;
                }) ;
            } ,
            function (msg) {
                Fwk.report_error('Failed to contact the Web service to obtain e-Log notificaton subscriptions: <span style="color:red;">'+msg+'</span>.') ;
            }) ;
        };
    }
    Class.define_class (ELog_Subscribe, FwkApplication, {}, {}) ;

    return ELog_Subscribe ;
}) ;
