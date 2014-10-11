define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader, Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ELog_Attachments.css') ;

    /**
     * The application for viewing message attachments in the experimental e-Log
     *
     * @returns {ELog_Attachments}
     */
    function ELog_Attachments (experiment, access_list) {

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

        this.on_update = function (sec) {
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

            this.container.html('<div id="elog-attachments"></div>') ;
            this.wa = this.container.find('div#elog-attachments') ;

            if (!this.access_list.elog.read_messages) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div id="ctrl">' +
'  <div style="float:right; margin-left:5px;"><button class="control-button" id="refresh" title="click to refresh the attachments list">Refresh</button></div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div id="body">' +
'  <div class="info" id="info" style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div style="margin-top:10px; font-size:80%;">' +
'    <table style="font-size:120%;"><tbody>' +
'      <tr>' +
'        <td><b>Sort by:</b></td>' +
'        <td><select name="sort" style="padding:1px;">' +
'              <option>posted</option>' +
'              <option>author</option>' +
'              <option>name</option>' +
'              <option>type</option>' +
'              <option>size</option>' +
'            </select></td>' +
'        <td><div style="width:20px;"></div></td>' +
'        <td><b>View as:</b></td>' +
'        <td><select name="view" style="padding:1px;">' +
'              <option>table</option>' +
'              <option>thumbnails</option>' +
'              <option>hybrid</option>' +
'            </select></td>' +
'        <td><div style="width:20px;"></div></td>' +
'        <td><b>Mix-in runs:</b></td>' +
'        <td><input name="runs" type="checkbox" '+(this.experiment.is_facility ? 'disabled' : '')+' /></td>' +
'        <td><div style="width:20px;"></div></td>' +
'        <td><button class="control-button" id="reverse">Show in Reverse Order</button></td>' +
'      </tr>' +
'    </tbody></table>' +
'  </div>' +
'  <div id="list"></div>' +
'</div>' ;
            this.wa.html(html) ;
            this.ctrl = this.wa.find('div#ctrl') ;
            this.body = this.wa.find('div#body') ;

            this.ctrl.find('button#refresh').button().click(function () {
                that.update_attachments() ;
            }) ;
            this.body.find('select[name="sort"]').change(function () {
                that.sort_attachments() ;
                that.display_attachments() ;
            }) ;
            this.body.find('select[name="view"]').change(function () {
                that.display_attachments() ;
            }) ;
            this.body.find('input[name="runs"]').change(function () {
                that.display_attachments() ;
            }) ;
            this.body.find('button#reverse').button().click(function() {
                that.attachments_last_request.reverse();
                that.display_attachments() ;
            });
            this.update_attachments() ;
        } ;


        this.attachments_last_request = null ;

        this.sort_attachments = function () {
            function compare_elements_by_posted (a, b) { return b.time64 - a.time64 ; }
            function compare_elements_by_author (a, b) { if (a.type !== 'a') return -1 ; if (b.type !== 'a') return 1 ; return ( a.e_author  < b.e_author ? -1 : (a.e_author > b.e_author ? 1 : 0 )) ; }
            function compare_elements_by_name   (a, b) { if (a.type !== 'a') return -1 ; if (b.type !== 'a') return 1 ; return ( a.a_name    < b.a_name   ? -1 : (a.a_name   > b.a_name   ? 1 : 0 )) ; }
            function compare_elements_by_type   (a, b) { if (a.type !== 'a') return -1 ; if (b.type !== 'a') return 1 ; return ( a.a_type    < b.a_type   ? -1 : (a.a_type   > b.a_type   ? 1 : 0 )) ; }
            function compare_elements_by_size   (a, b) { if (a.type !== 'a') return -1 ; if (b.type !== 'a') return 1 ; return   b.a_size    - a.a_size ; }
            var sort_function = null ;
            switch (this.body.find('select[name="sort"]').val()) {
            case 'posted' : sort_function = compare_elements_by_posted ; break ;
            case 'author' : sort_function = compare_elements_by_author ; break ;
            case 'name'   : sort_function = compare_elements_by_name ;   break ;
            case 'type'   : sort_function = compare_elements_by_type ;   break ;
            case 'size'   : sort_function = compare_elements_by_size ;   break ;
            }
            this.attachments_last_request.sort(sort_function) ;
        } ;

        function _a2link_url (a) {
            var title = 'show the message in the e-Log Search panel within the current Portal' ;
            var url =
'<a href="javascript:global_elog_search_message_by_id('+a.e_id+',true);" title="'+title+'" class="lb_link"><img src="../portal/img/link2message_32by32.png" /></a>' ;
            return url ;
        }

        function _a2url (a) {
            var title = 'show the message in the e-Log Search panel within the current Portal' ;
            var url =
'<a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank" title="'+title+'" class="lb_link">'+a.a_name+'</a>' ;
            return url ;
        }

        function _r2url (r, style) {
            var title = 'show the run in the e-Log Search panel within the current Portal' ;
            var img = style === 1 ?
                '../portal/img/link2run_32x32.png' :
                '../portal/img/link2run_f0f0f0_32x32.png' ;
            var url =
'<a href="javascript:global_elog_search_run_by_num('+r.r_num+',true);" title="'+title+'" class="lb_link"><img src="'+img+'" /></a>' ;
            return url ;
        }


        this.display_attachments_as_table = function () {
            var html =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_hdr" align="right" >Message ID</td>' +
'    <td class="table_hdr" align="center" >Posted</td>' +
'    <td class="table_hdr" align="right" >Author</td>' +
'    <td class="table_hdr" align="right" >Attachment Name</td>' +
'    <td class="table_hdr" align="right" >Type</td>' +
'    <td class="table_hdr" align="right" >Size</td>' +
'  </tr>' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '') ;
                var a = attachments[i] ;
                if (a.type !== 'a') continue ;
                html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left"><div style="float:left;">'+_a2link_url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.e_id+'</div><div style="clear:both;"></div></td>' +
'    <td class="table_cell '+extra_class+'">'+a.e_time  +'</td>' +
'    <td class="table_cell '+extra_class+'" align="right" >'+a.e_author+'</td>' +
'    <td class="table_cell '+extra_class+'" align="right" >'+_a2url(a)+'</td>' +
'    <td class="table_cell '+extra_class+'" align="right" >'+a.a_type+'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right" align="right" >'+a.a_size+'</td>' +
'  </tr>' ;
            }
            html +=
'</tbody></table>' ;
            return html ;
        } ;

        this.display_attachments_as_table_with_runs = function () {
            var html =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_hdr" align="right"  >Run</td>' +
'    <td class="table_hdr" align="center" >Started</td>' +
'    <td class="table_hdr" align="right"  >Message</td>' +
'    <td class="table_hdr" align="center" >Posted</td>' +
'    <td class="table_hdr" align="right"  >Author</td>' +
'    <td class="table_hdr" align="right"  >Attachment Name</td>' +
'    <td class="table_hdr" align="right"  >Type</td>' +
'    <td class="table_hdr" align="right"  >Size</td>' +
'  </tr>' ;
            var run_specific_style = 'style="background-color:#f0f0f0;"' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '') ;
                var a = attachments[i] ;
                if (a.type === 'a') {
                    html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left">'+'</td>' +
'    <td class="table_cell '+extra_class+' ">'+'</td>' +
'    <td class="table_cell '+extra_class+' "><div style="float:left;">'+_a2link_url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.e_id+'</div><div style="clear:both;"></div></td>' +
'    <td class="table_cell '+extra_class+' ">'+a.e_time+'</td>' +
'    <td class="table_cell '+extra_class+' " align="right" >'+a.e_author+'</td>' +
'    <td class="table_cell '+extra_class+' " align="right" >'+_a2url(a)+'</td>' +
'    <td class="table_cell '+extra_class+' " align="right" >'+a.a_type+'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right" align="right" >'+a.a_size+'</td>' +
'  </tr>' ;
                } else {
                    html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left" '+run_specific_style+'><div style="float:left;">'+_r2url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.r_num+'</div><div style="clear:both;"></div></td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+a.r_begin+'</td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+'</td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+'</td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+'</td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+'</td>' +
'    <td class="table_cell '+extra_class+'" '+run_specific_style+'>'+'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right" '+run_specific_style+'>'+'</td>' +
'  </tr>' ;
                }
            }
            html +=
'</tbody></table>' ;
            return html ;
        } ;

        this.display_attachments_as_thumbnail = function () {
            var html = '' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var a = attachments[i] ;
                if (a.type !== 'a') continue ;
                var title =
                    'name: '+a.a_name+'\n' +
                    'type: '+a.a_type+'\n' +
                    'size: '+a.a_size+'\n' +
                    'posted: '+a.e_time+'\n' +
                    'author: '+a.e_author ;
                html +=
'<div style="float:left; margin-left:10px;">' +
'  <div style="float:left; margin-left:10px;">'+_a2link_url(a)+'</div>' +
'  <div style="float:left;" title="'+title+'"><a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank"><img style="height:160px; padding:8px;" src="../logbook/attachments/preview/'+a.a_id+'" /></a></div>' +
'  <div style="clear:both;"></div>' +
'</div>' ;
            }
            return html ;
        } ;

        this.display_attachments_as_thumbnail_with_runs = function () {
            var html = '' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var a = attachments[i] ;
                if (a.type === 'a') {
                    var title =
                        'name: '+a.a_name+'\n' +
                        'type: '+a.a_type+'\n' +
                        'size: '+a.a_size+'\n' +
                        'posted: '+a.e_time+'\n' +
                        'author: '+a.e_author ;
                    html +=
'<div style="float:left; border-top:solid 1px #d0d0d0;">' +
'  <div style="float:left; margin-left:10px; margin-top:10px;">'+_a2link_url(a)+'</div>' +
'  <div style="float:left;" title="'+title+'"><a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank"><img style="height:160px; padding:8px;" src="../logbook/attachments/preview/'+a.a_id+'" /></a></div>' +
'  <div style="clear:both;"></div>' +
'</div>' ;
                } else {
                    var title =
                        'run #: '+a.r_num+'\n' +
                        'begin: '+a.r_begin+'\n' +
                        'end: '  +a.r_end ;
                    html +=
'<div style="float:left; height:160px; padding:8px; border-left:solid 1px #d0d0d0; border-top:solid 1px #d0d0d0; font-weight:bold;" title="'+title+'"><div style="padding-top:3px;">'+_r2url(a, 1)+'</div><div style="padding-top:40px; font-size:20px; font-weight:bold;">'+a.r_num+'</div></div>' ;
                }
            }
            return html ;
        } ;

        this.display_attachments_as_hybrid = function () {
            var html =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_hdr" align="center" >Info</td>' +
'    <td class="table_hdr">Attachment</td>' +
'  </tr>' ;
            var title = 'open the attachment in a separate tab' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '') ;
                var a = attachments[i] ;
                if (a.type !== 'a') continue ;
                var thumb =
'<div style="float:left;" title="'+title+'">' +
'  <a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank">' +
'    <img style="height:130px;" src="../logbook/attachments/preview/'+a.a_id+'"/>' +
'  </a>' +
'</div>' ;
                var info =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Host Message</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right"><div style="float:left;">'+_a2link_url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.e_id+'</div><div style="clear:both;"></div></td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Posted</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.e_time+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Author</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.e_author+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Attachment Name</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+_a2url(a)+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Type</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_type+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Size</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_size+'</td>' +
'  </tr>' +
'</tbody></table>' ;
                html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left">'+info +'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right">' +thumb+'</td>' +
'  </tr>' ;
            }
            html +=
'</tbody></table>' ;
            return html ;
        } ;

        this.display_attachments_as_hybrid_with_runs = function () {
            var run_specific_style = 'style="background-color:#f0f0f0;"' ;
            var html =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_hdr" align="right"  >Run</td>' +
'    <td class="table_hdr" align="center" >Start of Run</td>' +
'    <td class="table_hdr" align="center" >Context</td>' +
'    <td class="table_hdr"                >Attachment</td>' +
'  </tr>' ;
            var title = 'open the attachment in a separate tab' ;
            var attachments = this.attachments_last_request ;
            for (var i=0; i < attachments.length; i++) {
                var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '') ;
                var a = attachments[i] ;
                if (a.type === 'a') {
                    var thumb =
'<div style="float:left;" title="'+title+'">' +
'  <a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank">' +
'    <img style="height:130px;" src="../logbook/attachments/preview/'+a.a_id+'"/>' +
'  </a>' +
'</div>' ;
                    var info =
'<table><tbody>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Message ID</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right"><div style="float:left;">'+_a2link_url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.e_id+'</div><div style="clear:both;"></div></td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Posted</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.e_time+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Author</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.e_author+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Attachment Name</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right" >'+_a2url(a)+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Type</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_type+'</td>' +
'  </tr>' +
'  <tr>' +
'    <td class="table_cell table_cell_bottom table_cell_left" align="right" >Size</td>' +
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_size+'</td>' +
'  </tr>' +
'</tbody></table>' ;
                    html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left"></td>' +
'    <td class="table_cell '+extra_class+'"></td>' +
'    <td class="table_cell '+extra_class+'">'+info +'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right">' +thumb+'</td>' +
'  </tr>' ;
                } else {
                    html +=
'  <tr>' +
'    <td class="table_cell '+extra_class+' table_cell_left" '+run_specific_style+'><div style="float:left;">'+_r2url(a)+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+a.r_num+'</div><div style="clear:both;"></div></td>' +
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'+a.r_begin+'</td>' +
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>' +
'    <td class="table_cell '+extra_class+' table_cell_right" '+run_specific_style+'>'          +'</td>' +
'  </tr>' ;
                }
            }
            html +=
'</tbody></table>' ;
            return html ;
        } ;

        this.display_attachments = function () {
            var html = '';
            var display_name =
                this.body.find('select[name="view"]').val() +
                (this.body.find('input[name="runs"]').attr('checked') ? 'runs' : '') ;

            switch (display_name) {
                case 'table'         : html = this.display_attachments_as_table              () ; break ;
                case 'tableruns'     : html = this.display_attachments_as_table_with_runs    () ; break ;
                case 'thumbnails'    : html = this.display_attachments_as_thumbnail          () ; break ;
                case 'thumbnailsruns': html = this.display_attachments_as_thumbnail_with_runs() ; break ;
                case 'hybrid'        : html = this.display_attachments_as_hybrid             () ; break ;
                case 'hybridruns'    : html = this.display_attachments_as_hybrid_with_runs   () ; break ;
            }
            this.body.find('#list').html(html) ;
        } ;

        this.update_attachments = function () {
            this.body.find('#updated').html('Updating attachments...') ;
            Fwk.web_service_GET (
                '../logbook/ws/RequestAllAttachments.php' ,
                {exper_id: that.experiment.id} ,
                function (data) {
                    that.attachments_last_request = data.Attachments ;
                    that.sort_attachments() ;
                    that.display_attachments() ;
                    var num_attachments = 0 ;
                    for (var i=0; i < that.attachments_last_request.length; i++) {
                        var a = that.attachments_last_request[i] ;
                        if (a.type === 'a') num_attachments++ ;
                    }
                    that.body.find('#info').html('<b>'+num_attachments+'</b> attachments') ;
                    that.body.find('#updated').html('[ Last update on: <b>'+data.Updated+'</b> ]') ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;
    }
    Class.define_class (ELog_Attachments, FwkApplication, {}, {});

    return ELog_Attachments ;
}) ;
