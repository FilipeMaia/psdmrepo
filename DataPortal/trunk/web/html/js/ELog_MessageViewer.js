define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/Widget', 'webfwk/RadioBox', 'webfwk/StackOfRows', 'webfwk/Fwk' ,
    'portal/ELog_Utils'] ,

function (
    cssloader ,
    Class, Widget, RadioBox, StackOfRows, Fwk ,
    ELog_Utils) {

    cssloader.load('../portal/css/ELog_MessageViewer.css') ;

/**
 * Display a run within a table row's body.
 *
 * @param {Object} parent
 * @param {Array} message
 * @returns {ELog_RunBody}
 */
function ELog_RunBody (parent, message) {

    var that = this ;
   
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackOfRows.StackRowBody.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.message = message ;

    // ----------------------------
    // Static variables & functions
    // ----------------------------

    this.run_url = function () {
        var idx = window.location.href.indexOf('?') ;
        var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=run:'+message.run_num;
        var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
        return html ;
    } ;

    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this.is_rendered = false ;

    this.m_cont = null ;
    this.ctrl_cont = null ;
    this.message_cont = null ;
    this.dialogs_cont = null ;
    this.form_cont = null ;
    this.form_attachments_cont = null ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var html =
'<div class="m-cont">' ;

        //////////////////// CONTROL BUTTONS ////////////////
        
        html +=
'  <div class="ctrl">' +
'    <div class="url-cont">'+this.run_url()+'</div>' +
'    <div class="button-cont"><button class="control-button" name="print"  title="print this run info">P</button></div>' +
'    <div class="button-cont"><button class="control-button" name="reply"  title="post a message which will be associated with this run"><b>&crarr;</b></button></div>' +
'    <div class="button-cont-last"></div>' +
'  </div>' ;

        //////////////////// PARAMETERS AND ATTRIBUTES OF THIS RUN ////////////////

        html +=
'  <div class="message">Loading...</div>' +
'  <div class="dialogs"></div>' ;

        html +=
'</div>' ;

        //////////////////// RENDER AND SETUP HANDLERS ////////////////
        
        this.container.html(html) ;
        
        this.m_cont = this.container.children('.m-cont') ;

        this.ctrl_cont = this.m_cont.children('.ctrl') ;
        this.ctrl_cont.children('.button-cont').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'reply': that.reply() ; break ;
                default:
                    console.log('ELog_RunBody.render(): this operation is not supported: '+op) ; break ;
            }
        }) ;

        this.message_cont = this.m_cont.children('.message') ;
        this.load_parameters() ;

        this.dialogs_cont = this.m_cont.children('.dialogs') ;
    } ;

    this.parameters_view = null ;

    this.load_parameters = function () {
        Fwk.web_service_GET (
            '../logbook/ws/RequestRunParams.php' ,
            {run_id: this.message.run_id} ,
            function (data) {
                that.parameters_view = new StackOfRows.StackOfRows() ;
                for (var i in data.params) {
                    var section = data.params[i] ;
                    var section_body_html =
'<table><thead>' ;
                    for (var num = section.params.length, j = 0; j < num; j++) {
                        var extra_class = j === num - 1 ? 'table_cell_bottom' : '' ;
                        var param = section.params[j];
                        section_body_html +=
'  <tr>' +
'    <td class="table_cell table_cell_left  '+extra_class+'" style="font-weight:normal; font-style:italic;" >'+param.descr+'</td>' +
'    <td class="table_cell                  '+extra_class+'" style="color:maroon"                           >'+param.value+'</td>' +
'    <td class="table_cell table_cell_right '+extra_class+'"                                                >'+param.name +'</td>' +
'  </tr>' ;
                    }
                    section_body_html +=
'</thead></table>' ;
                    that.parameters_view.add_row({
                        title: '<b>'+section.title+'</b>' ,
                        body:  section_body_html}) ;
                }
                that.parameters_view.display(that.message_cont) ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            });
    } ;
    
    this.reply = function () {
        this.ctrl_cont.css('display', 'none') ;
        this.message_cont.css('display', 'none') ;
        var html =
'<div style="color:maroon;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'<div style="float:left; margin-right:20px;">' +
'  <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/message_new.php" method="post">' +
'    <input type="hidden" name="id" value="'+this.experiment.id+'" />' +
'    <input type="hidden" name="scope" value="run" />' +
'    <input type="hidden" name="message_id" value="" />' +
'    <input type="hidden" name="run_id" value="'+this.message.run_id+'" />' +
'    <input type="hidden" name="shift_id" value="" />' +
'    <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'    <input type="hidden" name="num_tags" value="0" />' +
'    <input type="hidden" name="onsuccess" value="" />' +
'    <input type="hidden" name="relevance_time" value="" />' +
'    <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>' +
'    <div style="margin-top: 10px;">' +
'      <div style="float:left;">' + 
'        <div style="font-weight:bold;">Author:</div>' +
'        <input type="text" name="author_account" value="'+this.access_list.user.uid+'" size=32 style="padding:2px; margin-top:5px;" />' +
'      </div>' +
'      <div style="float:left; margin-left:20px;">' + 
'        <div style="font-weight:bold;">Attachments:</div>' +
'        <div class="attachments">' +
'          <div>' +
'            <input type="file" name="file2attach_0" />' +
'            <input type="hidden" name="file2attach_0" value=""/ >' +
'          </div>' +
'        </div>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'  </form>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the message">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.reply_post() ;   break ;
                case 'cancel': that.reply_cancel() ; break ;
                default:
                    console.log('ELog_RunBody.reply(): this operation is not supported: '+op) ; break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () {
            that.add_attachment() ;
        }) ;
    } ;

    this.reply_post = function () {
        if (this.form_cont.find('textarea[name="message_text"]').val() === '') {
            Fwk.report_error('Can not post the empty message. Please put some text into the message box.') ;
            return ;
        }
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post the reply w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status !== 'success') {
                    Fwk.report_error(data.Message) ; return ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                }

                // NOTE: we don't bother to do anything because the live message veiwer
                // is supposed to automatically refresh itself and pick up any new
                // messages posted here or anywhere.

                that.reply_cancel() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;
    this.add_attachment = function () {
        var num = this.form_attachments_cont.find('div').size() ;
        this.form_attachments_cont.append (
'<div>' +
' <input type="file"   name="file2attach_'+num+'" />' +
' <input type="hidden" name="file2attach_'+num+'" value="" />' +
'</div>'
        ) ;
        this.form_attachments_cont.find('input:file[name="file2attach_'+num+'"]').change(function () { that.add_attachment() ; }) ;
    } ;
    this.reply_cancel = function () {
        this.ctrl_cont.css('display', 'block') ;
        this.message_cont.css('display', 'block') ;
        this.dialogs_cont.html('') ;
    } ;
}
Class.define_class (ELog_RunBody, StackOfRows.StackRowBody, {}, {}) ;

/**
 * Display a message within a table row's body.
 *
 * @param {Object} parent
 * @param {Array} message
 * @returns {ELog_MessageBody}
 */
function ELog_MessageBody (parent, message) {

    var that = this ;
   
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackOfRows.StackRowBody.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.message = message ;

    // ----------------------------
    // Static variables & functions
    // ----------------------------

    this.message_url = function () {
        var idx = window.location.href.indexOf('?') ;
        var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=message:'+this.message.id;
        var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
        return html ;
    } ;

    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this.is_rendered = false ;

    this.m_cont = null ;
    this.ctrl_cont = null ;
    this.message_cont = null ;
    this.attachments_cont = null ;
    this.tags_cont = null ;
    this.dialogs_cont = null ;
    this.form_cont = null ;
    this.form_attachments_cont = null ;

    this.children_viewer = null ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var id = this.message.id ;

        var html =
'<div class="m-cont">' ;


        //////////////////// CONTROL BUTTONS ////////////////
        
        html +=
  '<div class="ctrl view" >' +
    '<div class="url-cont" >'+this.message_url()+'</div>' ;
        if (this.parent.deleted) {
            ;
        } else {
            if (this.message.deleted) html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="undelete" ' +
              'title="undelete this message to allow othe operations" >undelete</button>' +
    '</div>' ;
            else {
                html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="email" ' +
              'title="forward this message by e-mail to specified recipients" ><b>@</b></button>' +
    '</div>' +
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
        'name="print" ' +
        'title="print this message and all its children if any" >P</button> ' +
    '</div>' +
    '<div class="button-cont" >' +
    '  <button class="control-button control-button-important" ' +
              'name="delete" ' +
              'title="delete this message and all its children if any" >X</button>' +
    '</div>' ;
                if (!this.message.parent_id) html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="tags" ' +
              'title="add more tags to the message" >+ tags</button>' +
    '</div>' ;
                html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="attachments" ' +
              'title="add more attachments to the message" >+ attach</button>' +
    '</div>' ;
                if (this.access_list.elog.edit_messages) {
                    if (!this.message.parent_id) {
                        if (this.message.run_id) html +=
    '<div class="button-cont" > ' +
      '<button class="control-button" ' +
              'name="dettach" ' +
              'title="attach this message to a run" >- run</button>' +
    '</div>' ;
                        else html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="attach" ' +
              'title="dettach this message from the run" >+ run</button>' +
    '</div>' ;
                    }
                    html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="edit"        title="edit the message text" >E</button></div>' ;
                }
                html +=
    '<div class="button-cont" >' +
      '<button class="control-button" ' +
              'name="reply" ' +
              'title="reply to the message" ><b>&crarr;</b></button>' +
    '</div>' ;
            }
        }
        html +=
    '<div class="button-cont-last"></div>' +
  '</div>' ;


        //////////////////// MESSAGE TEXT ////////////////

        html +=
'  <div class="message view"></div>' ;


        //////////////////// ATTACHMENTS ////////////////

        if (this.message.attachments_num) {
            html +=
'  <div class="attachments view">' ;
            var title = 'click to open the attachment in a separate tab' ;
            for (var i in this.message.attachments) {
                var attach = this.message.attachments[i] ;
                var url = '<a href="../logbook/attachments/'+attach.id+'/'+attach.description+'" target="_blank" title="'+title+'"><img src="../logbook/attachments/preview/'+attach.id+'"></a>' ;
                html +=
'    <div class="attach">' +
'      <div class="preview">' +
'        <a href="../logbook/attachments/'+attach.id+'/'+attach.description+'" target="_blank" title="'+title+'"><img src="../logbook/attachments/preview/'+attach.id+'"></a>' +
'      </div>' +
'      <div class="attrname">file:</div><div class="attrval">'+attach.description+'</div><div class="attrend"></div>' +
'      <div class="attrname">type:</div><div class="attrval">'+attach.type+'</div><div class="attrend"></div>' +
'      <div class="attrname">size:</div><div class="attrval">'+attach.size+'</div><div class="attrend"></div>' + 
'</div>' ;
            }
            html +=
'    <div class="attach-last"></div>' +
'  </div>' ;
        }


        //////////////////// TAGS ////////////////

        if (this.message.tags_num) {
            html +=
'  <div class="tags view"><b>keywords</b>: ' ;
            for (var i in this.message.tags) {
                var tag = this.message.tags[i] ;
                html += '<span>'+tag.tag+'&nbsp;</span>' ;
            }
            html +=
'  </div>' ;
        }


        //////////////////// CHILDREN ////////////////

        if (this.message.children_num) {
            html +=
'  <div class="children view"></div>' ;
        }

        html +=
'  <div class="dialogs edit"></div>' ;
'</div>' ;



        //////////////////// RENDER AND SETUP HANDLERS ////////////////
        
        this.container.html(html) ;
        
        this.m_cont = this.container.children('.m-cont') ;

        this.ctrl_cont = this.m_cont.children('.ctrl') ;
        this.ctrl_cont.children('.button-cont').children('.control-button') .button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'reply':       that.reply() ; break ;
                case 'edit':        that.edit_message() ; break ;
                case 'tags':        that.edit_tags() ; break ;
                case 'attachments': that.edit_attachments() ; break ;
                case 'attach':
                case 'dettach':     that.edit_run(op) ; break ;
                case 'delete':
                case 'undelete':    that.delete_message(op) ; break ;
                case 'email':       that.email_message() ; break ;
                default:
                    console.log('ELog_MessageBody.render(): this operation is not supported: '+op) ; break ;
            }
        }) ;

        this.message_cont = this.m_cont.children('.message') ;
        this.message_cont.html(this.message.html1) ;

        this.attachments_cont = this.m_cont.children('.attachments') ;
        this.tags_cont = this.m_cont.children('.tags') ;

        if (this.message.children_num) {
            var hidden_header = true ;
            var instant_expand = true ;
            this.children_viewer = new ELog_MessageViewer (
                this ,
                this.m_cont.children('.children') ,
                {   hidden_header  : hidden_header ,
                    instant_expand : instant_expand ,
                    deleted        : this.message.deleted || this.parent.deleted
                }
            ) ;
            this.children_viewer.load(this.message.children) ;
        }

        this.dialogs_cont = this.m_cont.children('.dialogs') ;

        this.enable_view() ;
    } ;

    /*
     * Switcher between viewing and editing modes.
     * 
     * Note that there is a couple of wrapper functions defined after this one.
     * Those are meant to provide a more explicit interface for th erest of the code.
     * 
     * @param {boolean} view
     * @returns {undefined}
     */
    this.view_vs_edit = function (view) {
        if (view) {
            this.dialogs_cont.html('') ;
            this.m_cont.parent().removeClass('editing') ;
        } else {
            this.m_cont.parent().addClass('editing') ;
        }
        this.m_cont.children('.view').css('display', view ? 'block' : 'none') ;
        this.m_cont.children('.edit').css('display', view ? 'none'  : 'block') ;
    } ;

    this.enable_view = function () { this.view_vs_edit(true) ; } ;
    this.enable_edit = function () { this.view_vs_edit(false) ; } ;


    /**
     * Create a dialog for replying to the current message.
     * 
     * @returns {undefined}
     */
    this.reply = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'  <div style="margin-top:20px; padding-left:10px;">' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/message_new.php" method="post">' +
'      <input type="hidden" name="id" value="'+this.experiment.id+'" />' +
'      <input type="hidden" name="scope" value="message" />' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="run_id" value="" />' +
'      <input type="hidden" name="shift_id" value="" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="0" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <input type="hidden" name="relevance_time" value="" />' +
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>' +
'      <div style="margin-top: 10px;">' +
'        <div style="float:left;">' + 
'          <div style="font-weight:bold;">Author:</div>' +
'          <input type="text" name="author_account" value="'+this.access_list.user.uid+'" size=32 style="padding:2px; margin-top:5px;" />' +
'        </div>' +
'        <div style="float:left; margin-left:20px;">' + 
'          <div style="font-weight:bold;">Attachments:</div>' +
'          <div class="attachments">' +
'            <div>' +
'              <input type="file" name="file2attach_0" />' +
'              <input type="hidden" name="file2attach_0" value=""/ >' +
'            </div>' +
'          </div>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'      <input type="hidden" name="return_parent" value="1" />' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the message">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.reply_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.reply(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;
    } ;

    this.reply_post = function () {
        if (this.form_cont.find('textarea[name="message_text"]').val() === '') {
            Fwk.report_error('Can not post the empty message. Please put some text into the message box.') ;
            return ;
        }
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post the reply w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status !== 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                var new_message = data.Entry ;
                that.parent.update_row(that, new_message) ;

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;
    
    /**
     * Create a dialog for editing the message text and adding more attachments.
     * 
     * @returns {undefined}
     */
    this.edit_message = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Edit message text. Note the total limit of <b>25 MB</b> for attachments.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/message_update.php" method="post">' +
'      <input type="hidden" name="id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <textarea name="content" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="font-weight:bold; margin-top: 10px;">Extra attachments:</div>'+
'      <div class="attachments">' +
'        <div>' +
'          <input type="file" name="file2attach_0" />' +
'          <input type="hidden" name="file2attach_0" value=""/ >' +
'        </div>' +
'      </div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_message_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_message(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;

        this.form_cont.find('textarea[name="content"]').val(this.message.content) ;
    } ;
    this.edit_message_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.status !== 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // Refresh the current message tree

                var new_message = data.Entry ;
                that.parent.update_row(that, new_message) ;

                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    /**
     * Create a dialog for adding more tags to the current message.
     * 
     * @returns {undefined}
     */
    this.edit_tags = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Select additional tags or add the new ones.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/message_extend.php" method="post">' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="'+ELog_Utils.max_num_tags+'" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <div style="font-weight:bold;">Tags:</div>' +
'      <div class="tags"></div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_tags_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_tags(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        ELog_Utils.load_tags_and_authors (
            this.experiment.id ,
            this.form_cont.find('.tags') ,
            function (tags) {
                // TODO: enable all inputs on the page which can't be used before
                // the tags information is ready.
                ;
            } ,
            function (msg)  {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;
    this.edit_tags_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.status !== 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // Extend the message and refresh the current message tree

                var tags = data.Extended.tags ;
                for (var i in tags) {
                    that.message.tags.push(tags[i]) ;
                    that.message.tags_num++ ;
                }
                that.parent.update_row(that, that.message) ;
                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    /**
     * Create a dialog for adding more attachments the current message.
     * 
     * @returns {undefined}
     */
    this.edit_attachments = function () {

        this.enable_edit() ;

        var html =
'<div style="float:left; margin-right:20px; padding-top:5px;">' +
'  <div style="color:maroon;">Select attachments to upload. Note the total limit of <b>25 MB</b>.</div>' +
'  <div style="margin-top:20px; padding-left:10px;" >' +
'    <form id="'+this.message.id+'" enctype="multipart/form-data" action="../logbook/ws/message_extend.php" method="post">' +
'      <input type="hidden" name="message_id" value="'+this.message.id+'" />' +
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'      <input type="hidden" name="num_tags" value="0" />' +
'      <input type="hidden" name="onsuccess" value="" />' +
'      <div style="font-weight:bold;">Attachments:</div>' +
'      <div class="attachments">' +
'        <div>' +
'          <input type="file" name="file2attach_0" />' +
'          <input type="hidden" name="file2attach_0" value=""/ >' +
'        </div>' +
'      </div>' +
'    </form>' +
'  </div>' +
'</div>' +
'<div class="button-cont-left"><button class="control-button" name="post"   title="post the modifications">post</button></div>' +
'<div class="button-cont-left"><button class="control-button" name="cancel" title="cancel the operation and close this dialog">cancel</button></div>' +
'<div class="button-cont-left-last"></div>' ;
        this.dialogs_cont.html(html) ;
        this.dialogs_cont.children('.button-cont-left').children('.control-button').button().click(function () {
            var op = this.name ;
            switch (op) {
                case 'post'  : that.edit_attachments_post() ;  break ;
                case 'cancel': that.enable_view() ; break ;
                default: 
                    console.log('ELog_MessageBody.edit_attachments(): this operation is not supported: '+op) ;
                    that.enable_view() ;
                    break ;
            }
        }) ;
        this.form_cont = this.dialogs_cont.find('form#'+this.message.id) ;
        this.form_attachments_cont = this.form_cont.find('.attachments') ;
        this.form_attachments_cont.find('input:file[name="file2attach_0"]').change(function () { that.add_attachment() ; }) ;
    } ;
    this.edit_attachments_post = function () {

        this.dialogs_cont.children('.button-cont-left').children('.control-button').button('disable') ;

        // Use JQuery AJAX Form plug-in to post w/o reloading
        // the current page.
        //
        this.form_cont.ajaxSubmit ({
            success: function (data) {
                if (data.Status !== 'success') {
                    Fwk.report_error(data.Message) ;
                    this.dialogs_cont.children('.button-cont-left').children('.control-button').button('enable') ;
                    return ;
                }

                // Extend the message and refresh the current message tree

                var attachments = data.Extended.attachments ;
                for (var i in attachments) {
                    that.message.attachments.push(attachments[i]) ;
                    that.message.attachments_num++ ;
                }
                that.parent.update_row(that, that.message) ;
                that.enable_view() ;
            } ,
            complete: function () { } ,
            dataType: 'json'
        });
    } ;

    this.edit_run = function (operation) {
        var params = {id: this.message.id, scope: 'run'} ;
        switch (operation) {

            case 'attach':

                Fwk.form_dialog (
                    'Attaching message to a run' ,
                    '<b>Run<b> <input type="text" size=2 val=""/>' ,
                    function (form) {
                        var run_num = parseInt(form.find('input').val()) ;
                        if (!run_num) {
                            Fwk.report_error('Illegal run number. Please correct or cancel the operation.' ) ;
                            return false ;  // keep the dialog open
                        }
                        params.run_num = run_num ;
                        that.edit_run_submit(params) ;

                        return true ;  // close the dialog
                    }
                ) ;
                break ;

            case 'dettach':

                Fwk.ask_yes_no (
                    'Dettaching message from the run' ,
                    'Are you sure you want to dettach this message from run: <b>'+this.message.run_num+'</b>' ,
                    function () {
                        that.edit_run_submit(params) ;
                    }
                ) ;
                break ;
        }
    } ;
    this.edit_run_submit = function (params) {
        Fwk.web_service_GET (
            '../logbook/ws/message_move.php' ,
            params ,
            function (data) {

                // Extend the message and refresh the current message tree

                that.message.run_id = data.run_id ;
                that.message.run_num = data.run_num ;

                that.parent.update_row(that, that.message) ;
                that.enable_view() ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;

    this.delete_message = function (operation) {
        switch (operation) {
            case 'undelete':
                this.delete_message_submit(operation) ;
                break ;
            case 'delete':
                Fwk.ask_yes_no (
                    'Information Deletion Warning' ,
                    '<span style="color:red;">You have requested to delete the selected message. Are you sure?</span>' ,
                    function () {
                        that.delete_message_submit(operation) ;
                    }
                ) ;
                break ;
        }
    } ;
    this.delete_message_submit = function (operation) {
        Fwk.web_service_GET (
            '../logbook/ws/message_delete.php' ,
            {   id:        this.message.id ,
                operation: operation
            } ,
            function (data) {
                switch (operation) {
                    case 'undelete': that.parent.undelete_row(that) ; break ;
                    case 'delete':   that.parent.delete_row(that, data.deleted_time, data.deleted_by) ; break ;
                }
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;

    /**
     * This method is used to add more attachments to the form found at
     * the following editing dialogs:
     * 
     * - posting a reply to the message
     * - editing the message
     * - adding more attachments to the message
     * 
     * @returns {undefined}
     */
    this.add_attachment = function () {
        var num = this.form_attachments_cont.find('div').size() ;
        this.form_attachments_cont.append (
'<div>' +
' <input type="file"   name="file2attach_'+num+'" />' +
' <input type="hidden" name="file2attach_'+num+'" value="" />' +
'</div>'
        ) ;
        this.form_attachments_cont.find('input:file[name="file2attach_'+num+'"]').change(function () { that.add_attachment() ; }) ;
    } ;
    
    this._PREDEFINED_EMAIL_ADDRESSES = [
        {   recipient: 'pcds-help',  email: 'pcds-help@slac.stanford.edu',  descr: 'PCDS General Support'} ,
        {   recipient: 'pcds-it-l',  email: 'pcds-it-l@slac.stanford.edu',  descr: 'PCDS IT Infrastructure Support'} ,
        {   recipient: 'pcds-poc-l', email: 'pcds-poc-l@slac.stanford.edu', descr: 'LCLS Points Of Contacts Crew'} ,
        {   recipient: 'pcds-daq-l', email: 'pcds-daq-l@slac.stanford.edu', descr: 'LCLS DAQ Crew'} ,
        {   recipient: 'pcds-ana-l', email: 'pcds-ana-l@slac.stanford.edu', descr: 'LCLS Data Analysis & Data Management Crew'} ,
        {   recipient: 'self',       email: this.access_list.user.email,    descr: 'My SLAC address: <b>'+this.access_list.user.email+'</b>'}
    ] ;
    
    /**
     * Initiate a dialog for worwarding a select message to 
     * @returns {undefined}
     */
    this.email_message = function () {
        Fwk.form_dialog (
'Forwarding message via e-Mail gateway' ,
'<div style="padding-top: 5px;" > ' +
'  Select any pre-defined mailing list, your own SLAC address, or fill ' +
'  in arbitrary e-mail addresses of persons you want to share the content of this e-log entry. ' +
'</div>' +
'<div style="padding-top: 5px;" > ' +
'  <b>NOTE</b>: your recipients must have a valid UNIX account at SLAC in order to access attachments ' +
'  of the message entry.' +
'</div>' +
'<table><tbody> ' +
'  <tr> ' +
'    <td colspan="0" >&nbsp;</td> ' +
'  </tr> ' + _.reduce(this._PREDEFINED_EMAIL_ADDRESSES, function (html, e) { return html +=
'  <tr class="email selectable" > ' +
'    <td class="selector"                             ><input type="checkbox" /></td> ' +
'    <td class="recipient" style="font-weight:bold;"  >'+e.recipient+'</td> ' +
'    <td class="descr"     style="padding-left:10px;" >'+e.descr+'</td> ' +
'    <td class="email"                                ><input type="hidden" value="'+e.email+'" /></td> ' +
'  </tr> ' ; }, '') +
'  <tr> ' +
'    <td colspan="0" >&nbsp;</td> ' +
'  </tr> ' + _.reduce([1,2,3], function (html, idx) { return html +=
'  <tr class="email" > ' +
'    <td >&nbsp;</td> ' +
'    <td                   style="font-style:italic;" >e-mail:</td> ' +
'    <td class="email"     style="padding-left:10px;" ><input type="text" style="width:100%" title="full e-mail address of a recipient" value="" /></td> ' +
'  </tr> ' ; }, '') +
'</tbody></table> ' ,
            function (form) {
                var recipients = _.reduce (
                    $.makeArray(form.find('tr.email').children('td.email').children('input')) ,
                    function (recipients, input) {
                        var input = $(input) ;
                        var tr = input.parent().parent('tr.selectable') ;
                        if (tr.length) {
                            if (tr.children('td.selector').children('input').get(0).checked) {
                                var email = tr.children('td.email').children('input').val() ;
                                if (email) recipients.push(email) ;
                            }  
                        } else {
                            var email = input.val() ;
                            if (email) recipients.push(email) ;
                        }
                        return recipients ;
                    } ,
                    []
                ) ;
                if (!recipients.length) {
                    Fwk.report_error('Please, provide at least one e-mail address or cancel the operation.' ) ;
                    return false ;  // keep the dialog open
                }
                console.log('ELog_MessageBody.email_message() recipients:', recipients) ;
                Fwk.web_service_POST (
                    '../logbook/ws/message_forward.php' ,
                    {   id:         that.message.id ,
                        recipients: JSON.stringify(recipients)
                    } ,
                    function (data) { /* nothing to do on the successfull completion */ } ,
                    function (msg)  { Fwk.report_error(msg) ; }
                );
                return true ;   // close the dialog
            }
        ) ;
    } ;
}
Class.define_class (ELog_MessageBody, StackOfRows.StackRowBody, {}, {}) ;


/**
 * The base class for displaying a stack of messages witin a table row's body
 *
 * @param {Object} parent
 * @param {Array} messages
 * @param {Object} options
 * @param {String} id
 * @returns {ELog_MessageGroupBody}
 */
function ELog_MessageGroupBody (parent, messages, options, id) {
   
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackOfRows.StackRowBody.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent  = parent ;
    this.options = jQuery.extend(true, {}, options) ;
    this.id      = id;

    this.experiment  = parent.experiment ;
    this.access_list = parent.access_list ;

    this._messages = messages ;

    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this._viewer = null ;
    this.is_rendered = false ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var html =
'<div id="'+this.id+'" class="group-viewer" ></div>' ;
        this.container.html(html) ;

        this._viewer = new ELog_MessageViewerNoGroupping(this.parent, this.options) ;
        this._viewer.display(this.container.children('div#'+this.id)) ;
        this._viewer.load(this._messages) ;
    } ;
    
    this.insert_front = function (m) {
        if (this._messages.length) this._messages.splice(0, 0, m) ;
        else                       this._messages.push(m) ;
        if (this.is_rendered)
            this._viewer.update([m]) ;
    } ;
}
Class.define_class (ELog_MessageGroupBody, StackOfRows.StackRowBody, {}, {}) ;


/**
 * This class provides an interface for implementing the viewer proxy (class ELog_MessageViewer)
 * and various type sof viewers, such as: ELog_MessageViewerNoGroupping,
 * ELog_MessageViewerGroupByDay,  ELog_MessageViewerGroupByShift, etc.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerBase}
 */
function ELog_MessageViewerBase (parent, options) {

    // Always call the c-tor of the base class

    Widget.Widget.call(this) ;

    // -- parameters

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;

    // -- options

    this.hidden_header  = false ;
    this.instant_expand = false ;
    this.deleted        = false ;
    this.allow_groups   = false ;
    this.allow_runs     = true ;    // -- do not display anyting related to runs if false
    this.allow_shifts   = true ;    // -- do not display anyting related to shifts if false
    this.no_ymd         = false ;   // -- do not display the data in timestamps if true

    this.options = options || {} ;

    for (var opt in this.options) {
        var val = this.options[opt] ;
        this[opt] = val ? true : false ;
    }   

    // -- methods to be implemented by the base classes

    this.messages = function () {
        throw new Widget.WidgetError('ELog_MessageViewerBase::messages() the implementation must be provided by a derived class') ;
    } ;
    this.num_rows = function () {
        throw new Widget.WidgetError('ELog_MessageViewerBase::num_rows() the implementation must be provided by a derived class') ;
    } ;
    this.num_runs = function () {
        throw new Widget.WidgetError('ELog_MessageViewerBase::num_runs() the implementation must be provided by a derived class') ;
    } ;
    this.min_run  = function () {
        throw new Widget.WidgetError('ELog_MessageViewerBase::min_run() the implementation must be provided by a derived class') ;
    } ;
    this.max_run  = function () {
        throw new Widget.WidgetError('ELog_MessageViewerBase::max_run() the implementation must be provided by a derived class') ;
    } ;

    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::load() the implementation must be provided by a derived class') ;
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::update() the implementation must be provided by a derived class') ;
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::append() the implementation must be provided by a derived class') ;
    } ;

    /**
     * This handler is meant to be used by row (bodies) to report that their
     * content has been updated.
     * 
     * The method will update its local data storage (for the message content)
     * and redisplay the message tree.
     * 
     * @param StackRow old_row
     * @param Array new_message
     * @returns {undefined}
     */
    this.update_row = function (old_row, new_message) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::update_row() the implementation must be provided by a derived class') ;
    } ;

    this.undelete_row = function (row) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::undelete_row() the implementation must be provided by a derived class') ;
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::delete_row() the implementation must be provided by a derived class') ;
    } ;

    this.focus_at_message = function (message_id) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::focus_at_message() the implementation must be provided by a derived class') ;
    } ;

    this.focus_at_run = function (run_num) {
        throw new Widget.WidgetError('ELog_MessageViewerBase::focus_at_run() the implementation must be provided by a derived class') ;
    } ;

    /**
     * Produce a data object to be fed into the StackOfRows as a row
     * 
     * @param object m 
     * @returns array of data objects
     */
    this._message2row = function (m) {
        var posted = this.no_ymd ? m.hms : '<b>'+m.ymd+'</b>&nbsp;&nbsp;'+m.hms ;
        var row = {
            title: {
                posted: posted ,
                author: m.author ,
                subj:  '&nbsp;' ,
                id: '&nbsp;' ,
                attach: '&nbsp;' ,
                child: '&nbsp;' ,
                tag: '&nbsp;'
            } ,
            body: ''
        } ;
        if (this.allow_runs) {
            row.title.run      = m.run_num ? '<div class="m-run">'+m.run_num+'</div>' : '&nbsp;' ;
            row.title.duration = '&nbsp;' ;
        }
        if (m.is_run) {
            if (!this.allow_runs) {
                throw new Widget.WidgetError('ELog_MessageViewerBase::_message2row() the runs are not supported by the current configuration of the viewer') ;
            }
            row.title.run  = '<div class="m-run">'+m.run_num+'</div>' ;
            row.title.subj = this._run2subj(m) ;
            if (m.type !== 'begin_run') { row.title.duration = m.duration1 ; }
            row.body                = new ELog_RunBody(this, m) ;
            row.color_theme         = 'stack-theme-green' ;
            row.block_common_expand = true ;
        } else {
            row.title.subj = m.deleted ? '<span class="m-subj-deleted">'+m.subject+'</span>'+' <span class="m-subj-notes">deleted by <b>'+m.deleted_by+'</b> [ '+m.deleted_time+' ]</span>': m.subject ;
            row.title.id   = '<div class="m-id">'+m.id+'</div>' ;
            if (m.attachments_num) row.title.attach = '<img src="../portal/img/attachment.png" height="18">' ;
            if (m.children_num)    row.title.child = '<sup><b>&crarr;</b></sup>' ;
            if (m.tags_num)        row.title.tag = '<sup><b>T</b></sup>' ;
            row.body = new ELog_MessageBody(this, m) ;
        }
        return row ;
    } ;
    this._run2subj = function (m) {
        if (!this.allow_runs) {
            throw new Widget.WidgetError('ELog_MessageViewerBase::_run2subj() the runs are not supported by the current configuration of the viewer') ;
        }
        switch (m.type) {
            case 'run'       : return '<b>stop</b>';
            case 'end_run'   : return '<b>stop</b>' ;
            case 'begin_run' : return '<b>start</b>' ;
        }
        return '' ;
    } ;
}
Class.define_class (ELog_MessageViewerBase, Widget.Widget, {}, {}) ;



/**
 * Display a simple stream of messages and runs w/o any groupping.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerNoGroupping}
 */
function ELog_MessageViewerNoGroupping (parent, options) {

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // -- parameters

    this._messages = [] ;
    var hdr = [(this.no_ymd ? 
        {id: 'posted',   title: 'Time',      width:  80} :
        {id: 'posted',   title: 'Posted',    width: 150})
    ] ;
    if (this.allow_runs) hdr.push (
        {id: 'run',      title: 'Run',       width:  34, align: 'right'} ,
        {id: 'duration', title: 'Length',    width:  55, align: 'right', style: 'color:maroon;'}
    ) ;
    hdr.push (
        {id: 'attach',   title: '&nbsp',     width:  16} ,
        {id: 'tag',      title: '&nbsp;',    width:  16} ,
        {id: 'child',    title: '&nbsp;',    width:  20} ,
        {id: 'subj',     title: 'Subject',   width:   0} ,
        {id: '>'} ,
//        {id: 'id',       title: 'MessageId', width:  70} ,
        {id: 'author',   title: 'Author',    width:  90}
    ) ;
    this._table = new StackOfRows.StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            }
        }
    ) ;

    this._is_rendered = false ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {

        if (this._is_rendered) return ;
        this._is_rendered = true ;

        this._table.display(this.container) ;
        if (this.instant_expand) this._table.expand_or_collapse(true) ;
    } ;

    this.messages = function () { return this._messages ; } ;

    this.num_rows = function () { return this._table.num_rows() ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; } ;
    this.min_run  = function () { return this._min_run; } ;
    this.max_run  = function () { return this._max_run; } ;


    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this._messages = jQuery.extend(true, [], messages) ;
        this._messages.reverse() ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        this._table.reset() ;

        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this._messages[i] = m ;
            }
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            }
            this._table.append(this._message2row(m)) ;
        }
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                      this._messages.push(m) ;
                this._table.insert_front(this._message2row(m)) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages iat the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                      this._messages.push(m) ;
                this._table.append(this._message2row(m)) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
            }
        }
    } ;

    /**
     * This handler is meant to be used by row (bodies) to report that their
     * content has been updated.
     * 
     * The method will update its local data storage (for the message content)
     * and redisplay the message tree.
     * 
     * @param StackRow old_row
     * @param Array new_message
     * @returns {undefined}
     */
    this.update_row = function (old_row, new_message) {
        if (typeof new_message === 'string') new_message = eval('('+new_message+')') ;
        var id = old_row.message.id ;
        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (id === m.id) {
                this._messages[i] = new_message ;
                this._table.update_row (
                    old_row.row_id ,
                    this._message2row(new_message)) ;
                return ;
            }
        }
    } ;

    this.undelete_row = function (row) {
        var id = row.message.id ;
        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (id === m.id) {
                m.deleted = 0 ;
                m.deleted_time = '' ;
                m.deleted_by = '' ;
                this._table.update_row (
                    row.row_id ,
                    this._message2row(m)) ;
                return ;
            }
        }
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        var id = row.message.id ;
        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (id === m.id) {
                m.deleted = 1 ;
                m.deleted_time = deleted_time ;
                m.deleted_by = deleted_by ;
                this._table.update_row (
                    row.row_id ,
                    this._message2row(m)) ;
                return ;
            }
        }
    } ;

    this.focus_at_message = function (message_id) {
        for (var i in this._table.rows) {
            var row = this._table.rows[i] ;
            if (row.data_object.body.message.id == message_id) {
                var id = row.id ;
                this._table.expand_or_collapse_row(id, true, $('#fwk-center')) ;
                return ;
            }
        }
        console.log('ELog_MessageViewerNoGroupping.focus_at_message('+message_id+') the message was not found in StackOfRows') ;
    } ;

    this.focus_at_run = function (run_num) {
        for (var i in this._table.rows) {
            var row = this._table.rows[i] ;
            if (row.data_object.body.message.is_run && row.data_object.body.message.run_num == run_num) {
                var id = row.id ;
                this._table.expand_or_collapse_row(id, true, $('#fwk-center')) ;
                return ;
            }
        }
        console.log('ELog_MessageViewerNoGroupping.focus_at_run('+run_num+') the message was not found in StackOfRows') ;
    } ;
}
Class.define_class (ELog_MessageViewerNoGroupping, ELog_MessageViewerBase, {}, {}) ;


/**
 * Specialization for displaying a conatiner of mesasages within a table row's body
 *
 * @param {Object} parent
 * @param {Array} messages
 * @param {Object} options
 * @returns {ELog_DayBody}
 */
function ELog_DayBody (parent, messages, options) {
    var options = jQuery.extend(true, {no_ymd: true}, options) ;
    ELog_MessageGroupBody.call(this, parent, messages, options, 'day-viewer') ;
}
Class.define_class (ELog_DayBody, ELog_MessageGroupBody, {}, {}) ;

/**
 * Group messages by a day they were posted.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerByDay}
 */
function ELog_MessageViewerByDay (parent, options) {

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // -- parameters

    this._messages = [] ;
    this._days = {} ;

    var hdr = [
        {id: 'day',            title: 'Day',      width: 80} ,
        {id: 'runs-end',       title: 'Runs',     width: 60, align: 'right'} ,
        {id: 'runs-separator', title: '&nbsp;',   width: 10, align: 'center'} ,
        {id: 'runs-begin' ,    title: '&nbsp;',   width: 60, align: 'left'} ,
        {id: 'messages',       title: 'Messages', width: 40, align: 'right'}
    ] ;
    this._table = new StackOfRows.StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            } ,
            theme: 'stack-theme-aliceblue'
        }
    ) ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {
        this._table.display(this.container) ;
        if (this.instant_expand) this._table.expand_or_collapse(true) ;
    } ;

    this.messages = function () { return this._messages ; } ;

    this.num_rows = function () { return this._messages.length ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; } ;
    this.min_run  = function () { return this._min_run; } ;
    this.max_run  = function () { return this._max_run; } ;


    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this._messages = jQuery.extend(true, [], messages) ;
        this._messages.reverse() ;

        this._days = {} ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this._messages[i] = m ;
            }
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            }
            var ymd = m.ymd ;
            if (!_.has(this._days, ymd)) {
                this._days[ymd] = {
                    messages: []
                } ;
            }
            this._days[ymd].messages.push(m) ;
        }
        var ymds = _.keys(this._days) ;
        ymds.sort() ;
        ymds.reverse() ;

        this._table.reset() ;

        for (var i in ymds) {
            var ymd = ymds[i] ;
            this._days[ymd].messages.reverse() ;
            this._days[ymd].row_id = this._table.append(this._day2row(ymd)) ;
        }
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            var days = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
                var ymd = m.ymd ;
                if (!_.has(days, ymd)) {
                    days[ymd] = {
                        messages: []
                    } ;
                }
                days[ymd].messages.push(m) ;
            }

            var ymds = _.keys(days) ;
            ymds.sort() ;
            for (var i in ymds) {

                var ymd = ymds[i] ;
                if (_.has(this._days, ymd)) {

                    // Extend the list of messages at the day
                    var messages = days[ymd].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._days[ymd].messages.push(m) ;
                    }

                    // Update the title of the day
                    var row = this._table.get_row_by_id(this._days[ymd].row_id) ;
                    row.update_title(this._day2row_title(ymd), function (hdr_cont) {
                        hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
                    }) ;
                    
                    // Update the list of messages within the row's body
                    var row_body = row.get_body() ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        row_body.insert_front(m) ;
                    }
                } else {
                    
                    // Create a new row for the day
                    this._days[ymd] = days[ymd] ;
                    this._days[ymd].row_id = this._table.insert_front(this._day2row(ymd)) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages at the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            var days = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }            
                var ymd = m.ymd ;
                if (!_.has(days, ymd)) {
                    days[ymd] = {
                        messages: []
                    } ;
                }
                days[ymd].messages.push(m) ;
            }
            var ymds = _.keys(days) ;
            ymds.sort() ;
            ymds.reverse() ;   
            for (var i in ymds) {
                var ymd = ymds[i] ;
                if (_.has(this._days, ymd)) {
                    var day = this._days[ymd] ;
                    var messages = days[ymd].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        day.messages.push(m) ;
                    }
                    day.row_id = this._table.update_row(day.row_id, this._day2row(ymd)) ;
                } else {
                    this._days[ymd] = days[ymd] ;
                    this._days[ymd].row_id = this._table.append(this._day2row(ymd)) ;
                }
            }
        }
    } ;


    this.update_row = function (old_row, new_message) {
        console.log('ELog_MessageViewerByDay::update_row() not implemented for this viewer') ;
        return ;
    } ;

    this.undelete_row = function (row) {
        console.log('ELog_MessageViewerByDay::undelete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        console.log('ELog_MessageViewerByDay::delete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_message = function (message_id) {
        console.log('ELog_MessageViewerByDay::focus_at_message() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_run = function (run_num) {
        console.log('ELog_MessageViewerByDay::focus_at_run() not implemented for this viewer') ;
        return ;
    } ;

    this._day2row = function (ymd) {
        var row = {
            title: this._day2row_title(ymd) ,
            body:  new ELog_DayBody(this, this._days[ymd].messages, this.options)
        } ;
        return row ; 
    } ;
    this._day2row_title = function (ymd) {
        var messages = this._days[ymd].messages ;
        var first_run    = 0 ;
        var last_run     = 0 ;
        var num_messages = 0 ;
        for (var i in messages) {
            var m = messages[i] ;
            if (m.is_run) {
                if (!first_run) {
                    first_run = m.run_num ;
                    last_run  = m.run_num ;
                } else {
                    if (m.run_num < first_run) first_run = m.run_num ;
                    if (m.run_num > last_run)  last_run  = m.run_num ;
                }
            } else {
                num_messages++ ;
            }
        }
        var title = {
            'day'            : '<b>'+ymd+'</b>' ,
            'runs-end'       : last_run && (last_run !== first_run) ? last_run        : '&nbsp;' ,
            'runs-separator' : last_run && (last_run !== first_run) ? '&nbsp;-&nbsp;' : '&nbsp;' ,
            'runs-begin'     : first_run                            ? first_run       : '&nbsp;' ,
            'messages'       : num_messages                         ? num_messages    : '&nbsp;'
        } ;
        return title ; 
    } ;
}
Class.define_class (ELog_MessageViewerByDay, ELog_MessageViewerBase, {}, {}) ;


/**
 * Specialization for displaying a conatiner of mesasages within a table row's body
 *
 * @param {Object} parent
 * @param {Array} messages
 * @param {Object} options
 * @returns {ELog_ShiftBody}
 */
function ELog_ShiftBody (parent, messages, options) {
    ELog_MessageGroupBody.call(this, parent, messages, options, 'shift-viewer') ;
}
Class.define_class (ELog_ShiftBody, ELog_MessageGroupBody, {}, {}) ;

/**
 * Group messages by a shift they were posted.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerNoGroupping}
 */
function ELog_MessageViewerByShift (parent, options) {

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // -- parameters

    this._messages = [] ;
    this._shifts = {} ;

    var hdr = [
        {id: 'shift',          title: 'Shift', width: 160} ,
        {id: 'runs-end',       title: 'Runs',               width:  50, align: 'right'} ,
        {id: 'runs-separator', title: '&nbsp;',             width:  10, align: 'center'} ,
        {id: 'runs-begin' ,    title: '&nbsp;',             width:  60, align: 'left'} ,
        {id: 'messages',       title: 'Messages',           width:  60, align: 'right'} ,
        {id: '_',                                           width:  20} ,
        {id: 'goals' ,         title: '&nbsp;',             width: 460, align: 'left'}
    ] ;
    this._table = new StackOfRows.StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            } ,
            theme: 'stack-theme-aliceblue'
        }
    ) ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {
        this._table.display(this.container) ;
        if (this.instant_expand) this._table.expand_or_collapse(true) ;
    } ;

    this.messages = function () { return this._messages ; } ;

    this.num_rows = function () { return this._messages.length ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; } ;
    this.min_run  = function () { return this._min_run; } ;
    this.max_run  = function () { return this._max_run; } ;


    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param {Array} messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this._messages = jQuery.extend(true, [], messages) ;
        this._messages.reverse() ;

        this._shifts = {} ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        var shift_goals = {} ;  // -- harvest shift goals (if any) for each shift

        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this._messages[i] = m ;
            }
            var shift_begin_time = m.shift_begin.time ;
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            } else {
                if (_.find(m.tags, function (e) { return e.tag === 'SHIFT_GOALS' ; }))
                    shift_goals[shift_begin_time] = m.subject ;
            }
            if (!_.has(this._shifts, shift_begin_time)) {
                this._shifts[shift_begin_time] = {
                    messages: [] ,
                    goals: ''
                } ;
            }
            this._shifts[shift_begin_time].messages.push(m) ;
        }
        var shift_begin_times = _.keys(this._shifts) ;
        shift_begin_times.sort() ;
        shift_begin_times.reverse() ;

        this._table.reset() ;

        for (var i in shift_begin_times) {
            var shift_begin_time = shift_begin_times[i] ;
            this._shifts[shift_begin_time].messages.reverse() ;
            if (_.has(shift_goals, shift_begin_time)) this._shifts[shift_begin_time].goals = shift_goals[shift_begin_time] ;
            this._shifts[shift_begin_time].row_id = this._table.append(this._shift2row(shift_begin_time)) ;
        }
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            var shifts = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
                var shift_begin_time = m.shift_begin.time ;
                if (!_.has(shifts, shift_begin_time)) {
                    shifts[shift_begin_time] = {
                        messages: []
                    } ;
                }
                shifts[shift_begin_time].messages.push(m) ;
            }

            var shift_begin_times = _.keys(shifts) ;
            shift_begin_times.sort() ;
            for (var i in shift_begin_times) {

                var shift_begin_time = shift_begin_times[i] ;
                if (_.has(this._shifts, shift_begin_time)) {

                    // Extend the list of messages at the shift
                    var messages = shifts[shift_begin_time].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._shifts[shift_begin_time].messages.push(m) ;
                    }

                    // Update the title of the shift
                    var row = this._table.get_row_by_id(this._shifts[shift_begin_time].row_id) ;
                    row.update_title(this._shift2row_title(shift_begin_time), function (hdr_cont) {
                        hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
                    }) ;
                    
                    // Update the list of messages within the row's body
                    var row_body = row.get_body() ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        row_body.insert_front(m) ;
                    }
                } else {
                    
                    // Create a new row for the shift
                    this._shifts[shift_begin_time] = shifts[shift_begin_time] ;
                    this._shifts[shift_begin_time].row_id = this._table.insert_front(this._shift2row(shift_begin_time)) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages at the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            var shifts = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }            
                var shift_begin_time = m.shift_begin.time ;
                if (!_.has(shifts, shift_begin_time)) {
                    shifts[shift_begin_time] = {
                        messages: []
                    } ;
                }
                shifts[shift_begin_time].messages.push(m) ;
            }
            var shift_begin_times = _.keys(shifts) ;
            shift_begin_times.sort() ;
            shift_begin_times.reverse() ;   
            for (var i in shift_begin_times) {
                var shift_begin_time = shift_begin_times[i] ;
                if (_.has(this._shifts, shift_begin_time)) {
                    var shift = this._shifts[shift_begin_time] ;
                    var messages = shifts[shift_begin_time].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        shift.messages.push(m) ;
                    }
                    shift.row_id = this._table.update_row(shift.row_id, this._shift2row(shift_begin_time)) ;
                } else {
                    this._shifts[shift_begin_time] = shifts[shift_begin_time] ;
                    this._shifts[shift_begin_time].row_id = this._table.append(this._shift2row(shift_begin_time)) ;
                }
            }
        }
    } ;


    this.update_row = function (old_row, new_message) {
        console.log('ELog_MessageViewerByShift::update_row() not implemented for this viewer') ;
        return ;
    } ;

    this.undelete_row = function (row) {
        console.log('ELog_MessageViewerByShift::undelete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        console.log('ELog_MessageViewerByShift::delete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_message = function (message_id) {
        console.log('ELog_MessageViewerByShift::focus_at_message() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_run = function (run_num) {
        console.log('ELog_MessageViewerByShift::focus_at_run() not implemented for this viewer') ;
        return ;
    } ;

    this._shift2row = function (shift_begin_time) {
        var row = {
            title: this._shift2row_title(shift_begin_time) ,
            body:  new ELog_ShiftBody(this, this._shifts[shift_begin_time].messages, this.options)
        } ;
        return row ; 
    } ;
    this._shift2row_title = function (shift_begin_time) {
        var messages = this._shifts[shift_begin_time].messages ;
        var first_run    = 0 ;
        var last_run     = 0 ;
        var num_messages = 0 ;
        for (var i in messages) {
            var m = messages[i] ;
            if (m.is_run) {
                if (!first_run) {
                    first_run = m.run_num ;
                    last_run  = m.run_num ;
                } else {
                    if (m.run_num < first_run) first_run = m.run_num ;
                    if (m.run_num > last_run)  last_run  = m.run_num ;
                }
            } else {
                num_messages++ ;
            }
        }
        var ymd_hms = shift_begin_time.split(' ') ;
        var ymd = ymd_hms[0] ;
        var hms = ymd_hms[1] ;
        var title = {
            'shift'          : '<b>'+ymd+'</b>&nbsp;&nbsp;'+hms ,
            'runs-end'       : last_run && (last_run !== first_run) ? last_run        : '&nbsp;' ,
            'runs-separator' : last_run && (last_run !== first_run) ? '&nbsp;-&nbsp;' : '&nbsp;' ,
            'runs-begin'     : first_run                            ? first_run       : '&nbsp;' ,
            'messages'       : num_messages                         ? num_messages    : '&nbsp;' ,
            'goals'          : '<span style="color:maroon;">'+_.escape(this._shifts[shift_begin_time].goals)+'</span>'
        } ;
        return title ; 
    } ;
}
Class.define_class (ELog_MessageViewerByShift, ELog_MessageViewerBase, {}, {}) ;



/**
 * Specialization for displaying a conatiner of mesasages within a table row's body
 *
 * @param {Object} parent
 * @param {Array} messages
 * @param {Object} options
 * @returns {ELog_TagBody}
 */
function ELog_TagBody (parent, messages, options) {
    ELog_MessageGroupBody.call(this, parent, messages, options, 'tag-viewer') ;
}
Class.define_class (ELog_TagBody, ELog_MessageGroupBody, {}, {}) ;

/**
 * Group messages by a tag they were posted with.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerByAuthor}
 */
function ELog_MessageViewerByTag (parent, options) {

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // -- parameters

    this._messages = [] ;
    this._tags = {} ;

    var hdr = [
        {id: 'tag',            title: 'Tag',        width: 180} ,
        {id: 'runs-end',       title: 'Runs',       width:  50, align: 'right'} ,
        {id: 'runs-separator', title: '&nbsp;',     width:  10, align: 'center'} ,
        {id: 'runs-begin' ,    title: '&nbsp;',     width:  60, align: 'left'} ,
        {id: 'messages',       title: 'Messages',   width:  60, align: 'right'}
    ] ;
    this._table = new StackOfRows.StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            } ,
            theme: 'stack-theme-aliceblue'
        }
    ) ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {
        this._table.display(this.container) ;
        if (this.instant_expand) this._table.expand_or_collapse(true) ;
    } ;

    this.messages = function () { return this._messages ; } ;

    this.num_rows = function () { return this._messages.length ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; } ;
    this.min_run  = function () { return this._min_run; } ;
    this.max_run  = function () { return this._max_run; } ;


    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this._messages = jQuery.extend(true, [], messages) ;
        this._messages.reverse() ;

        this._tags = {} ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this._messages[i] = m ;
            }
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            }
            if (m.tags.length) {
                for (var j in m.tags) {
                    var tag_name = m.tags[j].tag ;
                    if (!_.has(this._tags, tag_name)) {
                        this._tags[tag_name] = {
                            messages: []
                        } ;
                    }
                    this._tags[tag_name].messages.push(m) ;
                }
            } else {
                var tag_name = '' ;
                if (!_.has(this._tags, tag_name)) {
                    this._tags[tag_name] = {
                        messages: []
                    } ;
                }
                this._tags[tag_name].messages.push(m) ;
            }
        }
        var tag_names = _.keys(this._tags) ;
        tag_names.sort() ;
        tag_names.reverse() ;

        this._table.reset() ;

        for (var i in tag_names) {
            var tag_name = tag_names[i] ;
            this._tags[tag_name].messages.reverse() ;
            this._tags[tag_name].row_id = this._table.append(this._tag2row(tag_name)) ;
        }
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            var tags = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
                if (m.tags.length) {
                    for (var j in m.tags) {
                        var tag_name = m.tags[j].tag ;
                        if (!_.has(tags, tag_name)) {
                            tags[tag_name] = {
                                messages: []
                            } ;
                        }
                        tags[tag_name].messages.push(m) ;
                    }
                } else {
                    var tag_name = '' ;
                    if (!_.has(tags, tag_name)) {
                        tags[tag_name] = {
                            messages: []
                        } ;
                    }
                    tags[tag_name].messages.push(m) ;
                }
            }

            var tag_names = _.keys(tags) ;
            tag_names.sort() ;
            for (var i in tag_names) {

                var tag_name = tag_names[i] ;
                if (_.has(this._tags, tag_name)) {

                    // Extend the list of messages with the tag
                    var messages = tags[tag_name].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._tags[tag_name].messages.push(m) ;
                    }

                    // Update the title of the tag bar
                    var row = this._table.get_row_by_id(this._tags[tag_name].row_id) ;
                    row.update_title(this._tag2row_title(tag_name), function (hdr_cont) {
                        hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
                    }) ;
                    
                    // Update the list of messages within the row's body
                    var row_body = row.get_body() ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        row_body.insert_front(m) ;
                    }
                } else {
                    
                    // Create a new row for the tag
                    this._tags[tag_name] = tags[tag_name] ;
                    this._tags[tag_name].row_id = this._table.insert_front(this._tag2row(tag_name)) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages at the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            var tags = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
                if (m.tags.length) {
                    for (var j in m.tags) {
                        var tag_name = m.tags[j].tag ;
                        if (!_.has(tags, tag_name)) {
                            tags[tag_name] = {
                                messages: []
                            } ;
                        }
                        tags[tag_name].messages.push(m) ;
                    }
                } else {
                    var tag_name = '' ;
                    if (!_.has(tags, tag_name)) {
                        tags[tag_name] = {
                            messages: []
                        } ;
                    }
                    tags[tag_name].messages.push(m) ;
                }
            }
            var tag_names = _.keys(tags) ;
            tag_names.sort() ;
            tag_names.reverse() ;   
            for (var i in tag_names) {
                var tag_name = tag_names[i] ;
                if (_.has(this._tags, tag_name)) {
                    var messages = tags[tag_name].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._tags[tag_name].messages.push(m) ;
                    }
                    this._tags[tag_name].row_id = this._table.update_row(this._tags[tag_name].row_id, this._tag2row(tag_name)) ;
                } else {
                    this._tags[tag_name] = tags[tag_name] ;
                    this._tags[tag_name].row_id = this._table.append(this._tag2row(tag_name)) ;
                }
            }
        }
    } ;


    this.update_row = function (old_row, new_message) {
        console.log('ELog_MessageViewerByTag::update_row() not implemented for this viewer') ;
        return ;
    } ;

    this.undelete_row = function (row) {
        console.log('ELog_MessageViewerByTag::undelete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        console.log('ELog_MessageViewerByTag::delete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_message = function (message_id) {
        console.log('ELog_MessageViewerByTag::focus_at_message() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_run = function (run_num) {
        console.log('ELog_MessageViewerByTag::focus_at_run() not implemented for this viewer') ;
        return ;
    } ;

    this._tag2row = function (tag_name) {
        var row = {
            title: this._tag2row_title(tag_name) ,
            body:  new ELog_TagBody(this, this._tags[tag_name].messages, this.options)
        } ;
        return row ; 
    } ;
    this._tag2row_title = function (tag_name) {
        var messages = this._tags[tag_name].messages ;
        var first_run    = 0 ;
        var last_run     = 0 ;
        var num_messages = 0 ;
        for (var i in messages) {
            var m = messages[i] ;
            if (m.is_run) {
                if (!first_run) {
                    first_run = m.run_num ;
                    last_run  = m.run_num ;
                } else {
                    if (m.run_num < first_run) first_run = m.run_num ;
                    if (m.run_num > last_run)  last_run  = m.run_num ;
                }
            } else {
                num_messages++ ;
            }
        }
        var title = {
            'tag'            : tag_name                             ? tag_name        : '&nbsp;' ,
            'runs-end'       : last_run && (last_run !== first_run) ? last_run        : '&nbsp;' ,
            'runs-separator' : last_run && (last_run !== first_run) ? '&nbsp;-&nbsp;' : '&nbsp;' ,
            'runs-begin'     : first_run                            ? first_run       : '&nbsp;' ,
            'messages'       : num_messages                         ? num_messages    : '&nbsp;'
        } ;
        return title ; 
    } ;
}
Class.define_class (ELog_MessageViewerByTag, ELog_MessageViewerBase, {}, {}) ;



/**
 * Specialization for displaying a conatiner of mesasages within a table row's body
 *
 * @param {Object} parent
 * @param {Array} messages
 * @param {Object} options
 * @returns {ELog_AuthorBody}
 */
function ELog_AuthorBody (parent, messages, options) {
    ELog_MessageGroupBody.call(this, parent, messages, options, 'author-viewer') ;
}
Class.define_class (ELog_AuthorBody, ELog_MessageGroupBody, {}, {}) ;


/**
 * Group messages by a shift they were posted.
 *
 * @param {Object} parent
 * @param {Object} options
 * @returns {ELog_MessageViewerNoGroupping}
 */
function ELog_MessageViewerByAuthor (parent, options) {

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // -- parameters

    this._messages = [] ;
    this._authors = {} ;

    var hdr = [
        {id: 'author',         title: 'Author',     width: 100} ,
        {id: 'runs-end',       title: 'Runs',       width:  50, align: 'right'} ,
        {id: 'runs-separator', title: '&nbsp;',     width:  10, align: 'center'} ,
        {id: 'runs-begin' ,    title: '&nbsp;',     width:  60, align: 'left'} ,
        {id: 'messages',       title: 'Messages',   width:  60, align: 'right'}
    ] ;
    this._table = new StackOfRows.StackOfRows (
        hdr ,
        [] ,
        {   hidden_header: this.hidden_header ,
            effect_on_insert: function (hdr_cont) {
               hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
            } ,
            theme: 'stack-theme-aliceblue'
        }
    ) ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {
        this._table.display(this.container) ;
        if (this.instant_expand) this._table.expand_or_collapse(true) ;
    } ;

    this.messages = function () { return this._messages ; } ;

    this.num_rows = function () { return this._messages.length ; } ;

    this._num_runs = 0 ;
    this._min_run = 0 ;
    this._max_run = 0 ;

    this.num_runs = function () { return this._num_runs; } ;
    this.min_run  = function () { return this._min_run; } ;
    this.max_run  = function () { return this._max_run; } ;


    /**
     * Reload internal message store and redisplay the list of messages.
     * 
     * NOTE: The method will make a local deep copy of the input messages.
     * 
     * @param Array messages
     * @returns {undefined}
     */
    this.load = function (messages) {

        this._messages = jQuery.extend(true, [], messages) ;
        this._messages.reverse() ;

        this._authors = {} ;

        this._num_runs = 0 ;
        this._min_run = 0 ;
        this._max_run = 0 ;

        for (var i in this._messages) {
            var m = this._messages[i] ;
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
                this._messages[i] = m ;
            }
            if (m.is_run) {
                this._num_runs++ ;
                this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
            }
            var author_name = m.author ;
            if (!_.has(this._authors, author_name)) {
                this._authors[author_name] = {
                    messages: []
                } ;
            }
            this._authors[author_name].messages.push(m) ;
        }
        var author_names = _.keys(this._authors) ;
        author_names.sort() ;
        author_names.reverse() ;

        this._table.reset() ;

        for (var i in author_names) {
            var author_name = author_names[i] ;
            this._authors[author_name].messages.reverse() ;
            this._authors[author_name].row_id = this._table.append(this._author2row(author_name)) ;
        }
    } ;

    /**
     * Insert new messages in front of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.update = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages in front of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will always get to the front.

            var authors = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }
                var author_name = m.author ;
                if (!_.has(authors, author_name)) {
                    authors[author_name] = {
                        messages: []
                    } ;
                }
                authors[author_name].messages.push(m) ;
            }

            var author_names = _.keys(authors) ;
            author_names.sort() ;
            for (var i in author_names) {

                var author_name = author_names[i] ;
                if (_.has(this._authors, author_name)) {

                    // Extend the list of messages of the author
                    var messages = authors[author_name].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._authors[author_name].messages.push(m) ;
                    }

                    // Update the title of the author bar
                    var row = this._table.get_row_by_id(this._authors[author_name].row_id) ;
                    row.update_title(this._author2row_title(author_name), function (hdr_cont) {
                        hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666 !important'}, 30000) ;
                    }) ;
                    
                    // Update the list of messages within the row's body
                    var row_body = row.get_body() ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        row_body.insert_front(m) ;
                    }
                } else {
                    
                    // Create a new row for the author
                    this._authors[author_name] = authors[author_name] ;
                    this._authors[author_name].row_id = this._table.insert_front(this._author2row(author_name)) ;
                }
            }
        }
    } ;

    /**
     * Append messages at the bottom of the table
     * 
     * @param Array new_messages
     * @returns {undefined}
     */
    this.append = function (new_messages) {

        var length = new_messages ? new_messages.length : 0;
        if (length) {

            // Put deep copies of the new messages at the very end of the of the local list.
            // Note that this will also reverse the order in which we got the new
            // messages so that the newest ones will alwats get to the front.

            new_messages.reverse() ;    // need this to append newst message first

            var authors = {} ;

            for (var i in new_messages) {
                var m = new_messages[i] ;
                if (typeof m === 'string') m = eval('('+m+')') ;
                m = jQuery.extend(true, {}, m) ;
                if (this._messages.length) this._messages.splice(0, 0, m) ;
                else                       this._messages.push(m) ;
                if (m.is_run) {
                    this._num_runs++ ;
                    this._min_run = Math.min(this._min_run ? this._min_run : m.run_num, m.run_num) ;
                    this._max_run = Math.max(this._max_run ? this._max_run : m.run_num, m.run_num) ;
                }            
                var author_name = m.author ;
                if (!_.has(authors, author_name)) {
                    authors[author_name] = {
                        messages: []
                    } ;
                }
                authors[author_name].messages.push(m) ;
            }
            var author_names = _.keys(authors) ;
            author_names.sort() ;
            author_names.reverse() ;   
            for (var i in author_names) {
                var author_name = author_names[i] ;
                if (_.has(this._authors, author_name)) {
                    var messages = authors[author_name].messages ;
                    for (var j in messages) {
                        var m = messages[j] ;
                        this._authors[author_name].messages.push(m) ;
                    }
                    this._authors[author_name].row_id = this._table.update_row(this._authors[author_name].row_id, this._author2row(author_name)) ;
                } else {
                    this._authors[author_name] = authors[author_name] ;
                    this._authors[author_name].row_id = this._table.append(this._author2row(author_name)) ;
                }
            }
        }
    } ;


    this.update_row = function (old_row, new_message) {
        console.log('ELog_MessageViewerByAuthor::update_row() not implemented for this viewer') ;
        return ;
    } ;

    this.undelete_row = function (row) {
        console.log('ELog_MessageViewerByAuthor::undelete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.delete_row = function (row, deleted_time, deleted_by) {
        console.log('ELog_MessageViewerByAuthor::delete_row() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_message = function (message_id) {
        console.log('ELog_MessageViewerByAuthor::focus_at_message() not implemented for this viewer') ;
        return ;
    } ;

    this.focus_at_run = function (run_num) {
        console.log('ELog_MessageViewerByAuthor::focus_at_run() not implemented for this viewer') ;
        return ;
    } ;

    this._author2row = function(author_name) {
        var row = {
            title: this._author2row_title(author_name) ,
            body:  new ELog_AuthorBody(this, this._authors[author_name].messages, this.options)
        } ;
        return row ; 
    } ;
    this._author2row_title = function (author_name) {
        var messages = this._authors[author_name].messages ;
        var first_run    = 0 ;
        var last_run     = 0 ;
        var num_messages = 0 ;
        for (var i in messages) {
            var m = messages[i] ;
            if (m.is_run) {
                if (!first_run) {
                    first_run = m.run_num ;
                    last_run  = m.run_num ;
                } else {
                    if (m.run_num < first_run) first_run = m.run_num ;
                    if (m.run_num > last_run)  last_run  = m.run_num ;
                }
            } else {
                num_messages++ ;
            }
        }
        var title = {
            'author'         : author_name                          ? author_name     : '&nbsp;'  ,
            'runs-end'       : last_run && (last_run !== first_run) ? last_run        : '&nbsp;' ,
            'runs-separator' : last_run && (last_run !== first_run) ? '&nbsp;-&nbsp;' : '&nbsp;' ,
            'runs-begin'     : first_run                            ? first_run       : '&nbsp;' ,
            'messages'       : num_messages                         ? num_messages    : '&nbsp;'
        } ;
        return title ; 
    } ;
}
Class.define_class (ELog_MessageViewerByAuthor, ELog_MessageViewerBase, {}, {}) ;


/**
 * The front-end dispatcher managing and coordinating specific message viewers
 *
 * @param {Object} parent
 * @param {JQuery} cont
 * @param {Object} options
 * @returns {ELog_MessageViewer}
 */
function ELog_MessageViewer (parent, cont, options) {

    var _that = this ;

    // Always call the c-tor of the base class

    ELog_MessageViewerBase.call(this, parent, options) ;

    // Parameters of the object

    this.cont = cont ;

    // Construct the default viewer instance
 
    this._body = null ;
    this._viewer = null ;
    this._rb = null ;

    /**
     * @function - overloading the function of the base class Widget
     */
    this.render = function () {

        if (this.allow_groups) {
            var html =
'<div id="ctrl"></div>' +
'<div id="body">' +
'  <div id="stream"></div>' +
'  <div id="day"></div>' ;
            if (this.allow_shifts) html +=
'  <div id="shift"></div>' ;
            html +=
'  <div id="tag"></div>' +
'  <div id="author"></div>' +
'</div>' ;
            this.cont.html(html) ;

            var cfg = [
                {  name:  "stream" ,
                    text:  "SIMPLE STREAM" ,
                    class: "control-button" ,
                    title: "The groupping is off. Messages are show in the same order they were posted."
                } , {
                    name:  "day" ,
                    text:  "GROUP BY DAY" ,
                    class: "control-button" ,
                    title: "Group messages by a day they were posted."
                }
            ] ;
            if (this.allow_shifts) cfg.push (
                {
                    name:  "shift" ,
                    text:  "GROUP BY SHIFT" ,
                    class: "control-button" ,
                    title: "Group messages by a shift in which they were posted."
                }
            ) ;
            cfg.push (
                {
                    name:  "tag" ,
                    text:  "GROUP BY TAG" ,
                    class: "control-button" ,
                    title: "Group messages by message tags."
                } , {
                    name:  "author" ,
                    text:  "GROUP BY AUTHOR" ,
                    class: "control-button" ,
                    title: "Group messages by message authors."
                }
            ) ;
            this._rb = new RadioBox (
                cfg ,
                function (name) {

                    // Borrow messages from the old viewer. Make sure they're sorted
                    // correctly. Otherwise the viewer will show them in the reverse order.

                    var messages = _that._viewer.messages() ;
                    messages.reverse() ;

                    switch (name) {
                        case 'stream' :
                            _that._viewer = new ELog_MessageViewerNoGroupping(_that.parent, _that.options) ;
                            _that._viewer.display(_that._body.children('div#stream')) ;
                            _that._body.children('div#day').html('') ;
                            _that._body.children('div#shift').html('') ;
                            _that._body.children('div#tag').html('') ;
                            _that._body.children('div#author').html('') ;
                            break ;
                        case 'day' :
                            _that._viewer = new ELog_MessageViewerByDay(_that.parent, _that.options) ;
                            _that._viewer.display(_that._body.children('div#day') ) ;
                            _that._body.children('div#stream').html('') ;
                            _that._body.children('div#shift').html('') ;
                            _that._body.children('div#tag').html('') ;
                            _that._body.children('div#author').html('') ;
                            break ;
                        case 'shift' :
                            _that._viewer = new ELog_MessageViewerByShift(_that.parent, _that.options) ;
                            _that._viewer.display(_that._body.children('div#shift') ) ;
                            _that._body.children('div#stream').html('') ;
                            _that._body.children('div#day').html('') ;
                            _that._body.children('div#tag').html('') ;
                            _that._body.children('div#author').html('') ;
                            break ;
                        case 'tag' :
                            _that._viewer = new ELog_MessageViewerByTag(_that.parent, _that.options) ;
                            _that._viewer.display(_that._body.children('div#tag') ) ;
                            _that._body.children('div#stream').html('') ;
                            _that._body.children('div#day').html('') ;
                            _that._body.children('div#shift').html('') ;
                            _that._body.children('div#author').html('') ;
                            break ;
                        case 'author' :
                            _that._viewer = new ELog_MessageViewerByAuthor(_that.parent, _that.options) ;
                            _that._viewer.display(_that._body.children('div#author') ) ;
                            _that._body.children('div#stream').html('') ;
                            _that._body.children('div#day').html('') ;
                            _that._body.children('div#shift').html('') ;
                            _that._body.children('div#tag').html('') ;
                            break ;
                    }
                    _that._viewer.load(messages) ;
                } ,

                {
                    activate: "stream"
                }
            ) ;
            this._rb.display(this.cont.children('div#ctrl')) ;

        } else {
            var html =
'<div id="body">' +
'  <div id="stream"></div>' +
'</div>' ;
            this.cont.html(html) ;
        }

        // Install the initial viewer

        this._body = this.cont.children('div#body') ;
        this._viewer = new ELog_MessageViewerNoGroupping(this.parent, this.options) ;
        this._viewer.display(this._body.children('div#stream') ) ;
    } ;

    // Forward base class's methods to the viewer's instance

    this.num_rows = function () { return this._viewer.num_rows() ; } ;
    this.num_runs = function () { return this._viewer.num_runs() ; } ;
    this.min_run  = function () { return this._viewer.min_run () ; } ;
    this.max_run  = function () { return this._viewer.max_run () ; } ;

    this.load     = function (messages)     { this._viewer.load  (messages) ; } ;
    this.update   = function (new_messages) { this._viewer.update(new_messages) ; } ;
    this.append   = function (new_messages) { this._viewer.append(new_messages) ; } ;

    this.update_row   = function (old_row, new_message)          { this._viewer.update_row  (old_row, new_message) ; } ;    
    this.undelete_row = function (row)                           { this._viewer.undelete_row(row) ; } ;
    this.delete_row   = function (row, deleted_time, deleted_by) { this._viewer.delete_row  (row, deleted_time, deleted_by) ; } ;

    this.focus_at_message = function (message_id) { this._viewer.focus_at_message(message_id) ; } ;
    this.focus_at_run     = function (run_num)    { this._viewer.focus_at_run    (run_num) ; } ;

    // Must be the last call. Otherwise the widget won't be able to see
    // functon 'render()' defined above in this code.

    this.display(this.cont) ;
}
Class.define_class (ELog_MessageViewer, ELog_MessageViewerBase, {}, {}) ;

    return ELog_MessageViewer ;
}) ;