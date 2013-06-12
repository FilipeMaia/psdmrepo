function p_appl_equipment () {

    var that = this ;

    this.when_done           = null ;
    this.create_form_changed = false ;

    /* -------------------------------------------------------------------------
     * Data structures and methods to be used/called by external users:
     *
     *   select(context, when_done)
     *      select a specific context
     *
     *   select_default()
     *      select default context as implemented in the object
     *
     *   if_ready2giveup(handler2call)
     *      check if the object's state allows to be released, and if so then
     *      call the specified function. Otherwise just ignore it. Normally
     *      this operation is used as a safeguard preventing releasing
     *      an interface focus if there is on-going unfinished editing
     *      within one of the interfaces associated with the object.
     */
    this.name      = 'equipment' ;
    this.full_name = 'Equipment' ;
    this.context   = '' ;
    this.default_context = 'inventory' ;

    this.select = function (context,when_done) {
        that.context   = context ;
        this.when_done = when_done ;
        this.init() ;
        switch (this.context) {
            case 'add':       this.init_add() ; return ;
            case 'inventory': this.init_inventory() ; return ;
        }
    } ;
    this.select_default = function () {
        if (this.context == '') this.context = this.default_context ;
        this.init() ;
        switch (this.context) {
            case 'add':       this.init_add() ; return ;
            case 'inventory': this.init_inventory() ; return ;
        }
    } ;
    this.if_ready2giveup = function (handler2call) {
        if ((this.context == 'add') && this.create_form_changed) {
            ask_yes_no (
                'Unsaved Data Warning',
                'You are about to leave the page while there are unsaved data in the form. Are you sure?',
                handler2call,
                null) ;
            return ;
        }
        handler2call() ;
    } ;
    this.search_equipment_by = function (id) {
        this.equipment_search_impl({equipment_id: id}) ;
    } ;
    this.search_equipment_by_slacid_range = function (id) {
        this.equipment_search_impl({slacid_range_id: id}) ;
    } ;
    this.search_equipment_by_manufacturer = function (id) {
        this.equipment_search_impl({manufacturer_id: id}) ;
    } ;
    this.search_equipment_by_slacid_range = function (id) {
        this.equipment_search_impl({slacid_range_id: id}) ;
    } ;
    this.search_equipment_by_status = function (id) {
        this.equipment_search_impl({status_id: id}) ;
    }  ;
    this.search_equipment_by_status2 = function (id) {
        this.equipment_search_impl({status2_id: id}) ;
    }  ;
    this.search_equipment_by_location = function (id) {
        this.equipment_search_impl({location_id: id}) ;
    } ;
    this.search_equipment_by_room = function (id) {
        this.equipment_search_impl({room_id: id}) ;
    } ;
    this.search_equipment_by_model = function (id) {
        this.equipment_search_impl({model_id: id}) ;
    } ;

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */

    // Deep copies of equipment objects to allow editing and history operations
    // while the search results change.
    
    this.equipment_editing = {} ;
    this.equipment_history = {} ;

    this.tabs = null ;
    this.equipment_view_mode = null ;

    this.initialized = false ;
    this.init = function () {
        if (this.initialized) return ;
        this.initialized = true ;

        var inventory_controls = $('#equipment-inventory-controls') ;
        inventory_controls.find('button[name="search"]').
            button().
            click(function () {
                that.equipment_search() ; }) ;

        inventory_controls.find('button[name="reset"]')
            .button().
            click(function () {
                that.equipment_search_reset() ; }) ;

        $('#equipment-add-save').
            button().
            button('disable').
            click(function () {
                that.equipment_save() ; }) ;

        $('#equipment-add-reset').
            button().
            click(function () {
                that.init_add() ;
                $('#equipment-add-save').button('disable') ; }) ;

        $('.equipment-add-form-element').change(function () {
            that.add_form_validate() ;
        }) ;
        
        this.tabs = $('#equipment-inventory').find('#tabs').tabs() ;
        $('#equipment-inventory').find('#tabs span.ui-icon-close.edit').live('click', function () {
            var panelId = $(this).closest('li').attr('aria-controls') ;
            that.equipment_edit_tab_close(panelId) ;
        });
        $('#equipment-inventory').find('#tabs span.ui-icon-close.history').live('click', function () {
            var panelId = $(this).closest('li').attr('aria-controls') ;
            that.equipment_history_tab_close(panelId) ;
        });

        var view = $('#equipment-inventory').find('div#view') ;
        view.buttonset() ;

        var view_table = view.find('#view_table') ;
        view_table.click(function () {
            that.equipment_view_mode = 'table';
            that.equipment_display() ;
        }) ;
        var view_grid = view.find('#view_grid') ;
        view_grid.click(function () {
            that.equipment_view_mode = 'grid';
            that.equipment_display() ;
        }) ;
        this.equipment_view_mode = view_table.attr('checked') ? 'table' : 'grid' ;
        
        $('#equipment-inventory').find('#option_model_image').change(function () {
            var checked = $(this).attr('checked') ? true : false ;
            var elements = $('#equipment-inventory-table').find('div.equipment-model-image') ;
            if (checked) {
                elements.removeClass('hidden').addClass('visible') ;
                elements.each(function () {
                    if (!$(this).html()) {
                        var equipment_id = $(this).attr('name') ;
                        $(this).html('<a class="link" href="../irep/equipment_model_attachments/'+equipment_id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/equipment_model_attachments/preview/'+equipment_id+'" width="102" height="72" /></a>') ;
                    }
                }) ;
            } else {
                elements.removeClass('visible').addClass('hidden') ;
            }
        }) ;
        $('#equipment-inventory').find('#option_attachment_preview').change(function () {
            var checked = $(this).attr('checked') ? true : false ;
            var tgl = $('#equipment-inventory-table').find('span.equipment-attachment-tgl') ;
            var con = $('#equipment-inventory-table').find('div.equipment-attachment-preview') ;
            if (checked) {
                tgl.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
                con.removeClass('hidden').addClass('visible') ;
                con.each(function () {
                    if (!$(this).html()) {
                        var attachment_id = $(this).attr('name') ;
                        $(this).html('<a class="link" href="../irep/equipment_attachments/'+attachment_id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab"><img src="../irep/equipment_attachments/preview/'+attachment_id+'" /></a>') ;
                    }
                }) ;
            } else {
                tgl.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
                con.removeClass('visible').addClass('hidden') ;
            }
        }) ;
    } ;
    this.can_edit_inventory = function () {
        return global_current_user.can_edit_inventory ;
    } ;

    /* -------------
     *   Inventory
     * -------------
     */
    this.equipment = [] ;
    this.equipment_by_id = [] ;

    this.init_inventory = function () {

        // Get the most recent selectons from the database and update
        // the from while preserving previously made user choices (if any).
        //
        var form_elem = $('#equipment-inventory-form') ;

        var status_elem = form_elem.find('select[name="status"]') ;
        var selected_status = status_elem.val() ;

        var status2_elem = form_elem.find('select[name="status2"]') ;
        var selected_status2 = status2_elem.val() ;

        var manufacturer_elem = form_elem.find('select[name="manufacturer"]') ;
        var selected_manufacturer_id = manufacturer_elem.val() ;

        var model_elem = form_elem.find('select[name="model"]') ;
        var selected_model_id = model_elem.val() ;

        var location_elem = form_elem.find('select[name="location"]') ;
        var selected_location_id = location_elem.val() ;

        var custodian_elem = form_elem.find('select[name="custodian"]') ;
        var selected_custodian = custodian_elem.val() ;

        var tag_elem = form_elem.find('select[name="tag"]') ;
        var selected_tag = tag_elem.val() ;

        web_service_GET ('../irep/ws/equipment_search_options.php', {}, function (data) {

            var html = '<option value=""></option>' ;
            var html2 = '<option value=""></option>' ;
            for (var i in data.option.status) {
                var status = data.option.status[i] ;
                html += '<option value="'+status.name+'">'+status.name+'</option>' ;
                if (selected_status == status.name) {
                    for (var j in status.status2) {
                        var status2 = status.status2[j] ;
                        html2 += '<option value="'+status2.name+'">'+status2.name+'</option>' ;
                    }
                }
            }
            status_elem.html(html) ;
            status_elem.val(selected_status) ;

            status2_elem.html(html2) ;
            status2_elem.val(selected_status2) ;

            status_elem.change (function () {
                var selected_status = $(this).val() ;
                var html2 = '<option value=""></option>' ;
                for (var i in data.option.status) {
                    var status = data.option.status[i] ;
                    if (selected_status == status.name) {
                        for (var j in status.status2) {
                            var status2 = status.status2[j] ;
                            html2 += '<option value="'+status2.name+'">'+status2.name+'</option>' ;
                        }
                    }
                }
                status2_elem.html(html2) ;
            }) ;

            html = '<option value="0"></option>' ;
            for (var i in data.option.manufacturer) {
                var manufacturer = data.option.manufacturer[i] ;
                html += '<option value="'+manufacturer.id+'">'+manufacturer.name+'</option>' ;
            }
            manufacturer_elem.html(html) ;
            manufacturer_elem.val(selected_manufacturer_id) ;

            html = '<option value="0"></option>' ;
            for (var i in data.option.model) {
                var model = data.option.model[i] ;
                html += '<option value="'+model.id+'">'+model.name+'</option>' ;
            }
            model_elem.html(html) ;
            model_elem.val(selected_model_id) ;

            html = '<option value="0"></option>' ;
            for (var i in data.option.location) {
                var location = data.option.location[i] ;
                html += '<option value="'+location.id+'">'+location.name+'</option>' ;
            }
            location_elem.html(html) ;
            location_elem.val(selected_location_id) ;

            html = '<option value=""></option>' ;
            for (var i in data.option.custodian) {
                var custodian = data.option.custodian[i] ;
                html += '<option value="'+custodian+'">'+custodian+'</option>' ;
            }
            custodian_elem.html(html) ;
            custodian_elem.val(selected_custodian) ;

            html = '<option value=""></option>' ;
            for (var i in data.option.tag) {
                var tag = data.option.tag[i] ;
                html += '<option value="'+tag+'">'+tag+'</option>' ;
            }
            tag_elem.html(html) ;
            tag_elem.val(selected_tag) ;
        }) ;
        this.equipment_display() ;
    } ;
    this.equipment_select_tab = function (idx) {
        if (idx != this.tabs.tabs('option','active')) this.tabs.tabs('option', 'active', idx) ;
    };
    this.equipment_inventory_table = null ;

    this.equipment_display = function () {
        switch (this.equipment_view_mode) {
            case 'table': this.equipment_display_table() ; break;
            case 'grid' : this.equipment_display_grid () ; break;
            default:
                report_error('implementation error, unsupported view mode: '+this.equipment_view_mode) ;
                break ;
        }
    } ;
    this.equipment_display_table = function () {
        var option_model_image = $('#equipment-inventory').find('#option_model_image').attr('checked') ? true : false ;
        var option_attachment_preview = $('#equipment-inventory').find('#option_attachment_preview').attr('checked') ? true : false ;
        var elem = $('#equipment-inventory-table') ;
        var hdr = [
            {   name: '<center>STATUS</center>' ,
                coldef: [
                    {   name: 'status' , hideable: true } ,
                    {   name: 'sub-status' , hideable: true }
                ]
            } ,
            {   name: 'OPERATIONS', hideable: true, sorted: false ,
                type: {
                    after_sort: function () {
                        if (that.can_edit_inventory()) {
                            elem.find('.equipment-inventory-delete').
                                button().
                                click(function () {
                                    var id = this.name ;
                                    that.equipment_delete(id, this) ;
                                }) ;
                            elem.find('.equipment-inventory-edit').
                                button().
                                click(function () {
                                    var id = this.name ;
                                    that.equipment_edit(id) ;
                                }) ;
                        }
                        elem.find('.equipment-inventory-history').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.equipment_history(id) ;
                            }) ;
                        elem.find('.equipment-inventory-print').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.equipment_print(id) ;
                            }) ;
                        elem.find('.equipment-inventory-link').
                            button().
                            click(function () {
                                var id = this.name ;
                                that.equipment_url(id) ;
                            }) ;
                    }
                }
            } ,
            {   name: 'manufacturer', hideable: true } ,
            {   name: 'model', hideable: true } ,
            {   name: 'serial #', hideable: true } ,
            {   name: 'SLAC ID', hideable: true } ,
            {   name: 'PC #', hideable: true } ,
            {   name: 'location', hideable: true } ,
            {   name: 'custodian', hideable: true } ,
            {   name: 'tags', hideable: true, sorted: false } ,
            {   name: 'attachments', hideable: true, sorted: false } ,
            {   name: 'modified', hideable: true, style: ' white-space: nowrap;' } ,
            {   name: 'by user', hideable: true }
        ] ;

        var rows = [] ;
        for (var i in this.equipment) {

            var equipment = this.equipment[i] ;

            var tags_html = '' ;
            for (var j in equipment.tag) {
                var t = equipment.tag[j] ;
                tags_html +=
'      <div class="equipment-tag">' +
'        <span>'+t.name+'</span>' +
'      </div>' ;
            }

            var attachments_html = '' ;
            for (var j in equipment.attachment) {
                var a = equipment.attachment[j] ;
                attachments_html +=
'      <div style="padding:5px;">'+
'        <div style="float:left;">' +
'          <span class="toggler equipment-attachment-tgl ui-icon el-l-a-tgl ' + (option_attachment_preview ? 'ui-icon-triangle-1-s' : 'ui-icon-triangle-1-e') + '" id="equipment-attachment-tgl-'+a.id+'" onclick="equipment.toggle_attachment('+a.id+')" ></span>' +
'        </div>' +
'        <div style="float:left; margin-left:5px;">' +
'          <a class="link" href="../irep/equipment_attachments/'+a.id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab">'+a.name+'</a>' +
'          ( <b>type :</b> '+a.document_type+'  <b>size :</b> '+a.document_size_bytes+' )' +
'        </div>' +
'        <div style="clear:both;"></div>' + (option_attachment_preview ?
'        <div id="equipment-attachment-con-'+a.id+'" class="visible equipment-attachment-preview" name="'+a.id+'" style="margin-left:20px; padding:5px;" ><a class="link" href="../irep/equipment_attachments/'+a.id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab"><img src="../irep/equipment_attachments/preview/'+a.id+'" /></a></div>' :
'        <div id="equipment-attachment-con-'+a.id+'" class="hidden equipment-attachment-preview" name="'+a.id+'" style="margin-left:20px; padding:5px;" ></div>') +
'      </div>' ;
            }
            rows.push ([
                equipment.status ,
                equipment.status2 ,
                (this.can_edit_inventory() ?
                    Button_HTML('D', {
                        name:    equipment.id,
                        classes: 'equipment-inventory-delete',
                        title:   'delete this equipment from the database' }) +
                    Button_HTML('E', {
                        name:    equipment.id,
                        classes: 'equipment-inventory-edit',
                        title:   'edit this equipment or change its status' }) :
                    ''
                ) +
                Button_HTML('H', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-history',
                    title:   'show a history of this equipment' }) +
                Button_HTML('P', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-print',
                    title:   'print a summary page on this equipment' }) +
                Button_HTML('url', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-link',
                    title:   'persistent URL for this equipment' }) ,

                equipment.manufacturer ,

                '<div>'+equipment.model+'</div>'+(option_model_image ?
                '<div class="visible equipment-model-image" name="'+equipment.id+'" style="float:left; padding:5px;"><a class="link" href="../irep/equipment_model_attachments/'+equipment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/equipment_model_attachments/preview/'+equipment.id+'" width="102" height="72" /></a></div>' :
                '<div class="hidden equipment-model-image" name="'+equipment.id+'" style="float:left; padding:5px;"></div>') ,
                equipment.serial ,
                equipment.slacid ,
                equipment.pc ,
                equipment.location ,
                equipment.custodian ,
                tags_html ,
                attachments_html ,
                equipment.modified_time ,
                equipment.modified_uid
            ]) ;
        }
        this.equipment_inventory_table = new Table (
            'equipment-inventory-table' ,
            hdr, rows ,
            {} ,
            config.handler('inventory', 'equipment_inventory_table')
        ) ;
        this.equipment_inventory_table.display() ;
        this.equipment_select_tab(0) ;
    } ;
    this.equipment_display_grid = function () {
        var elem = $('#equipment-inventory-table') ;
        var option_model_image = $('#equipment-inventory').find('#option_model_image').attr('checked') ? true : false ;
        var option_attachment_preview = $('#equipment-inventory').find('#option_attachment_preview').attr('checked') ? true : false ;
        var cell_left  = 'class="table_cell table_cell_left  " style="border:0; padding-right:0px;" align="right"' ;
        var cell_right = 'class="table_cell table_cell_right " style="border:0; padding-right:10px;"' ;
        var html = '' ;
        for (var i in this.equipment) {
            var equipment = this.equipment[i] ;

            var tags_html = '' ;
            var num_tags = 0 ;
            for (var j in equipment.tag) {
                var t = equipment.tag[j] ;
                tags_html +=
'      <div style="float:left;" class="equipment-tag">' +
'        <span>'+t.name+'</span>' +
'      </div>' ;
                num_tags++ ;
            }
            var attachments_html = '' ;
            var num_attachments = 0 ;
            for (var j in equipment.attachment) {
                var a = equipment.attachment[j] ;
                attachments_html +=
'      <div style="padding:5px;">'+
'        <div style="float:left;">' +
'          <span class="toggler equipment-attachment-tgl ui-icon el-l-a-tgl ' + (option_attachment_preview ? 'ui-icon-triangle-1-s' : 'ui-icon-triangle-1-e') + '" id="equipment-attachment-tgl-'+a.id+'" onclick="equipment.toggle_attachment('+a.id+')" ></span>' +
'        </div>' +
'        <div style="float:left; margin-left:5px;">' +
'          <a class="link" href="../irep/equipment_attachments/'+a.id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab">'+a.name+'</a>' +
'          ( <b>type :</b> '+a.document_type+'  <b>size :</b> '+a.document_size_bytes+' )' +
'        </div>' +
'        <div style="clear:both;"></div>' + (option_attachment_preview ?
'        <div id="equipment-attachment-con-'+a.id+'" class="visible equipment-attachment-preview" name="'+a.id+'" style="margin-left:20px; padding:5px;" ><a class="link" href="../irep/equipment_attachments/'+a.id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab"><img src="../irep/equipment_attachments/preview/'+a.id+'" /></a></div>' :
'        <div id="equipment-attachment-con-'+a.id+'" class="hidden equipment-attachment-preview" name="'+a.id+'" style="margin-left:20px; padding:5px;" ></div>') +
'      </div>' ;
                num_attachments++ ;
            }
            html +=
'<div class="equipment-grid-cell" style="float:left;">' +
'  <div class="header">' +
'    <div style="float:left;">' +
                (this.can_edit_inventory() ?
                    Button_HTML('D', {
                        name:    equipment.id,
                        classes: 'equipment-inventory-delete',
                        title:   'delete this equipment from the database' }) +
                    Button_HTML('E', {
                        name:    equipment.id,
                        classes: 'equipment-inventory-edit',
                        title:   'edit this equipment or change its status' }) :
                    ''
                ) +
                Button_HTML('H', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-history',
                    title:   'show a history of this equipment' })+
                Button_HTML('P', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-print',
                    title:   'print a summary page on this equipment' }) +
                Button_HTML('url', {
                    name:    equipment.id,
                    classes: 'equipment-inventory-link',
                    title:   'persistent URL for this equipment' }) +
'    </div>' ;
                for(var j=0; j < num_attachments; j++) html += '<div style="float:right; margin-right:2px;"><img src="../irep/img/attachment.png" /></div> ' ;
                for(var j=0; j < num_tags;        j++) html += '<div style="float:right; font-weight:bold; margin-right:2px;">T</div> ' ;
                html +=
'    <div style="clear:both;"></div>' +
'  </div>'+
'  <div class="body">' +
'    <table><tbody>'+
'      <tr>' +
'        <td '+cell_left+' >SLAC ID :</td>' +
'        <td '+cell_right+' >'+equipment.slacid+'</td>' +
'        <td '+cell_left+' >Status :</td>' +
'        <td '+cell_right+' >'+equipment.status+'</td>' +
'        <td '+cell_left+' >Sub-status :</td>' +
'        <td '+cell_right+' >'+equipment.status2+'</td>' +
'        <td '+cell_left+' rowspan="4">' + (option_model_image ?
'          <div class="visible equipment-model-image" name="'+equipment.id+'" style="float:left; padding:5px;"><a class="link" href="../irep/equipment_model_attachments/'+equipment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/equipment_model_attachments/preview/'+equipment.id+'" width="102" height="72" /></a></div>' :
'          <div class="hidden equipment-model-image" name="'+equipment.id+'" style="float:left; padding:5px;"></div>') +
'          </td>' +
'      </tr>' +
'      <tr>' +
'        <td '+cell_left+' >Manuf :</td>' +
'        <td '+cell_right+' >'+equipment.manufacturer+'</td>' +
'        <td '+cell_left+' >Model : </td>' +
'        <td '+cell_right+' >'+equipment.model+'</td>' +
'        <td '+cell_left+' >Serial # :</td>' +
'        <td '+cell_right+' >'+equipment.serial+'</td>' +
'      </tr>' +
'      <tr>' +
'        <td '+cell_left+' >PC :</td>' +
'        <td '+cell_right+' >'+equipment.pc+'</td>' +
'        <td '+cell_left+' >Custodian :</td>' +
'        <td '+cell_right+' >'+equipment.custodian+'</td>' +
'        <td '+cell_left+' >Location :</td>' +
'        <td '+cell_right+' >'+equipment.location+'</td>' +
'      </tr>' +
'      <tr>' +
'        <td '+cell_left+' >Room :</td>' +
'        <td '+cell_right+' >'+equipment.room+'</td>' +
'        <td '+cell_left+' >Rack :</td>' +
'        <td '+cell_right+' >'+equipment.rack+'</td>' +
'        <td '+cell_left+' >Elevation :</td>' +
'        <td '+cell_right+' >'+equipment.elevation+'</td>' +
'      </tr>' +
'    </tbody></table>' +
            (equipment.description != '' ?
'    <div style="margin-top:5px; padding:10px; border-top:solid 1px #c0c0c0;">' +
'      <pre>'+equipment.description+'</pre>' +
'    </div>' : '') +
            (num_tags ? 
'    <div style="margin-top:5px; padding:5px; padding-top:10px; border-top:solid 1px #c0c0c0;">' +
'      <div style="float:left;" class="equipment-tag-hdr">Tags:</div>' +
       tags_html +
'      <div style="clear:both;"></div>' +
     '</div>' : '') +
            (num_attachments ?
'    <div style="margin-top:5px; padding:5px; padding-top:10px; border-top:solid 1px #c0c0c0;">'+attachments_html+'</div>' : '') +
'  </div>' +
'  <div class="footer">' + 
'    <table><tbody>'+
'      <tr>' +
'        <td '+cell_left+' >Last modified :</td>' +
'        <td '+cell_right+' >'+equipment.modified_time+'</td>' +
'        <td '+cell_left+' >By :</td>' +
'        <td '+cell_right+' >'+equipment.modified_uid+'</td>' +
'      </tr>' +
'    </tbody></table>' +
'  </div>' +
'</div>' ;
        }
        html +=
'<div style="clear:both;"></div>';
        elem.html(html) ;
        if (that.can_edit_inventory()) {
            elem.find('.equipment-inventory-delete').
                button().
                click(function () {
                    var id = this.name ;
                    that.equipment_delete(id, this) ;
                }) ;
            elem.find('.equipment-inventory-edit').
                button().
                click(function () {
                    var id = this.name ;
                    that.equipment_edit(id) ;
                }) ;
        }
        elem.find('.equipment-inventory-history').
            button().
            click(function () {
                var id = this.name ;
                that.equipment_history(id) ;
            }) ;
        elem.find('.equipment-inventory-print').
            button().
            click(function () {
                var id = this.name ;
                that.equipment_print(id) ;
            }) ;
        elem.find('.equipment-inventory-link').
            button().
            click(function () {
                var id = this.name ;
                that.equipment_url(id) ;
            }) ;
        this.tabs.tabs('refresh') ;
    } ;
    this.toggle_attachment = function (id) {
        var elem = $('#equipment-inventory-table') ;
        var tgl = elem.find('#equipment-attachment-tgl-'+id) ;
        var con = elem.find('#equipment-attachment-con-'+id) ;
        if (tgl.hasClass('ui-icon-triangle-1-e')) {
            tgl.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
            con.removeClass('hidden').addClass('visible') ;
            if (con.html() == '')
                con.html('<a class="link" href="../irep/equipment_attachments/'+id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab"><img src="../irep/equipment_attachments/preview/'+id+'" /></a>') ;
        } else {
            tgl.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
            con.removeClass('visible').addClass('hidden') ;
        }
    } ;
    this.equipment_delete = function (equipment_id, button_elem) {
        ask_yes_no (
            'Confirm Equipment Delete' ,
            'Are you sure you want to remove this equipment from the database? Note that this is irreversable operation.' ,
            function () {
                var button = $(button_elem) ;
                button.button('disable') ;
                var jqXHR = $.get('../irep/ws/equipment_delete.php', {equipment_id: equipment_id}, function (data) {
                    if (data.status != 'success') {
                        report_error(data.message, null) ;
                        button.button('enable') ;
                        return ;
                    }
                    for(var i in that.equipment) {
                        var equipment = that.equipment[i] ;
                        if (equipment.id == equipment_id) {
                            delete that.equipment_by_id[equipment.id] ;
                            delete that.equipment[i] ;
                        }
                    }
                    that.equipment_display() ;
                },
                'JSON').error(function () {
                    report_error('operation failed because of: '+jqXHR.statusText, null) ;
                    button.button('enable') ;
                }) ;
            }
        ) ;
    } ;
    this.equipment_edit = function (equipment_id) {
        web_service_GET ('../irep/ws/equipment_search_options.php', {}, function (data) {
            that.equipment_edit_impl(equipment_id, data.option) ;
        }) ;
    } ;
    this.equipment_properies_before_edit = null ;
    this.equipment_edit_impl = function (equipment_id, option) {

        var equipment = this.equipment_by_id[equipment_id] ;
        var panelId = 'edit_'+equipment_id ;

        // Check if the panel is already open. Activate it if found.
        //
        if (panelId in this.equipment_editing) {
            this.tabs.children('div').each(function (i) {
                if (this.id == panelId) that.equipment_select_tab(i) ;
            }) ;
            return ;
        }

        // Add a new panel to teh tab
        //
        this.tabs.find('.ui-tabs-nav').append (
'<li><a href="#'+panelId+'" style="color:red;">Editing...</a> <span class="ui-icon ui-icon-close edit">Remove Tab</span></li>'
        );
        var required_field_html = '<span style="color:red ; font-size:120% ; font-weight:bold ;"> * </span>' ;
        this.tabs.append (
'<div id="'+panelId+'">' +
'  <div style="border:solid 1px #b0b0b0; padding:20px;">' +
'    <div style="float:left; margin-bottom:10px; width:720px;">' +
'      <b>When editing equipment properties keep in mind the following:</b>' +
'      <ul>' +
'        <li>equipment searial number is unique for the given manufacturer and model</li>' +
'        <li>SLAC Property Control Number (PC) is unique</li>' +
'        <li>if the desired propertly location is not found in the dictionary and if you do not have' +
'            the dictionary privilege then contact administrators of this software to register new location' +
'            in the database</li>' +
'        <li>when changing a status of the equipment you will always be asked to provide' +
'            a comment on the operation when saving results of the editing session</li>' +
'        <li>a custodian can be selected from a list of known one or just added to the database by entering' +
'            a desired name into the corresponding input field. Note that the name put into the input field' +
'            will be the final name saved in the end of the editing session.</li>' +
'    </div>' +
'    <div style="float:left; padding:5px; margin-bottom:20px;">' +
'      <button name="save"          >Save</button>' +
'      <button name="save-w-comment">Save w/ Comment</button>' +
'      <button name="cancel"        >Cancel</button>' +
'    </div>' +
'    <div style="clear:both;"></div>' +

'    <div id="editor_tabs" style="font-size:12px;">' +

'      <ul>' +
'        <li><a href="#general_tab">General</a></li>' +
'        <li><a href="#attachments_tab">Attachments</a></li>' +
'        <li><a href="#tags_tab">Tags</a></li>' +
'      </ul>' +

'      <div id="general_tab" >' +
'        <div style=" border:solid 1px #b0b0b0; padding:20px;" >' +

'          <div style="margin-bottom:10px; padding-bottom:10px; border-bottom:dashed 1px #c0c0c0;">' +
'            <table><tbody>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Status</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><select name="status" class="equipment-edit-element"></select></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Sub-status</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><select name="status2" class="equipment-edit-element"></select></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >SLACid</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " >'+equipment.slacid+'</td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Manufacturer</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " >'+equipment.manufacturer+'</td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Model</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " >'+equipment.model+'</td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Serial # '+required_field_html+'</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><input type="text" name="serial" class="equipment-edit-element" size="20" style="padding:2px ;" value="'+equipment.serial+'" /></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >PC #</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><input type="text" name="pc" class="equipment-edit-element" size="20" style="padding:2px ;" value="'+equipment.pc+'" /></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Custodian</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " colspan="3">' +
'                  <input type="text" name="custodian" class="equipment-edit-element" size="20" style="padding:2px ;" value="" />' +
'                  ( other known custodians: <select name="known_custodians" class="equipment-edit-element"></select> )</td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Location</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><select name="location" class="equipment-edit-element"></select></td>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Room</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><select name="room" class="equipment-edit-element"></select></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " colspan="2" >&nbsp;</td>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Rack (Cabinet)</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><input type="text" name="rack" class="equipment-edit-element" value="" /></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " colspan="2" >&nbsp;</td>' +
'                <td class="table_cell table_cell_left  equipment-edit-cell " >Elevation (Shelf)</td>' +
'                <td class="table_cell table_cell_right equipment-edit-cell " ><input type="text" name="elevation" class="equipment-edit-element" value="" /></td>' +
'              </tr>' +
'              <tr>' +
'                <td class="table_cell table_cell_left  table_cell_bottom equipment-edit-cell" valign="top" >Description</td>' +
'                <td class="table_cell table_cell_right table_cell_bottom equipment-edit-cell" valign="top" colspan="3"><textarea cols=56 rows=4 name="description" class="equipment-edit-element" style="padding:4px ;" title="Here be an arbitrary description"></textarea></td>' +
'              </tr>' +
'            </tbody></table>' +
'          </div>' +
'          '+required_field_html+' required field' +
'        </div>' +
'      </div>' +

'      <div id="attachments_tab" >' +
'        <div style=" border:solid 1px #b0b0b0; padding:20px; padding-top:30px;" >' +
'          <div style="margin-bottom:20px; padding-left:10px;">' +
             Button_HTML('add more attachments', {
                 name:    equipment.id ,
                 classes: 'equipment-attachment-add' ,
                 title:   'click to add an attachment placeholder' }) +
'          </div>' +
'          <form enctype="multipart/form-data" action="../irep/ws/equipment_attachment_upload.php" method="post">' +
'            <input type="hidden" name="equipment_id" value="'+equipment.id+'" />' +
'            <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
'            <div class="attachments-new"></div>' +
'          </form>' +
'          <div class="attachments"></div>' +
'        </div>' +
'      </div>' +

'      <div id="tags_tab" >' +
'        <div style=" border:solid 1px #b0b0b0; padding:20px; padding-top:30px;" >' +
'          <div style="margin-bottom:20px; padding-left:10px;">' +
             Button_HTML('add more tags', {
                 name:    equipment.id ,
                 classes: 'equipment-tag-add' ,
                 title:   'click to add an placeholder for one more tag' }) +
'          </div>' +
'          <div class="tags-new"></div>' +
'          <div class="tags-old"></div>' +
'        </div>' +
'      </div>' +

'  </div>' +
'</div>'
        ) ;
        this.tabs.tabs('refresh') ;
        this.equipment_select_tab(-1) ;

        var panel_elem = $('#'+panelId) ;
        panel_elem.find('#editor_tabs').tabs() ;

        // Make a deep copy of the equipment object to decouple the editing session
        // from the search operations.
        //
        this.equipment_editing[panelId] = jQuery.extend (
            true ,
            {} ,
            that.equipment_by_id[equipment_id]
        ) ;

        // Buttons
        //
        panel_elem.find('button[name="save"]').button().click(function () {
            that.equipment_edit_save(panelId, equipment_id, false) ;
        }) ;
        panel_elem.find('button[name="save-w-comment"]').button().click(function () {
            that.equipment_edit_save(panelId, equipment_id, true) ;
        }) ;
        panel_elem.find('button[name="cancel"]').button().click(function () {
            that.equipment_edit_tab_close(panelId) ;
        }) ;

        // Finalize the form elements using the latest options
        //
        var html = '' ;
        var html2 = '' ;
        for (var i in option.status) {
            var status = option.status[i] ;
            html += '<option value="'+status.name+'">'+status.name+'</option>' ;
            if (equipment.status == status.name) {
                for (var j in status.status2) {
                    var status2 = status.status2[j] ;
                    html2 += '<option value="'+status2.name+'">'+status2.name+'</option>' ;
                }
            }
        }
        var status_elem = panel_elem.find('select[name="status"]') ;
        status_elem.html(html) ;
        status_elem.val(equipment.status) ;

        var status2_elem = panel_elem.find('select[name="status2"]') ;
        status2_elem.html(html2) ;
        status2_elem.val(equipment.status2) ;

        status_elem.change (function () {
            var selected_status = $(this).val() ;
            var html2 = '' ;
            for (var i in option.status) {
                var status = option.status[i] ;
                if (selected_status == status.name) {
                    for (var j in status.status2) {
                        var status2 = status.status2[j] ;
                        html2 += '<option value="'+status2.name+'">'+status2.name+'</option>' ;
                    }
                }
            }
            status2_elem.html(html2) ;
        }) ;

        html = '' ;
        html2 = '' ;
        var selected_location_id = 0 ;
        var selected_room_id = 0 ;
        for (var i in option.location) {
            var location = option.location[i] ;
            html += '<option value="'+location.id+'">'+location.name+'</option>' ;
            if (equipment.location == location.name) {
                selected_location_id = location.id ;
                for (var j in location.room) {
                    var room = location.room[j] ;
                    html2 += '<option value="'+room.id+'">'+room.name+'</option>' ;
                    if (equipment.room == room.name) selected_room_id = room.id ;
                }
            }
        }
        var location_elem = panel_elem.find('select[name="location"]') ;
        location_elem.html(html) ;
        location_elem.val(selected_location_id) ;

        var room_elem = panel_elem.find('select[name="room"]') ;
        room_elem.html(html2) ;
        room_elem.val(selected_room_id) ;

        location_elem.change (function () {
            var selected_location_id = $(this).val() ;
            var html2 = '' ;
            for (var i in option.location) {
                var location = option.location[i] ;
                if (selected_location_id == location.id) {
                    for (var j in location.room) {
                        var room = location.room[j] ;
                        html2 += '<option value="'+room.id+'">'+room.name+'</option>' ;
                    }
                }
            }
            room_elem.html(html2) ;
        }) ;

        var rack_elem = panel_elem.find('input[name="rack"]') ;
        rack_elem.val(equipment.rack) ;

        var elevation_elem = panel_elem.find('input[name="elevation"]') ;
        elevation_elem.val(equipment.elevation) ;

        var custodian_input_elem = panel_elem.find('input[name="custodian"]') ;
        custodian_input_elem.val(equipment.custodian) ;
        html = '' ;
        var selected_custodian = null ;
        for (var i in option.custodian) {
            var custodian = option.custodian[i] ;
            html += '<option value="'+custodian+'">'+custodian+'</option>' ;
            if (equipment.custodian == custodian) selected_custodian = custodian ;
        }
        var custodian_select_elem = panel_elem.find('select[name="known_custodians"]') ;
        custodian_select_elem.html(html) ;
        if (selected_custodian != null ) custodian_select_elem.val(equipment.custodian) ;
        custodian_select_elem.change(function () {
            custodian_input_elem.val($(this).val());
        });
        var description_elem = panel_elem.find('textarea[name="description"]') ;
        description_elem.val(equipment.description) ;

        // Attachment management

        html = '' ;
        for (var i in equipment.attachment) {
            var a = equipment.attachment[i] ;
            html +=
'<div id="'+a.id+'"  class="equipment-attachment-edit-entry" >' +
'  <div style="float:left; width:72px;">' +
            Button_HTML('delete', {
                name:    a.id,
                classes: 'equipment-attachment-delete visible',
                title:   'delete this attachment' }) +
            Button_HTML('un-delete', {
                name:    a.id,
                classes: 'equipment-attachment-cancel hidden',
                title:   'cancel previously made intent to delete this attachment' }) +
'  </div>' +
'  <div style="float:left;">' +
'    <a class="link" href="../irep/equipment_attachments/'+a.id+'/file" target="_blank" title="click on the image to open/download a full size attachment in a separate tab"><img src="../irep/equipment_attachments/preview/'+a.id+'" /></a>' +
'  </div>' +
'  <div style="float:left; margin-left:10px;">' +
'    <table><tbody>' +
'      <tr><td align="right"><b>name :</b></td><td> '+a.name+'</td></tr>' +
'      <tr><td align="right"><b>type :</b></td><td> '+a.document_type+'</td></tr>' +
'      <tr><td align="right"><b>size :</b></td><td> '+a.document_size_bytes+'</td></tr>' +
'      <tr><td align="right"><b>added :</b></td><td> '+a.create_time+'</td></tr>' +
'      <tr><td align="right"><b>by :</b></td><td> '+a.create_uid+'</td></tr>' +
'    </tbody></table>' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' ;
        }

        var attachments_elem = panel_elem.find('div.attachments') ;
        attachments_elem.html(html) ;

        this.equipment_new_attachment_counter = 1 ;

        var form_attachments_elem = panel_elem.find('form div.attachments-new') ;

        panel_elem.find('button.equipment-attachment-add').button().click(function () {
            var html =
'<div id="'+that.equipment_new_attachment_counter+'" class="equipment-attachment-new-edit-entry" >' +
'  <div style="float:left; width:72px;">' +
            Button_HTML('delete', {
                name:    that.equipment_new_attachment_counter ,
                classes: 'equipment-attachment-new-cancel',
                title:   'cancel previously made intent to add this attachment' }) +
'  </div>' +
'  <div style="float:left;">' +
'    <input type="file" name="file2attach_'+that.equipment_new_attachment_counter+'" onchange="equipment.equipment_attachment_added('+panelId+','+that.equipment_new_attachment_counter+')" />' +
'    <input type="hidden" name="file2attach_'+that.equipment_new_attachment_counter+'" value="" />' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' ;
            form_attachments_elem.prepend(html) ;
            form_attachments_elem.find('button.equipment-attachment-new-cancel[name="'+that.equipment_new_attachment_counter+'"]').button().click(function () {
                var counter = this.name ;
                form_attachments_elem.find('div.equipment-attachment-new-edit-entry#'+counter).remove() ;
            }) ;
            that.equipment_new_attachment_counter++ ;
        }) ;
        attachments_elem.find('button.equipment-attachment-delete').button().click(function () {
            var attachment_id = this.name ;
            $(this).removeClass('visible').addClass('hidden') ;
            attachments_elem.find('button.equipment-attachment-cancel[name="'+attachment_id+'"]').removeClass('hidden').addClass('visible') ;
            $('div.equipment-attachment-edit-entry#'+attachment_id).addClass('equipment-edit-entry-modified') ;
            that.tabs.tabs('refresh') ;
        }) ;
        attachments_elem.find('button.equipment-attachment-cancel').button().click(function () {
            var attachment_id = this.name ;
            $(this).removeClass('visible').addClass('hidden') ;
            attachments_elem.find('button.equipment-attachment-delete[name="'+attachment_id+'"]').removeClass('hidden').addClass('visible') ;
            $('div.equipment-attachment-edit-entry#'+attachment_id).removeClass('equipment-edit-entry-modified') ;
            that.tabs.tabs('refresh') ;
        }) ;


        // -----------------
        //   New tags
        // -----------------

        var tags_new_elem = panel_elem.find('div.tags-new') ;

        panel_elem.find('button.equipment-tag-add').button().click(function () {
            tags_new_elem.prepend (
'<div class="equipment-tag-new-edit-entry" >' +
'  <div style="float:left; width:72px;">' +
     Button_HTML('delete', {
        title:   'cancel previously made intent to add this tag' }) +
'  </div>' +
'  <div style="float:left;">' +
'    <input type="text" value="" />' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>'
            ) ;
            var elem = tags_new_elem.find('div.equipment-tag-new-edit-entry').first() ;
            elem.find('input').change(function () {
                if ($(this).val() == '') elem.removeClass('equipment-edit-entry-modified') ;
                else                     elem.addClass   ('equipment-edit-entry-modified') ;
            }) ;
            elem.find('button').button().click (function () {
                elem.remove() ;
            }) ;
        }) ;


        // -----------------
        //   Existing tags
        // -----------------

        var tags_old_elem = panel_elem.find('div.tags-old') ;

        html = '' ;
        for (var i in equipment.tag) {
            var t = equipment.tag[i] ;
            html +=
'<div id="'+t.id+'" class="equipment-tag-edit-entry" >' +
'  <div style="float:left; width:72px;">' +
            Button_HTML('delete', {
                name:    t.id,
                classes: 'equipment-tag-delete visible',
                title:   'delete this tag' }) +
            Button_HTML('un-delete', {
                name:    t.id,
                classes: 'equipment-tag-cancel hidden',
                title:   'cancel previously made intent to delete this tag' }) +
'  </div>' +
'  <div style="margin-left:10px; padding-left:20px; font-weight:bold;">'+t.name+'</div>' +
'</div>' ;
        }
        tags_old_elem.html(html) ;
        tags_old_elem.find('button.equipment-tag-delete').button().click(function () {
            var tag_id = this.name ;
            $(this).removeClass('visible').addClass('hidden') ;
            tags_old_elem.find('button.equipment-tag-cancel[name="'+tag_id+'"]').removeClass('hidden').addClass('visible') ;
            $(this).parentsUntil('div.equipment-tag-edit-entry').parent().addClass('equipment-edit-entry-modified') ;
        }) ;
        tags_old_elem.find('button.equipment-tag-cancel').button().click(function () {
            var tag_id = this.name ;
            $(this).removeClass('visible').addClass('hidden') ;
            tags_old_elem.find('button.equipment-tag-delete[name="'+tag_id+'"]').removeClass('hidden').addClass('visible') ;
            $(this).parentsUntil('div.equipment-tag-edit-entry').parent().removeClass('equipment-edit-entry-modified') ;
        }) ;


        // Track changes in the edited fields and highlight rows where changes
        // occur.

        this.equipment_properies_before_edit = {
            'status'           : equipment.status ,
            'status2'          : equipment.status2 ,
            'serial'           : equipment.serial ,
            'pc'               : equipment.pc ,
            'location'         : equipment.location ,
            'room'             : equipment.room ,
            'rack'             : equipment.rack ,
            'elevation'        : equipment.elevation ,
            'custodian'        : equipment.custodian ,
            'known_custodians' : equipment.custodian ,
            'description'      : equipment.description
        } ;
        panel_elem.find('.equipment-edit-element').change(function() {
            var name = $(this).attr('name') ;
            // Special processing for locations because they're indexed in <select> with
            // their numeric identifiers not names. Though, we still have names shown in
            // the interface.
            //
            var val = $(this).val() ;
            if (name == 'location') {
                for (var i in option.location) {
                    var location = option.location[i] ;
                    if (val == location.id) {
                        val = location.name ;
                        break ;
                    }
                }
            }
            $(this).closest('tr').css('background-color', that.equipment_properies_before_edit[name] == val ? '' : '#FFDCDC') ;
        }) ;
    } ;
    this.equipment_attachment_added = function (panelId, counter) {
        // 
        // TODO: I still don't unmderstand why am I getting an HTML DOM element
        // instead of a number here?
        //
        var panel_elem = $('div#'+panelId.id) ;
        var form_attachments_elem = panel_elem.find('form div.attachments-new') ;
        var filename = form_attachments_elem.find('input[name="file2attach_'+counter+'"]').val() ;
        panel_elem.find('div.equipment-attachment-new-edit-entry#'+counter).css('background-color', (filename == '' ? '' : '#ffdcdc')) ;
    } ;
    this.equipment_edit_save = function (panelId, equipment_id, with_comment) {
        var equipment = this.equipment_by_id[equipment_id] ;
        var panel_elem = $('#'+panelId) ;
        with_comment = with_comment || (panel_elem.find('select[name="status"]').val() != equipment.status) || (panel_elem.find('select[name="status2"]').val() != equipment.status2) ;
        if (with_comment) {
            ask_for_input ('Comment Request', 'Please, provide a comment for the operation:', function (comment) {
                that.equipment_edit_save_impl(panelId, equipment_id, comment) ;
            }) ;
            return ;
        }
        this.equipment_edit_save_impl(panelId, equipment_id, '') ;
    } ;
    this.equipment_edit_save_impl = function (panelId, equipment_id, comment) {
        
        // This procedure will do two-stage commit
        //
        // 1: upload attachments
        // 2: update other parameters of an equipment and delete attachments which were
        //    marked as deleted in the 'Attachments' tab.
        //
        // Note that the firest stage (if succeeded) can not be reverted.
        
        var equipment                = this.equipment_by_id[equipment_id] ;
        var panel_elem               = this.tabs.find('div#'+panelId) ;
        var save_button              = panel_elem.find('button[name="save"]').button() ;
        var save_with_comment_button = panel_elem.find('button[name="save-w-comment"]').button() ;
        var cancel_button            = panel_elem.find('button[name="cancel"]').button() ;

        save_button.button('disable') ;
        save_with_comment_button.button('disable') ;
        cancel_button.button('disable') ;

        panel_elem.find('form').ajaxSubmit({

            success: function(data) {
                if (data.status != 'success') {
                    report_error(data.message) ;
                    save_button.button('enable') ;
                    save_with_comment_button.button('enable') ;
                    cancel_button.button('enable') ;
                    return ;
                }
                that.equipment_edit_save_attributes (

                    equipment.id ,
                    panel_elem ,
                    comment ,

                    function (data) {

                        that.equipment_edit_tab_close(panelId) ;

                        // If the modified equipment is still in thej local search set then update
                        // the set and redisplay the search results table.
                        //
                        for(var i in data.equipment) {
                            var new_equipment = data.equipment[i] ;
                            if (new_equipment.id == equipment_id) {
                                for(var j in that.equipment) {
                                    var equipment = that.equipment[j] ;
                                    if (equipment.id == equipment_id) {
                                        that.equipment[j] = new_equipment ;
                                        that.equipment_by_id[equipment_id] = new_equipment ;
                                        that.equipment_display() ;
                                        break ;
                                    }
                                }
                                break ;
                            }
                        }
                    } ,

                    function () {
                        save_button.button('enable') ;
                        save_with_comment_button.button('enable') ;
                        cancel_button.button('enable') ;
                    }
                ) ;
            } ,

            error: function() {
                report_error('failed to contact the server in order to upload attachment(s)') ;
                save_button.button('enable') ;
                save_with_comment_button.button('enable') ;
                cancel_button.button('enable') ;
            } ,

            dataType: 'json'
        }) ;
    } ;
    this.equipment_edit_save_attributes = function(equipment_id, panel_elem, comment, on_success, on_error) {
        var tags2add = [] ;
        panel_elem.find('div.tags-new').children('div.equipment-tag-new-edit-entry').each(function () {
            var name = $(this).find('input').val() ;
            if (name != '') tags2add.push(name) ;
        }) ;
        var tags2remove = [] ;
        panel_elem.find('div.tags-old').children('div.equipment-tag-edit-entry.equipment-edit-entry-modified').each(function () {
            var tag_id = $(this).attr('id') ;
            tags2remove.push(tag_id) ;
        }) ;
        var attachments2remove = [] ;
        panel_elem.find('div.attachments').children('div.equipment-attachment-edit-entry.equipment-edit-entry-modified').each(function () {
            var attachment_id = $(this).attr('id') ;
            attachments2remove.push(attachment_id) ;
        }) ;
        var params = {
            equipment_id:       equipment_id ,
            status:             panel_elem.find('select[name="status"]').val() ,
            status2:            panel_elem.find('select[name="status2"]').val() ,
            serial:             panel_elem.find('input[name="serial"]').val() ,
            pc:                 panel_elem.find('input[name="pc"]').val() ,
            location_id:        panel_elem.find('select[name="location"]').val() ,
            room_id:            panel_elem.find('select[name="room"]').val() ,
            rack:               panel_elem.find('input[name="rack"]').val() ,
            elevation:          panel_elem.find('input[name="elevation"]').val() ,
            custodian:          panel_elem.find('input[name="custodian"]').val() ,
            description:        panel_elem.find('textarea[name="description"]').val() ,
            comment:            comment ,
            tags2add:           JSON.stringify(tags2add) ,
            tags2remove:        JSON.stringify(tags2remove) ,
            attachments2remove: JSON.stringify(attachments2remove)
        } ;
        var jqXHR = $.post('../irep/ws/equipment_update.php', params, function (data) {
            if (data.status != 'success') {
                report_error(data.message, null) ;
                on_error() ;
            } else {
                on_success(data) ;
            }
        },
        'JSON').error(function () {
            report_error('saving failed because of: '+jqXHR.statusText, null) ;
            on_error() ;
        }) ;
    } ;
    this.equipment_edit_tab_close = function (panelId) {
        $('#equipment-inventory').find('#tabs').find('a[href="#'+panelId+'"]').closest('li').remove() ;
        $('#'+panelId).remove() ;
        this.tabs.tabs('refresh') ;
        delete that.equipment_editing[panelId] ;
    } ;
    this.equipment_history = function (equipment_id) {
        var equipment = this.equipment_by_id[equipment_id] ;
        var panelId = 'history_'+equipment_id ;

        // Check if the panel is already open. Activate it if found.
        //
        if (panelId in this.equipment_history) {
            this.tabs.children('div').each(function (i) {
                if (this.id == panelId) that.equipment_select_tab(i) ;
            }) ;
            return ;
        }

        // Add a new panel to the tab
        //
        this.tabs.find('.ui-tabs-nav').append (
'<li><a href="#'+panelId+'">History</a> <span class="ui-icon ui-icon-close history">Remove Tab</span></li>'
        );
        this.tabs.append (
'<div id="'+panelId+'">'+
'  <div style="border:solid 1px #b0b0b0; padding:20px; padding-left:10px;">'+
'    <div style="float:left; max-width:720px;">'+
'      <table><tbody>'+
'        <tr><td class="table_cell table_cell_left  " style="border:0; padding-right:0px;" >Manufacturer</td>'+
'            <td class="table_cell table_cell_right " style="border:0; padding-right:10px;" >'+equipment.manufacturer+'</td>'+
'            <td class="table_cell table_cell_left  " style="border:0; padding-right:0px;" >Model</td>'+
'            <td class="table_cell table_cell_right " style="border:0; padding-right:10px;" >'+equipment.model+'</td>'+
'            <td class="table_cell table_cell_left  " style="border:0; padding-right:0px;" >Serial #</td>'+
'            <td class="table_cell table_cell_right " style="border:0; padding-right:10px;" >'+equipment.serial+'</td></tr>'+
'      </tbody></table>'+
'      <div style="margin-top:10px;" id="equipment-history-table-'+equipment_id+'"></div>'+
'    </div>'+
'    <div style="float:left; margin-left:20px;">'+
'      <button name="close">Close</button>'+
'    </div>'+
'    <div style="clear:both ;"></div>'+
'  </div>'+
'</div>'
        ) ;
        this.tabs.tabs('refresh') ;
        this.equipment_select_tab(-1) ;

        // Make a deep copy of the equipment object to decouple the history session
        // from the search operations.
        //
        this.equipment_history[panelId] = jQuery.extend (
            true ,
            {} ,
            that.equipment_by_id[equipment_id]
        ) ;

        var panel_elem = $('#'+panelId) ;
        panel_elem.find('button[name="close"]').button().click(function () {
            that.equipment_history_tab_close(panelId) ;
        }) ;
        
        // Now add the table of events
        //
        var hdr = [
            {   name: 'time', style: ' white-space: nowrap;' } ,
            {   name: 'event' } ,
            {   name: 'user' } ,
            {   name: 'comments', hideable: true, sorted: false }
        ] ;
        rows = [] ;
        this.equipment_history[panelId].table = new Table (
            'equipment-history-table-'+equipment_id ,
            hdr, rows ,
            {} ,
            config.handler('inventory', 'equipment_history_table')
        );
        this.equipment_history[panelId].table.display() ;
        this.equipment_history[panelId].table.erase(Table.Status.Loading) ;
        web_service_GET (
            '../irep/ws/history_get.php' ,
            {equipment_id: equipment_id} ,
            function (data) {
                var rows = [] ;
                for (var i in data.history) {
                    var e = data.history[i] ;
                    var comments = '' ;
                    for (var j in e.comments) {
                        var c = e.comments[j] ;
                        if ((comments != '') && (c != '')) comments += '<br>' ;
                        comments += c;
                    }
                    rows.push (
                        [
                            e.event_time ,
                            e.event ,
                            e.event_uid ,
                            comments
                        ]
                    ) ;
                }
                that.equipment_history[panelId].table.load(rows) ;
            }
        ) ;
    } ;
    this.equipment_history_tab_close = function (panelId) {
        $('#equipment-inventory').find('#tabs').find('a[href="#'+panelId+'"]').closest('li').remove() ;
        $('#'+panelId).remove() ;
        this.tabs.tabs('refresh') ;
        delete that.equipment_history[panelId] ;
    } ;
    this.equipment_print = function (id) {
        alert('here be the print action') ;
    } ;
    this.equipment_url = function (id) {
        var url = window.location.href ;
        var idx = url.indexOf('?') ;
        url = (idx == -1 ? url  : url.substr(0, idx))+'?equipment_id='+id ;
        window.open(url,'_blank') ;
    } ;
    this.equipment_search = function () {
        var form_elem = $('#equipment-inventory-form') ;
        var params = {
            status:          form_elem.find('select[name="status"]').val() ,
            status2:         form_elem.find('select[name="status2"]').val() ,
            manufacturer_id: form_elem.find('select[name="manufacturer"]').val() ,
            model_id:        form_elem.find('select[name="model"]').val() ,
            serial:          form_elem.find('input[name="serial"]').val() ,
            pc:              form_elem.find('input[name="pc"]').val() ,
            location_id:     form_elem.find('select[name="location"]').val() ,
            custodian:       form_elem.find('select[name="custodian"]').val() ,
            tag:             form_elem.find('select[name="tag"]').val() ,
            description:    form_elem.find('input[name="description"]').val()
        } ;
        this.equipment_search_impl(params) ;
    } ;
    this.equipment_search_impl = function (params) {
        var inventory_controls = $('#equipment-inventory-controls') ;
        var search_button      = inventory_controls.find('button[name="search"]') ;
        var reset_button       = inventory_controls.find('button[name="reset"]') ;
        search_button.button('disable') ;
        reset_button.button('disable') ;

        var jqXHR = $.post('../irep/ws/equipment_search.php', params, function (data) {
            search_button.button('enable') ;
            reset_button.button ('enable') ;
            if (data.status != 'success') {
                report_error(data.message, null) ;
                return ;
            }
            that.equipment = data.equipment ;
            that.equipment_by_id = {} ;
            for(var i in that.equipment) {
                var equipment = that.equipment[i] ;
                that.equipment_by_id[equipment.id] = equipment ;
            }
            that.equipment_display() ;
        },
        'JSON').error(function () {
            report_error('saving failed because of: '+jqXHR.statusText, null) ;
            search_button.button('enable') ;
            reset_button.button ('enable') ;
        }) ;
    } ;
    this.equipment_search_reset = function () {
        var form_elem = $('#equipment-inventory-form') ;
        form_elem.find('select[name="status"]'      ).val(0) ;
        form_elem.find('select[name="status2"]'     ).val(0) ;
        form_elem.find('select[name="manufacturer"]').val(0) ;
        form_elem.find('select[name="model"]'       ).val(0) ;
        form_elem.find('input[name="serial"]'       ).val('') ;
        form_elem.find('input[name="pc"]'           ).val('') ;
        form_elem.find('input[name="slacid"]'       ).val('') ;
        form_elem.find('select[name="location"]'    ).val(0) ;
        form_elem.find('select[name="custodian"]'   ).val('') ;
        form_elem.find('select[name="tag"]'         ).val('') ;
        form_elem.find('input[name="description"]'  ).val('') ;
        this.equipment = [] ;
        this.equipment_display() ;
    } ;

    /* ----------------------------------
     *   Adding equipmenmt to inventory
     * ----------------------------------
     */
    this.init_add = function () {

        that.create_form_changed = false ;

        var form_elem            = $('#equipment-add-form') ;
        var manufacturer_elem    = form_elem.find('select[name="manufacturer"]') ;
        var model_elem           = form_elem.find('select[name="model"]') ;
        var serial_elem          = form_elem.find('input[name="serial"]') ;
        var pc_elem              = form_elem.find('input[name="pc"]') ;
        var slacid_elem          = form_elem.find('input[name="slacid"]') ;
        var location_elem        = form_elem.find('select[name="location"]') ;
        var room_elem            = form_elem.find('select[name="room"]') ;
        var rack_elem            = form_elem.find('input[name="rack"]') ;
        var elevation_elem       = form_elem.find('input[name="elevation"]') ;
        var custodian_elem       = form_elem.find('input[name="custodian"]') ;
        var custodian_known_elem = form_elem.find('select[name="custodian"]') ;
        var description_elem     = form_elem.find('textarea[name="description"]') ;

        model_elem.attr ('disabled', 'disabled') ;
        serial_elem.attr('disabled', 'disabled') ;

        web_service_GET ('../irep/ws/equipment_search_options.php', {}, function (data) {

            var html = '<option value="0"></option>' ;
            for (var i in data.option.manufacturer) {
                var manufacturer = data.option.manufacturer[i] ;
                html += '<option value="'+manufacturer.id+'">'+manufacturer.name+'</option>' ;
            }
            manufacturer_elem.html(html) ;
            manufacturer_elem.change(function () {
                var manufacturer_id = this.value ;
                if (manufacturer_id == 0) {
                    model_elem.val(0) ;
                    model_elem.attr ('disabled', 'disabled') ;
                    serial_elem.val('') ;
                    serial_elem.attr('disabled', 'disabled') ;
                    return ;
                }
                for (var i in data.option.manufacturer) {
                    var manufacturer = data.option.manufacturer[i] ;
                    if (manufacturer_id == manufacturer.id) {
                        var html = '<option value="0"></option>' ;
                        for (var j in manufacturer.model) {
                            var model = manufacturer.model[j] ;
                            html += '<option value="'+model.id+'">'+model.name+'</option>' ;
                        }
                        model_elem.html(html) ;
                        model_elem.removeAttr('disabled') ;
                        break ;
                    }
                }
            }) ;

            var html = '<option value="0"></option>' ;
            model_elem.html(html) ;
            model_elem.change(function () {
                var model_id = this.value ;
                if (model_id == 0) {
                    serial_elem.val('') ;
                    serial_elem.attr('disabled', 'disabled') ;
                } else {
                    serial_elem.removeAttr('disabled') ;
                }
            }) ;

            serial_elem.val('') ;
            pc_elem.val('') ;
            slacid_elem.val('') ;

            var html  = '<option value="0"></option>' ;
            for (var i in data.option.location) {
                var location = data.option.location[i] ;
                html += '<option value="'+location.id+'">'+location.name+'</option>' ;
            }
            location_elem.html(html) ;

            var html2 = '<option value="0"></option>' ;
            room_elem.html(html2) ;

            location_elem.change (function () {
                var selected_location_id = $(this).val() ;
                var html2 = '<option value="0"></option>' ;
                for (var i in data.option.location) {
                    var location = data.option.location[i] ;
                    if (selected_location_id == location.id) {
                        for (var j in location.room) {
                            var room = location.room[j] ;
                            html2 += '<option value="'+room.id+'">'+room.name+'</option>' ;
                        }
                    }
                }
                room_elem.html(html2) ;
            }) ;
            rack_elem.val('') ;
            elevation_elem.val('') ;

            custodian_elem.val(global_current_user.uid) ;
            var html = '<option value="'+global_current_user.uid+'"></option>' ;
            for (var i in data.option.custodian) {
                var custodian = data.option.custodian[i] ;
                html += '<option value="'+custodian+'">'+custodian+'</option>' ;
            }
            custodian_known_elem.html(html) ;
            custodian_known_elem.change(function () {
                var custodian = this.value ;
                custodian_elem.val(custodian) ;
            }) ;

            description_elem.val('') ;

            that.add_form_validate() ;
        }) ;
    } ;
    
    this.add_form_validate = function () {

        var button_save       = $('#equipment-add-save') ;
        var form_elem         = $('#equipment-add-form') ;
        var manufacturer_elem = form_elem.find('select[name="manufacturer"]') ;
        var model_elem        = form_elem.find('select[name="model"]') ;
        var slacid_elem       = form_elem.find('input[name="slacid"]') ;
        var slacid_info_elem  = slacid_elem.next  ('span.form_element_info') ;

        var manufacturer_id = parseInt(manufacturer_elem.val()) ;
        var model_id        = parseInt(model_elem.val()) ;
        var slacid          = parseInt(slacid_elem.val()) ;

        manufacturer_elem.next('span.form_element_info').html(manufacturer_id ? '' : '&larr; please select') ; 
        model_elem.next       ('span.form_element_info').html(model_id        ? '' : '&larr; please select') ; 
        if (!slacid) {
            slacid_info_elem.html('&larr; please fill and press ENTER to validate') ;
            button_save.button (
                manufacturer_id && model_id && slacid ?
                'enable' :
                'disable'
            ) ;
        } else {
            web_service_GET (
                '../irep/ws/slacid_validate.php' ,
                {slacid: slacid} ,
                function (data) {
                    if (data.slacid.in_use) {
                        slacid_info_elem.html('&larr; the identifier is already in use') ;
                        slacid = 0 ;
                    } else {
                        if (data.slacid.is_valid ) {
                            slacid_info_elem.html('') ;
                        } else {
                            slacid_info_elem.html('&larr; the identifier is not allowed, please correct') ;
                            slacid = 0 ;
                        }
                    }
                    button_save.button (
                        manufacturer_id && model_id && slacid ?
                        'enable' :
                        'disable'
                    ) ;
                } ,
                function () {
                    slacid_info_elem.html('&larr; the validation failed, but you may still proceed with saving results') ;
                    button_save.button (
                        manufacturer_id && model_id && slacid ?
                        'enable' :
                        'disable'
                    ) ;
                }
            ) ;
        }

        this.create_form_changed = manufacturer_id || model_id || slacid ;
    } ;
    
    this.equipment_save = function () {
        var form_elem = $('#equipment-add-form') ;
        var params = {
            model_id:    form_elem.find('select[name="model"]').val() ,
            serial:      form_elem.find('input[name="serial"]').val() ,
            pc:          form_elem.find('input[name="pc"]').val() ,
            slacid:      form_elem.find('input[name="slacid"]').val() ,
            location_id: form_elem.find('select[name="location"]').val() ,
            room_id:     form_elem.find('select[name="room"]').val() ,
            rack:        form_elem.find('input[name="rack"]').val() ,
            elevation:   form_elem.find('input[name="elevation"]').val() ,
            custodian:   form_elem.find('input[name="custodian"]').val() ,
            description: form_elem.find('textarea[name="description"]').val()
        } ;
        var jqXHR = $.post('../irep/ws/equipment_new.php', params, function (data) {
            if (data.status != 'success') {
                report_error(data.message, null) ;
                $('#equipment-add-save').button('enable') ;
                return ;
            }
            that.init_add() ;
        },
        'JSON').error(function () {
            $('#equipment-add-save').button('enable') ;
            report_error('saving failed because of: '+jqXHR.statusText, null) ;
        }) ;
    } ;
}
var equipment = new p_appl_equipment() ;