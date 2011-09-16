class ModuleConfig( object ):
    """Place to store configuration of a given pyana module.
    """
    def __init__(self, name, address = None):
        self.name = name
        self.address = address

        self.label = "%s" % (name)
        if address is not None: 
            self.label = "%s:%s" % (name,address)
            
        self.options = {}
        self.options["source"] = self.address
            
    def dump(self, indent = ""):
        print "%sName:    %s"%(indent, self.name)
        print "%sAddress: %s"%(indent, self.address)
        for okey,oval in self.options.iteritems():
            print "%s  %s = %s"%(indent, okey,oval)

    def config_snippet(self):
        text = "[XtcExplorer.%s]\n" % self.name

        if self.address :
            text = "[XtcExplorer.%s:%s]\n" % (self.name,self.address)
        
        optkeys = self.options.keys()
        optkeys.sort()
        for n in optkeys:
            if self.options[n] is not None: 
                text += "%s = %s \n" % (n, self.options[n])
            else :
                text += "%s = \n" % n
            
        return text

class ImageModConfig( ModuleConfig ):
    """Class ImageModConfig

    Place to store configuration information for a pyana_image module
    """ 
    def __init__(self, address = None):
        ModuleConfig.__init__(self, "pyana_image_beta", address )
        # list of quantities to plot:
        self.quantities = []
        
        # set defaults:
        self.options["dark_img_file"] = None
        self.options["out_avg_file"] = None
        self.options["out_shot_file"] = None
        self.options["threshold"] = None

    def add_modifier(self,quantity,modifier):
        """Add modifiers to a plot quantity
        @param quantity        plot quantity name
        @param modifier        a string
        """
        item = (qnt for qnt in self.options["quantities"].split() if qnt.startswith(quantity) ).next()
        newconfigline = self.options["quantities"].replace(item,"%s:%s"%\
                                                           (quantity,modifier))
#                                                           (quantity,''.join(str(modifier,).split())))
        self.options["quantities"] = newconfigline

    def set_opt_quantities(self, quantity, value):
        """Set plot quantities by name
        """
        if value > 0 : 
            self.quantities.append(quantity)
        else :
            self.quantities.remove(quantity)
        self.options["quantities"] = " ".join(self.quantities)
            
    def set_opt_imXY(self, value):
        self.set_opt_quantities("image",value)

    def set_opt_roi(self, value):
        self.set_opt_quantities("roi",value)

    def set_opt_spectr(self, value):
        self.set_opt_quantities("spectrum",value)

    def set_opt_projX(self, value):
        self.set_opt_quantities("projX",value)

    def set_opt_projY(self, value):
        self.set_opt_quantities("projY",value)
        
    def set_opt_projR(self, value):
        self.set_opt_quantities("projR",value)


class ScanModConfig( ModuleConfig ):
    """Class ScanModConfig

    Place to store configuration information for a pyana_scan module
    """
    def __init__(self, address = None):
        ModuleConfig.__init__(self, "pyana_scan", address)
        self.options["controlpv"] = self.address
        self.options["input_epics"] = None
        self.options["input_scalars"] = None


class IpimbModConfig( ModuleConfig ):
    """Class IpimbModConfig

    Place to store configuration information for a pyana_ipimb module
    """
    def __init__(self, address = None):
        ModuleConfig.__init__(self, "pyana_ipimb_beta", address )
        self.options["quantities"] = "fex:pos fex:sum fex:channels"
        self.quantities = []

    def set_opt_quantities(self, quantity, value):
        """Set plot quantities by name
        """
        if value > 0 : 
            self.quantities.append(quantity)
        else :
            self.quantities.remove(quantity)
        self.options["quantities"] = " ".join(self.quantities)

    def set_opt_chRaw(self, value):
        self.set_opt_quantities("raw:channels",value)

    def set_opt_ch0(self, value):
        self.set_opt_quantities("raw:ch0",value)

    def set_opt_ch1(self, value):
        self.set_opt_quantities("raw:ch1",value)

    def set_opt_ch2(self, value):
        self.set_opt_quantities("raw:ch2",value)

    def set_opt_ch3(self, value):
        self.set_opt_quantities("raw:ch3",value)

    def set_opt_chVolt(self, value):
        self.set_opt_quantities("raw:channelsV",value)

    def set_opt_ch0Volt(self, value):
        self.set_opt_quantities("raw:ch0volt",value)

    def set_opt_ch1Volt(self, value):
        self.set_opt_quantities("raw:ch1volt",value)

    def set_opt_ch2Volt(self, value):
        self.set_opt_quantities("raw:ch2volt",value)

    def set_opt_ch3Volt(self, value):
        self.set_opt_quantities("raw:ch3volt",value)

    def set_opt_chFex(self, value):
        self.set_opt_quantities("fex:channels",value)

    def set_opt_ch0Fex(self, value):
        self.set_opt_quantities("fex:ch0",value)

    def set_opt_ch1Fex(self, value):
        self.set_opt_quantities("fex:ch1",value)

    def set_opt_ch2Fex(self, value):
        self.set_opt_quantities("fex:ch2",value)

    def set_opt_ch3Fex(self, value):
        self.set_opt_quantities("fex:ch3",value)

    def set_opt_sum(self, value):
        self.set_opt_quantities("fex:sum",value)

    def set_opt_pos(self, value):
        self.set_opt_quantities("fex:pos",value)

    def set_opt_posX(self, value):
        self.set_opt_quantities("fex:posx",value)

    def set_opt_posY(self, value):
        self.set_opt_quantities("fex:posy",value)


class WaveformModConfig( ModuleConfig ):
    """Class WaveformModConfig

    Place to store configuration information for a pyana_waveform module
    """
    def __init__(self, address = None):
        ModuleConfig.__init__(self, "pyana_waveform", address)

class EncoderModConfig( ModuleConfig ):
    """Class EncoderModConfig
    """
    def __init__(self, address = None):
        ModuleConfig.__init__(self,"pyana_encoder", address)

class EpicsModConfig( ModuleConfig ):
    """Class EpicsModConfig
    """
    def __init__(self, address = None):
        ModuleConfig.__init__(self,"pyana_epics", address )


class BldModConfig( ModuleConfig ):
    """Class BldModConfig
    (singleton?)
    """
    def __call__(self):
        return self
    
    def __init__(self, address = None):
        ModuleConfig.__init__(self,"pyana_bld", address )
        self.label = self.name

        self.options["do_ebeam"] = "False"
        self.options["do_gasdetector"] = "False"
        self.options["do_phasecavity"] = "False"

class PlotterModConfig( ModuleConfig ):
    def __init__(self, address = None):
        ModuleConfig.__init__(self,"pyana_plotter_beta", address)
        del self.options["source"]

        self.plot_n = "100"
        self.accum_n = None
        self.displaymode = None
        self.ipython = None

        self.options["display_mode"] = self.displaymode
        self.options["ipython"] = self.ipython
        if self.plot_n is not None : 
            self.options["plot_every_n"] = self.plot_n
        if self.accum_n is not None :
            self.options["accumulate_n"] = self.accum_n

        
class JobConfig( object ):
    """Place to store pyana configuration
    """
    def __init__(self) :
        
        # assume all events        
        self.run_n = None
        self.skip_n = None
        self.num_cpu = None
        

class Configuration( object ):
    """Class Configuration
    Place to store pyana configuration
    """
    def __init__(self, file = None) :
        """Constructor
        @param file   name of pyana.cfg configuration file
        """
        
        self.jobconfig = JobConfig()
        self.plotconfig = PlotterModConfig()
        self.file = file
        self.config_text = None

        # keep list of modules, key is device address
        self.modules = {}
        self.scan = False

        # make a module configuration object, based on key
        # returns a module configuratio object with a unique label
        # (except if singleton, like BldInfo data, return the existing instance)
        self.make_modconf = { 'DeviceName' : ModuleConfig,
                              'ControlPV'  : ScanModConfig,
                              'EpicsPV'    : EpicsModConfig,
                              
                              'Cspad'      : ImageModConfig,
                              'Cspad2x2'   : ImageModConfig,
                              'TM6740'     : ImageModConfig, 
                              'Opal1000'   : ImageModConfig, 
                              'Fccd'       : ImageModConfig, 
                              'Princeton'  : ImageModConfig,
                              'pnCCD'      : ImageModConfig,
                              'YAG'        : ImageModConfig, 
                              
                              'Acq'        : WaveformModConfig,
                              'ETof'       : WaveformModConfig,
                              'ITof'       : WaveformModConfig,
                              'Mbes'       : WaveformModConfig,
                              #'Camp'      : WaveformModConfig,
                              
                              'Ipimb'      : IpimbModConfig,
                              'IPM'        : IpimbModConfig,
                              'DIO'        : IpimbModConfig,
                              
                              'Encoder'    : EncoderModConfig,
                              
                              'EBeam'      :     BldModConfig,
                              'FEEGasDetEnergy': BldModConfig,
                              'PhaseCavity':     BldModConfig,
                              
                              'NotImplementedYet' : ModuleConfig }
        
        
    def add_to_scan(self,checkbox_label):
        if not self.scan:
            print "ERROR: add_to_scan called, but scan is set to False", self.scan
        
        self.update_text()

    def get_key(self, checkbox_label):
        key = checkbox_label
        if '|' in checkbox_label :
            keys = checkbox_label.split('|')
            key = keys[1].split('-')[0]
        elif ' ' in checkbox_label :
            key = checkbox_label.split(' ')[0]

        elif 'IPM' in checkbox_label: key = 'IPM'
        elif 'DIO' in checkbox_label: key = 'DIO'        
        return key
        

    def add_module(self, checkbox_label):
        """Schedule a module to be run by pyana
        (add it to a dictionary with checkbox label as the key)
        """
        key = self.get_key(checkbox_label)
        print "key to module lookup" , key

        module = self.make_modconf[key](checkbox_label)
        print "module label ", module.label
        self.modules[module.label] = module
        self.update_text()

        return module
        
    
    def remove_module(self, checkbox_label ) :
        """Remove a module from the list of pyana modules
        """
        key = self.get_key(checkbox_label)

        print "Beore remove: \n"
        print self.modules

        if key in self.modules:
            del self.modules[checkbox_label]

        print "\nAfter remove\n"
        print self.modules
        print        
        self.update_text()
        
    def update_text(self):
        """configuration text for pyana.cfg file
        """

        # first the generic stuff
        self.config_text = "[pyana]"
        self.config_text += "\nmodules ="

        config_module_list = []
        
        for label, module in self.modules.iteritems():
            self.config_text += " "
            self.config_text += "XtcExplorer.%s "%(label)
            config_module_list.append( module.config_snippet() )
        self.config_text += "XtcExplorer.pyana_plotter_beta "

        if self.jobconfig.run_n is not None: 
            self.config_text += "\nnum-events = %s"%self.jobconfig.run_n

        if self.jobconfig.skip_n is not None:
            self.config_text += "\nskip-events  = %s"%self.jobconfig.skip_n

        if self.jobconfig.num_cpu is not None: 
            self.config_text += "\nnum-cpu = %s"%self.jobconfig.num_cpu

        self.config_text += "\n"

        # then configuration for each module
        for snippet in config_module_list:
            self.config_text += "\n"
            self.config_text += snippet
        self.config_text += "\n%s"%(self.plotconfig.config_snippet())
        

        # add linebreaks if needed
        self.config_text = self.add_linebreaks(self.config_text, width=70)


    def add_linebreaks(self, text, width=50):
        lines = text.split('\n')
        l = 0
        for line in lines :
            if len(line) > width : # split line
                words = line.split(" ")
                i = 0
                newlines = []
                newline = ""
                while len(newline) <= width and i <len(words) :
                    newline += (words[i]+" ")
                    i += 1
                    if len(newline) > width or i==len(words):
                        newlines.append(newline)
                        newline = "     "
                        
                # now replace the original line with newlines
                l = lines.index(line)
                lines.remove(line)
                if len(newlines)>1 :
                    newlines.reverse()
                for linje in newlines :
                    if linje.strip() != '' :
                        lines.insert(l,linje)
                    
        text = "\n".join(lines)
        return text
