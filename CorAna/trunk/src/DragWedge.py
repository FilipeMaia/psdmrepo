#!/usr/bin/env python
#----------------------------------
import sys
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import math # cos(x), sin(x), radians(x), degrees(), atan2(y,x)

from Drag import *

class DragWedge( Drag, lines.Line2D ) :  #patches.CirclePolygon

    def __init__(self, xy=None, radius=None, theta1=0, theta2=10, width=15, linewidth=2, linestyle='solid', color='b', picker=10, str_of_pars=None) :
        """Draw a wedge centered at x, y center with radius r that sweeps theta1 to theta2 (in degrees) in positive angle direction.
        If width is given, then a partial wedge is drawn from inner radius r - width to outer radius r."""
        Drag.__init__(self, linewidth, color, linestyle, my_type='Wedge')

        if str_of_pars is not None : 
            x,y,r0,w,t1,t2,lw,col,s,t,rem = self.parse_str_of_pars(str_of_pars)
            self.center = (xc,yc) = (x,y)
            self.isSelected    = s
            self.myType        = t
            self.isRemoved     = rem
            self.isInitialized = True

        elif xy is None :      # Default initialization
            self.center = (xc,yc) = (10,10)
            r0, w, t1, t2, col = 10, width, theta1, theta2, color 
            self.isInitialized = False

        elif radius is None : # Semi-default initialization
            self.center = (xc,yc) = xy
            r0, w, t1, t2, col = 10, width, theta1, theta2, color 
            self.isInitialized = False

        else :
            self.center = (xc,yc) = xy
            r0, w, t1, t2, col = radius, width, theta1, theta2, color 
            self.isInitialized = True

        self.set_wedge_pars(self.center, r0, w, t1, t2)
        xarr, yarr = self.get_xy_arrays_for_wedge(xc, yc, r0, w, t1, t2)
        lines.Line2D.__init__(self, xarr, yarr, linewidth=linewidth, color=col, picker=picker)

        self.set_picker(picker)
        self.myPicker = picker
        self.press    = None
        self.theta1_offset = self.get_theta_offset(theta1)
        self.theta2_offset = self.get_theta_offset(theta2)
        self.n_rings = 1
        self.n_sects = 1

        #self.print_test_theta_sheet_number()


    def get_theta_offset(self, theta) :
        """For angle theta in degrees returns its offset w.r.t. the 0-sheet [-180,180)
        """
        return 360 * self.get_theta_sheet_number(theta)


    def get_theta_sheet_number(self, theta) :
        """Returns the sheet number of the angle theta in degree
        [-540,-180) : sheet =-1
        [-180, 180) : sheet = 0
        [ 180, 540) : sheet = 1 ...
        """
        n_sheet = int( int(theta + 180) / 360 )
        #print 'theta, n_sheet=', theta, n_sheet
        return n_sheet
        

    def print_test_theta_sheet_number(self) :
        for t in range(-900,901,30) :
            print 'theta=',t,'   sheet=', self.get_theta_sheet_number(t),'   offset=', self.get_theta_offset(t)


    def bring_theta_in_range(self, theta, range=(-180, 180) ) :
        """Recursive method which brings the input theta in the specified range.
           The default range value corresponds to the range of math.atan2(y,x) : [-pi, pi]
        """
        theta_min, theta_max = range
        theta_corr = theta
        if   theta <  theta_min : theta_corr += 360            
        elif theta >= theta_max : theta_corr -= 360
        else : return theta
        return self.bring_theta_in_range(theta_corr)


    def set_standard_wedge_parameters(self) :
        """Reset the wedge parameters to standard: width > 0, t2>t1 
        """
        if self.width < 0 :            
            self.width  = -self.width 
            self.radius += self.width 
 
        if self.theta2 < self.theta1 : # Swap t1 and t2 to get t1<t2
            t = self.theta2
            self.theta2 = self.theta1
            self.theta1 = t

        # Bring theta1 to 0 sheet
        theta_offset = self.get_theta_offset(self.theta1)
        self.theta1 -= theta_offset 
        self.theta2 -= theta_offset


    def set_wedge_pars(self, xy, radius, width, theta1, theta2) :
        self.center = xy
        self.radius = radius
        self.width  = width
        self.theta1 = theta1
        self.theta2 = theta2
        #self.set_xy_arrays_for_wedge()


    def set_center(self, center) :
        self.center = center
        self.set_xy_arrays_for_wedge()


    def set_radius(self, radius) :
        self.radius = radius
        self.set_xy_arrays_for_wedge()


    def set_width(self, width) :
        self.width = width
        self.set_xy_arrays_for_wedge()


    def set_theta1(self, theta1) :
        self.theta1 = theta1
        self.set_xy_arrays_for_wedge()


    def set_theta2(self, theta2) :
        self.theta2 = theta2
        self.set_xy_arrays_for_wedge()


    def set_xy_arrays_for_wedge(self) :
        xc,yc = self.center
        radius,width,theta1,theta2 = self.radius, self.width, self.theta1, self.theta2
        xarr, yarr = self.get_xy_arrays_for_wedge(xc,yc,radius,width,theta1,theta2)

        self.set_xdata(xarr)
        self.set_ydata(yarr)


    #def get_xy_arrays_for_current_wedge(self) :
    #    return self.get_xdata(), self.get_ydata()


    def get_xy_arrays_for_wedge(self,xc,yc,radius,width,theta1,theta2) :      
        """Return arrays of X and Y polyline coordinates which define the shape of the wedge"""

        t1_rad = math.radians(theta1)
        t2_rad = math.radians(theta2)

        Npoints = int(abs(theta2-theta1)/3)+1

        TarrF = np.linspace(t1_rad, t2_rad, num=Npoints, endpoint=True)
        TarrB = np.array(np.flipud(TarrF))

        rmin = radius - width
        xarrF = xc + radius * np.cos(TarrF)
        yarrF = yc + radius * np.sin(TarrF)
        xarrB = xc + rmin   * np.cos(TarrB)
        yarrB = yc + rmin   * np.sin(TarrB)

        xarr  = np.hstack([xarrF,xarrB,xarrF[0]])
        yarr  = np.hstack([yarrF,yarrB,yarrF[0]])

        return xarr, yarr #.flatten(), yarr.flatten()


    def get_number_of_slices_for_wedge(self) :
        return self.n_rings, self.n_sects


    def get_list_of_pars(self) :
        x,y =      self.center
        r   =      self.radius
        w   =      self.width
        t1  =      self.theta1
        t2  =      self.theta2
        lw  =      self.get_linewidth()
        #col =      self.get_color() 
        col =      self.myCurrentColor
        s   =      self.isSelected
        t   =      self.myType
        rem =      self.isRemoved
        return (x,y,r,w,t1,t2,lw,col,s,t,rem)


    def parse_str_of_pars(self, str_of_pars) :
        pars = str_of_pars.split()
        #print 'pars:', pars
        t   = pars[0]
        x   = float(pars[1])
        y   = float(pars[2])
        r   = float(pars[3])
        w   = float(pars[4])
        t1  = float(pars[5])
        t2  = float(pars[6])
        lw  = int(pars[7])
        col = str(pars[8])
        s   = self.dicBool[pars[9].lower()]
        rem = self.dicBool[pars[10].lower()]
        return (x,y,r,w,t1,t2,lw,col,s,t,rem)


    def get_str_of_pars(self) :
        x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_pars()
        return '%s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %d %s %s %s' % (t,x,y,r,w,t1,t2,lw,col,s,rem)


    def print_pars(self) :
        print 't,x,y,r,w,t1,t2,lw,col,s,rem =', self.get_str_of_pars()


    def my_contains(self, click_r, click_theta, theta1, theta2, dtpick):
        x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_pars()
        drpick = self.myPicker

        #dt = math.degrees( drpick / r ) # picker size (degree) in angular direction
        #click_r = self.distance( clickxy, (x,y) )
        #click_theta = math.degrees( math.atan2(click_y-y, click_x-x) ) # Click angle in the range [-180,180]
        t1_in_range = self.bring_theta_in_range(t1)
        t2_in_range = self.bring_theta_in_range(t2)

        # Check for RADIAL direction
        if click_r < (r + drpick) and click_r > (r - w - drpick) : 
            self.inWideRing = True
        else :
            self.inWideRing = False
            
        if click_r < (r - drpick) and click_r > (r - w + drpick) : 
            self.inNarrowRing = True
        else :
            self.inNarrowRing = False

        # Check for entire ring with 2 cuts
        if abs(t2-t1) > 360 :
            self.isRing = True
            if self.inWideRing :
                if not self.inNarrowRing                     : return True
                elif abs(click_theta - t1_in_range) < dtpick : return True
                elif abs(click_theta - t2_in_range) < dtpick : return True
                else                                         : return False

        if theta2 > theta1 : # normal, positive direction of the arc

            if click_theta > theta1 - dtpick and click_theta < theta2 + dtpick :
                self.inWideSector = True
            else :
                self.inWideSector = False
                
            if click_theta > theta1 + dtpick and click_theta < theta2 - dtpick :
                self.inNarrowSector = True
            else :
                self.inNarrowSector = False
                
        else :   # opposite, negative direction of the arc 

            if click_theta > theta1 - dtpick or click_theta < theta2 + dtpick :
                self.inWideSector = True
            else :
                self.inWideSector = False
                
            if click_theta > theta1 + dtpick or click_theta < theta2 - dtpick :
                self.inNarrowSector = True
            else :
                self.inNarrowSector = False

        if self.inWideRing and self.inWideSector and not (self.inNarrowRing and self.inNarrowSector) :
                return True

        return False


    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.axes: return

        clickxy = click_x, click_y = event.xdata, event.ydata

        if self.isInitialized :

            x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_pars()
            click_r = self.distance( clickxy, (x,y) )
            click_theta = math.degrees( math.atan2(click_y-y, click_x-x) ) # Click angle in the range [-180,180]
            drpick = self.myPicker
            dtpick = math.degrees( drpick / r ) # picker size (degree) in angular direction
            
            theta1 = self.bring_theta_in_range(t1)
            theta2 = self.bring_theta_in_range(t2)

            #if not self.contains(event) : return
            if not self.my_contains(click_r, click_theta, theta1, theta2, dtpick) : return

            #self.print_pars()
            #print 'x,y,r,w,t1,t2,dtpick,click_theta,_r = ', x,y,r,w,t1,t2,dtpick, click_theta, click_r
            #self.set_center(clickxy) 

            r0   = self.radius
            w0   = self.width
            xy0  = self.center

            clkAtRmin   = False
            clkAtRmax   = False
            clkAtTheta1 = False
            clkAtTheta2 = False

            if abs(click_r - r0)      < drpick : clkAtRmax = True                
            if abs(click_r - (r0-w0)) < drpick : clkAtRmin = True
            if click_theta > theta1-dtpick and click_theta < theta1+dtpick : clkAtTheta1 = True
            if click_theta > theta2-dtpick and click_theta < theta2+dtpick : clkAtTheta2 = True

# Numeration in vertindex the vertices and sides of the wedge
#
#        t1                t2
#        
# Rmin   3--------8--------4
#        |                 |
#        |                 |
#        5                 6
#        |                 |
#        |                 |
# Rmax   1--------7--------2

            vertindex = 0  # undefined 
            if   clkAtRmax :
                vertindex = 7                
                if clkAtTheta1 : vertindex = 1
                if clkAtTheta2 : vertindex = 2

            elif clkAtRmin :
                vertindex = 8                
                if clkAtTheta1 : vertindex = 3
                if clkAtTheta2 : vertindex = 4

            elif clkAtTheta1   : vertindex = 5 
            elif clkAtTheta2   : vertindex = 6 

            #print 'vertindex= ',vertindex

            self.press = xy0, clickxy, click_r, click_theta, r0, w0, self.theta1, self.theta2, vertindex

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the object position is not defined yet:

            vertindex = 2
            x0,y0 = xy0 = self.center #= (40,60)
            click_r = self.distance( clickxy, xy0 )
            click_theta = math.degrees( math.atan2(click_y-y0, click_x-x0) ) # Click angle in the range [-180,180]
            self.theta2 = self.theta1 = click_theta
            self.center = xy0
            self.radius = click_r

            self.press  = xy0, clickxy, click_r, click_theta, self.radius, self.width, self.theta1, self.theta2, vertindex

        self.dt_offset = 0
        self.dt_old    = 0
        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        current_x, current_y = current_xy = event.xdata , event.ydata

        xy0, click_xy, click_r, click_t, radius, width, theta1, theta2, vertindex = self.press
        x0, y0 = xy0
        click_x, click_y = click_xy

        current_r = self.distance( current_xy, xy0 )
        current_t = math.degrees( math.atan2(current_y-y0, current_x-x0) ) # Click angle in the range [-180,180]


        if event.button is 3 and self.isInitialized : 
            dx = current_x - click_x
            dy = current_y - click_y
            self.center = x0+dx, y0+dy
            self.set_xy_arrays_for_wedge()
            self.on_motion_graphic_manipulations()
            return

        dr = current_r - click_r
        dt = current_t - click_t

        # Jump ovet the sheet cut
        if dt - self.dt_old >  180 : self.dt_offset -= 360
        if dt - self.dt_old < -180 : self.dt_offset += 360
        self.dt_old = dt        
        dt += self.dt_offset

        if   vertindex == 0 : #side
            print 'WORNING, NON-POSSIBLE vertindex =', vertindex

        elif vertindex == 1 :
            self.theta1 = theta1 + dt
            self.radius = radius + dr
            self.width  = width  + dr

        elif vertindex == 2 :
            self.theta2 = theta2 + dt
            self.radius = radius + dr
            self.width  = width  + dr

        elif vertindex == 3 :
            self.theta1 = theta1 + dt
            self.width  = width  - dr

        elif vertindex == 4 :
            self.theta2 = theta2 + dt
            self.width  = width  - dr

        elif vertindex == 5 :
            self.theta1 = theta1 + dt

        elif vertindex == 6 :
            self.theta2 = theta2 + dt

        elif vertindex == 7 :
            self.radius = radius + dr
            self.width  = width  + dr

        elif vertindex == 8 :
            self.width  = width  - dr

        self.set_xy_arrays_for_wedge()

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
    #    'on release we reset the press data'
        self.set_standard_wedge_parameters()
        self.on_release_graphic_manipulations()
        #if self.press is not None : self.print_pars()
        if self.press is not None : self.maskIsAvailable = False        
        self.press = None


    def get_poly_verts(self):
        """Creates a set of (closed) poly vertices for mask"""
        #xarr, yarr = self.get_xy_arrays_for_current_wedge()
        xarr, yarr = self.get_data()
        return zip(xarr, yarr)


#-----------------------------
#-----------------------------
#-----------------------------
# Test
#-----------------------------
#-----------------------------
#-----------------------------

from DragObjectSet import *

#-----------------------------
#-----------------------------

def generate_list_of_objects(img_extent) :
    """Produce the list of random objects for test purpose.
    1. Generates initial list of random objects
    2. Add them to the figure axes
    3. Connect with signals.
    4. Returns the list of created objects.
    """
    xmin,xmax,ymin,ymax = img_extent 
    print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nobj = 5
    x = xmin+(xmax-xmin)*np.random.rand(nobj)
    y = ymin+(ymax-ymin)*np.random.rand(nobj)
    r = (ymin-ymax)/3*np.random.rand(nobj) + 30

    obj_list = []
    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragWedge((x[indobj], y[indobj]), radius=r[indobj], color='g')
        obj_list.append(obj)

    return obj_list

#-----------------------------

def main_full_test():
    """Full test of the class DragWedge, using the class DragObjectSet
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. use the class DragObjectSet to switch between Add/Move/Remove modes for full test of the object
     """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    t = DragObjectSet(fig, axes, DragWedge, useKeyboard=True)
    t .set_list_of_objs(list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10')
    plt.show()

#-----------------------------

def main_simple_test():
    """Simple test of the class DragWedge.
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. add one more object with initialization at 1st click-and-drag of mouse-left button
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    obj = DragWedge() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10')
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
