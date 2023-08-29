# RESTRICTED MOMENTS SOFTWARE
# --------------------------------------------------------------------
# INPUT:

file_path = 'data/HEA-HPT-025-1R-111.dat'

# Transform to K (wavelength in nm, or None if data in K originally)
wavelength_nm = 0.154056

# Apply BaseLine Subtraction (True or False)
subtract_BaseLine = True

# Calculating m3 moment (True or False)
calc_m3 = False

# fit rho^2-et in m4
fit_rho2 = False

# peak name id (optional, default=None or '')
cut_peak_name_id = ''

# --------------------------------------------------------------------
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit



def get_peak_data(x_values, y_values):
    Y_min = min(y_values)
    Y_max = max(y_values)
    j = 0
    for i, y in enumerate(y_values):
        if y == Y_max:
            j = i
    X_0 = x_values[j]
    return X_0, Y_min, Y_max

def on_xlims_change(event_ax):
    print(" updated xlims: ", event_ax.get_xlim())
    return event_ax.get_xlim()

def on_ylims_change(event_ax):
    print(" updated ylims: ", event_ax.get_ylim())

def select_interval(x_values, y_values, xlabel='x', ylabel='y', title='', set_log_y=False, vertical_lines=False,
                    linestyle='line', zero_vertical=False, enable_zoom=True, show_grid=True, color='black', savefile=None):
    x_lim = None
    fig, ax = plt.subplots()

    if linestyle=='line':
        ax.plot(x_values, y_values, color=color, linewidth=1)
    elif linestyle=='dot':
        ax.plot(x_values, y_values, 'o', color=color, markersize=1)

    if zero_vertical:
        X_0, Y_min, Y_max = get_peak_data(x_values, y_values)
        ax.plot([0, 0], [Y_min, Y_max * 1.1], color="red", linewidth=0.5, linestyle='dashed')

    if vertical_lines:
        my_line_1 = Draggable_Vertical_Line(ax, x_values[0], min(y_values), max(y_values))
        my_line_2 = Draggable_Vertical_Line(ax, x_values[-1], min(y_values), max(y_values))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if set_log_y:
        ax.set_yscale('log')
    if show_grid:
        ax.grid(True)
    if enable_zoom:
        ax.callbacks.connect('xlim_changed', on_xlims_change)
    plt.show()
    if enable_zoom and vertical_lines==False:
        x_lim = ax.get_xlim()
        print(f"final xlim: [{x_lim[0]}, {x_lim[1]}]")
    if vertical_lines:
        x_lim = sorted([my_line_1.get_x(), my_line_2.get_x()])
        print(f"final xlim: [{x_lim[0]}, {x_lim[1]}]")

    if savefile is not None:
        ax.figure.savefig(savefile+'_peak.png')
        print(f'- save figure: {savefile}')

    return x_lim

def new_x_lims(x_values, y_values, xmin, xmax):

    x_values_new, y_values_new = [], []
    for x, y in zip(x_values, y_values):
        if x >= xmin and x <= xmax:
            x_values_new.append(x)
            y_values_new.append(y)
    return x_values_new, y_values_new

def read_data_file(file_path, wavelength_nm=None, cut_peak_name_id=None):
    """read datafile based on file path (getting x and y values), then plotting them, cut them based on Zooming of the plot, then transform x to q """
    print('reading data file...')
    x_values = []
    y_values = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, start=1):
            values = line.strip().split()
            if len(values) == 2:
                try:
                    x_values.append(float(values[0]))
                    y_values.append(float(values[1]))
                except ValueError as e:
                    print(f'Value error:{e}\nin line ({i}), x={values[0]}, y={values[1]}')
                    sys.exit()
        print(f'number of data points: {i}')

    # Adatok ábrázolása, Zoom-olással kiválasztani területet, ha az szükséges (pl teljes pattern van)
    x_lim = select_interval(x_values, y_values, xlabel='x (2theta or K)', ylabel='Intensity', title='Select Interval by Zooming then Exit',
                                      set_log_y=False, linestyle='line', color='green', show_grid=False)

    # zoomolt adatok újraszámolása
    x_values, y_values = new_x_lims(x_values, y_values, x_lim[0], x_lim[1])

    # kivágott csúcs kimentése fájlba
    if cut_peak_name_id is not None and len(cut_peak_name_id) > 0:
        outname = file_name
        save_file(outname, '_' + cut_peak_name_id + '.dat', x_values, y_values)

    # Átváltás K-ba
    if wavelength_nm is not None:
        x_values = [2.0 * math.sin((math.radians(x)) / 2.0) / wavelength_nm for x in x_values]

    return x_values, y_values

class Draggable_Line(object):
    """
    this line will be draggable by the endpoints of the line on a plot
    you can click with right mouse button and drag the endpoints and the line will refresh to match the endpoints
    the xs and ys coordinates can be read to get the endpoint coordinates,
        for example:
        my_line = Draggable_Line(line, epsilon=20)
        x_BG = my_line.xs
        y_BG = my_line.ys
    When you click on the plot, d distances from endpoints calculated, epsilon is a limit for d distance, line will move within epsilon click only
    """
    #epsilon = 30

    def __init__(self, line, epsilon=0.1):
        canvas = line.figure.canvas
        self.canvas = canvas
        self.line = line
        self.axes = line.axes
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.epsilon = epsilon

        self.ind = None

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def get_ind(self, event):
        x = np.array(self.line.get_xdata())
        y = np.array(self.line.get_ydata())
        #d = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)
        # d calculated only for x-distances. With y it can be strange when x and y are not in the same range (i.e. xlim=0.1, ylim=10000)
        d = np.sqrt((x - event.xdata) ** 2)
        #print(f'distances of line edges from click point (only in xrange) ={d}, epsilon={self.epsilon}')
        if min(d) > self.epsilon:
            return None
        if d[0] < d[1]:
            return 0
        else:
            return 1

    def button_press_callback(self, event):
        # left mouse button: if event.button != 1
        # right mouse button: if event.button != 3
        if event.button != 3:
            return
        self.ind = self.get_ind(event)
        #print(self.ind)

        self.line.set_animated(True)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        if event.button != 3:
            return
        self.ind = None
        self.line.set_animated(False)
        self.background = None
        self.line.figure.canvas.draw()
        print(f'x={sorted(list(map(lambda x: round(x,3), self.xs)))}, y={sorted(list(map(lambda x: round(x,3), self.ys)))}')

    def motion_notify_callback(self, event):
        if event.inaxes != self.line.axes:
            return
        if event.button != 3:
            return
        if self.ind is None:
            return
        self.xs[self.ind] = event.xdata
        self.ys[self.ind] = event.ydata
        self.line.set_data(self.xs, self.ys)

        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)

class Draggable_Vertical_Line(object):

    def __init__(self, ax, x_pos, y_min, y_max, linestyle='dashed'): #linestyle='solid'
        self.o = "v"
        self.ax = ax
        self.canv = ax.get_figure().canvas
        self.x = [x_pos, x_pos]
        self.y = [y_min, y_max]
        self.line = Line2D(self.x, self.y, linestyle=linestyle, picker=5, color="red", linewidth=1)
        self.ax.add_line(self.line)
        self.canv.draw_idle()

        #self.canv.mpl_connect('button_press_event', self.button_press_callback)
        self.canv.mpl_connect('pick_event', self.clickonline)


    """def button_press_callback(self, event):
        if event.button == 1:
            print("left mouse button")
            self.button = 'l'
        if event.button == 3:
            print("right mouse button")
            self.button = 'r'"""

    def clickonline(self, event):
        if event.artist == self.line:
            #print("line selected ", event.artist)
            self.follower = self.canv.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.canv.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        if self.o == "h":
            self.line.set_ydata([event.ydata, event.ydata])
        else:
            self.line.set_xdata([event.xdata, event.xdata])
        self.canv.draw_idle()

    def releaseonclick(self, event):
        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
        else:
            self.XorY = self.line.get_xdata()[0]

        print(f' {self.XorY}')

        self.canv.mpl_disconnect(self.releaser)
        self.canv.mpl_disconnect(self.follower)

    '''def set_x(self, set_x1, set_x2):
        self.line.set_xdata([set_x1,set_x2])'''

    def get_x(self):
        result = round(self.line.get_xdata()[0], 4)
        #print(f' -get_x():{result}')
        return result
        #return round(self.line.get_xdata()[0], 4)

class Fit_Curve(object):

    def __init__(self,  ax, x_lim, x_data, y_data, function_type='m4',
                 color_moment='green', color_fit_interval= 'red', color_fitline='black', fitres=100,
                 xlabel='x', ylabel='y', title='Fit Curve\ndrag vertical line with left mouse button to select fit interval\nclick right button to fit',
                 markersize=1, filename='figure'):
        self.color_moment = color_moment
        self.color_fit_interval = color_fit_interval
        self.color_fitline = color_fitline
        self.fitres = fitres

        self.function_type = function_type
        self.ax = ax
        self.canv = ax.get_figure().canvas
        self.x_data = x_data
        self.y_data = y_data
        self.x_lim = x_lim
        self.xlabel = xlabel
        self.m3_xlim = 0.1*self.x_lim[0]
        self.ylabel = ylabel
        self.title = title
        self.markersize = markersize
        self.filename = filename + '.png'

        self.ax.plot(self.x_data, self.y_data, 'o', color=self.color_moment, markersize=1)
        self.limit_line_1 = Draggable_Vertical_Line(self.ax, self.x_lim[0], min(self.y_data), max(self.y_data))
        self.limit_line_2 = Draggable_Vertical_Line(self.ax, self.x_lim[1], min(self.y_data), max(self.y_data))
        self.ax.set_xlim(left=0)
        if self.function_type != 'm3':
            self.ax.set_ylim(bottom=0)
        else:
            self.ax.set_xlim(left=self.m3_xlim)
            self.ax.set_ylim(bottom=min(self.y_data))
            self.ax.set_ylim(top=max(self.y_data))
            self.ax.set_xscale('log')
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

        self.x_values_new, self.y_values_new = [], []
        self.canv.draw_idle()
        self.canv.mpl_connect('button_press_event', self.fit_right_mouse_click)

        self.rho_d = None, None
        self.s2 = None
        self.fitted_y = None
        self.my_function = None
        self.functions = {'m4': self.m4_function, 'square': self.square_function, 'linear': self.linear_function, 'exp': self.exponential_function,
                          'm2': self.m2_function, 'm4_rho2': self.m4_function_rho2, 'm3': self.m3_function}

    # fit functions, extend: define a function then put it to functions dictionary above
    def linear_function(self, x, a, b):
        return a*x + b

    def square_function(self, x, a, b):
        return a*x**2 + b

    def exponential_function(self, x, a, b, c):
        return a*np.exp(-b*x) + c

    def m4_function(self, x, a, b):
        return a * x + b

    def m4_function_rho2(self, x, a, b, c, d):
        result = 0
        try:
            np.seterr(all='ignore')
            result = a*x + b + c * np.log(x/d)**2 / x**2
        except:
            pass
        return result

    def m2_function(self, x,a,b,c):
        result = 0
        try:
            np.seterr(all='ignore')
            result = (a * x + b * np.log(x/c))
        except:
            pass
        return result

    def m3_function(self,x, a, b):
        result = 0
        try:
            np.seterr(all='ignore')
            result = (a * np.log(x / b))
        except:
            pass
        return result


    def get_R_value(self, ydata, yfit):
        r_squared = None
        try:
            ydata = np.array(ydata)
            yfit = np.array(yfit)
            residuals = ydata - yfit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = round(1-(ss_res / ss_tot), 4)
        except:
            pass
        return r_squared

    def get_m4_parameters(self,a,b):
        rho_14, d_nm = None, None
        try:
            rho_14 = round(b*(4*math.pi**2)*0.01,2)
            d_nm = round(2/(a*3*math.pi**2)*1e6,2)
        except:
            pass
        return rho_14, d_nm

    def get_m4_rho2_parameters(self,a,b,c,d):
        rho_14, d_nm, sqrt_rho_2, q1 = None, None, None, None
        try:
            rho_14 = round(b*(4*math.pi**2)*0.01,2)
            d_nm = round(2/(a*3*math.pi**2)*1e6,2)
            sqrt_rho_2 = round(math.sqrt(c*4*math.pi*math.pi*math.pi*math.pi/3) ,5)
            q1 = round(d,5)
        except:
            pass
        return rho_14, d_nm, sqrt_rho_2, q1

    def get_m2_parameters(selfs, a, b, c):
        rho_14, d_nm, q0 = None, None, 0.01
        try:
            rho_14 = round(b*(2*math.pi**2)*0.01,2)
            d_nm = round(2/(a*math.pi**2)*1e6,2)
            q0 = round(c,5)
        except:
            pass
        return rho_14, d_nm, q0

    def fit_right_mouse_click(self, event):
        if event.button == 3:
            print("click right mouse button, fitting curve")

            # get fitting interval
            self.x_lim = sorted([self.limit_line_1.get_x(), self.limit_line_2.get_x()])
            print(f' - xlim after right click: {self.x_lim}')

            # clear ax
            plt.cla()

            # new x, y values within x_lim interval
            self.x_values_new, self.y_values_new = [], []
            for x, y in zip(self.x_data, self.y_data):
                if x >= self.x_lim[0] and x <= self.x_lim[1]:
                    self.x_values_new.append(x)
                    self.y_values_new.append(y)

            # initial guess for fit parameters, it is crucial for moments
            p0 = None
            if self.function_type == 'm2':
                p0 = [1e3,1e3,0.01]
            if self.function_type == 'm4':
                p0 = [1000, 100]
            if self.function_type == 'm4_rho2':
                p0 = [1e3, 1e2, 1, 1e-2]
            if self.function_type == 'm3':
                p0 = [1e3,0.01]

            # fit within x_lim interval
            success_fit = False
            fitting_parameters = None
            plot_label = ''
            try:
                fitting_parameters, covariance = curve_fit(self.functions[self.function_type], self.x_values_new, self.y_values_new, method='lm', p0=p0)
                success_fit = True
                print(f' - fit parameters: {fitting_parameters}')

                if self.function_type == 'm4' and success_fit:
                    print(f' - [rho_14, d_nm] = {self.get_m4_parameters(*fitting_parameters)}')
                    rho_14, d_nm = self.get_m4_parameters(*fitting_parameters)
                    #rho_dim = '10' + '14'.upper() + ' m' + '-2'.upper()
                    plot_label = f' <ρ*> = {rho_14} [10¹⁴ m⁻²]\n d = {d_nm} nm'
                    self.rho_d = rho_14, d_nm
                if self.function_type == 'm2' and success_fit:
                    print(f' - [rho_14, d_nm, q0] = {self.get_m2_parameters(*fitting_parameters)}')
                    rho_14, d_nm, q0 = self.get_m2_parameters(*fitting_parameters)
                    plot_label = f' <ρ*> = {rho_14} [10¹⁴ m⁻²]\n d = {d_nm} nm\n q0 = {q0}'
                    self.rho_d = rho_14, d_nm
                if self.function_type == 'm4_rho2' and success_fit:
                    print(f' - [rho_14, d_nm, rho_2, q1] = {self.get_m4_rho2_parameters(*fitting_parameters)}')
                    rho_14,d_nm, sqrt_rho_2, q1 = self.get_m4_rho2_parameters(*fitting_parameters)
                    plot_label = f' <ρ*> = {rho_14} [10¹⁴ m⁻²]\n d = {d_nm} nm\n <ρ*²>½ = {sqrt_rho_2}\n q1 = {q1}'
                if self.function_type == 'm3' and success_fit:
                    s2, q3 = fitting_parameters
                    self.s2 = round(-s2/6, 6)
                    print(f' - [s_2, q3] = [{self.s2}, {q3}]')
                    plot_label = f' <s²> = {self.s2} [10²¹ m⁻³]\nq3 = {q3}'
            except:
                print('cannot fit')
                success_fit = False


            # create x points for fit curve
            x_min = 0
            x_max = self.x_data[-1]
            x_fit = np.linspace(x_min, x_max, self.fitres)

            # redraw ax: datapoints, datapoints with x_lim intervals, draggable vertical lines for indicating x_lim,
                # and fittet curve if fit succes
            self.ax.plot(self.x_data, self.y_data, 'o', color=self.color_moment, markersize=1)
            self.ax.plot(self.x_values_new, self.y_values_new, 'o', color=self.color_fit_interval, markersize=1)
            self.limit_line_1 = Draggable_Vertical_Line(self.ax, self.x_lim[0], min(self.y_data), max(self.y_data))
            self.limit_line_2 = Draggable_Vertical_Line(self.ax, self.x_lim[1], min(self.y_data), max(self.y_data))
            if success_fit:
                self.ax.plot(x_fit, self.functions[self.function_type](x_fit, *fitting_parameters), '-', color=self.color_fitline)
                self.ax.annotate(plot_label, xy=(0,0.9*max(self.y_data)))
            self.ax.set_xlim(left=0)
            if self.function_type != 'm3':
                self.ax.set_ylim(bottom=0)
            else:
                self.m3_xlim = 0.1*self.x_lim[0]
                self.ax.set_xlim(left=self.m3_xlim)
                self.ax.annotate(plot_label, xy=(self.x_lim[0], 0.9 * max(self.y_data)))
                self.ax.set_ylim(bottom=min(self.y_data))
                self.ax.set_ylim(top=max(self.y_data))
                self.ax.set_xscale('log')
            self.ax.set_ylim(top=1.05*max(self.y_data))
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            self.ax.set_title(self.title)
            self.canv.draw()

            # calculate R value
            if success_fit:
                y_fit = list(map(lambda x: self.functions[self.function_type](x, *fitting_parameters), self.x_values_new))
                self.fitted_y = list(map(lambda x: self.functions[self.function_type](x, *fitting_parameters), self.x_data))
                R_value = self.get_R_value(self.y_values_new, y_fit)
                #print(self.y_values_new)
                #print(y_fit)
                print(f' - R value = {R_value}')

            # save file
            self.ax.figure.savefig(self.filename)
            print(f' - save figure: {self.filename}')


def select_BG(x_values, y_values, x_BG, y_BG):
    """
    plotting x and y values and a draggable baseline, you can click to the endpoint of the line and drag
    returns with endpoint coordinates of the baseline
    """
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'o', color="green", markersize=1)
    line = Line2D(x_BG, y_BG, marker='o', color="red", markersize=5, fillstyle='none')  # markerfacecolor='blue',
    ax.add_line(line)
    epsilon = 0.1 * np.sqrt((x_values[0] - x_values[-1]) ** 2)
    my_line = Draggable_Line(line, epsilon=epsilon)
    ax.set_xlabel('q [1/nm]')
    ax.set_ylabel('Intensity')
    ax.set_title('Select BaseLine\nclick on the endpoint with right mouse button\nhold and drag to move the line')
    plt.show()
    x_BG = list(map(lambda x: round(x, 4), my_line.xs))
    y_BG = list(map(lambda x: round(x, 4), my_line.ys))
    if x_BG[1] < x_BG[0]:
        x_BG = [x_BG[1], x_BG[0]]
        y_BG = [y_BG[1], y_BG[0]]

    print(f'BackGround coordinates: x_BG={x_BG}, y_BG={y_BG}')

    return x_BG, y_BG

def calc_linear_parms(x_BG, y_BG):
    x0, x1 = x_BG[0], x_BG[1]
    y0, y1 = y_BG[0], y_BG[1]
    slope = (y1 - y0) / (x1 - x0)
    interception = y1 - slope * x1
    return slope, interception

def linear_function(x_values, slope, interception):
    return [slope*x+interception for x in x_values]

def subtract_BG(x_values, y_values, x_BG, y_BG):
    """subtract linear baseline from y values"""
    x_BG, y_BG = select_BG(x_values, y_values, x_BG, y_BG)

    slope, interception = calc_linear_parms(x_BG, y_BG)
    print(f'slope={slope}, interception={interception}')

    BG_intensity = linear_function(x_values, slope, interception)

    y_values_noBG = []
    for y, y_bg in zip(y_values, BG_intensity):
        y_values_noBG.append(y - y_bg)

    return y_values_noBG, x_BG, y_BG

def numerical_integral(x_values, y_values, a=None, b=None):
    x, y = [], []
    if a is None:
        a = x_values[0]
    if b is None:
        b = x_values[-1]
    for _x, _y in zip(x_values,y_values):
        if a <= _x and _x <= b:
            x.append(_x)
            y.append(_y)
    integral = 0
    # trapezoid
    for i in range(1, len(x)):
        delta_x = x[i] - x[i-1]
        integral += (y[i] + y[i-1]) * 0.5 * delta_x

    return integral

def calc_center_of_gravity(x_values, y_values, a, b):
    x, y, xy = [], [], []
    for _x, _y in zip(x_values,y_values):
        if a <= _x and _x <= b:
            x.append(_x)
            y.append(_y)
            xy.append(_x*_y)
    #print(f'CoG[simpson]={simps(xy,x)/simps(y,x)}, CoG[trapezoid]= {numerical_integral(x,xy)/numerical_integral(x,y)}')
    #return simps(xy,x)/simps(y,x)
    return numerical_integral(x,xy)/numerical_integral(x,y)

def correct_center_of_gravity(x_values, y_values):
    """fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'o', color="green", markersize=1)
    my_line = draggable_vertical_horizontal(ax, "vertical", 0.0, min(y_values), max(y_values))
    ax.add_line(my_line)"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, color="green", markersize=1,  linewidth=1)
    #my_line = Draggable_Vertical_Line(ax, "v", 0.0, min(y_values), max(y_values))
    my_line = Draggable_Vertical_Line(ax,0.0, min(y_values),max(y_values),linestyle='solid')
    #ax.grid(True)

    ax.set_xlabel('q [1/nm]')
    ax.set_ylabel('Intensity')
    ax.set_title('Correct Peak Center\nclick on the line, drag then click again to fix the position')
    plt.show()
    print(f'Peak Center: {my_line.get_x()}')
    return my_line.get_x()

def momentum(x_values, y_values, k, qmax=None):

    # k-adik momentum integrandusa
    integrand = []
    for x, y in zip(x_values, y_values):
        integrand.append(y*(x**k))

    # dq meghatározása
    dq = (x_values[-1] - x_values[0]) / len(x_values)
    #print(f'dq={dq}')

    if qmax is None:
        qmax = x_values[-1]

    if qmax/dq > 1000:
        print(f'\n too many datapoints for moments: {qmax/dq}, reduced to 1000')
        dq = qmax/1000

    # integrálás -q-tól q-ig
    q,i = 0,0
    moment, q_values = [], []
    while q <= qmax:
        moment.append(numerical_integral(x_values, integrand, -q, q))
        q_values.append(q)
        q += dq
        i+=1
    print(f' moments datapoints: {i} ')

    # korrekciók
    moment_final = []
    for q, m in zip(q_values, moment):
        if k == 4:
            if q != 0:
                moment_final.append(1e6*m/q/q)
            else:
                moment_final.append(0)
        else:
            moment_final.append(1e6*m)

    return q_values, moment_final

def fit_momentum(k, q_val, moments, xlim, color='green', filename = '', file_name=''):
    rho_star, d_nm = None, None

    fig, ax = plt.subplots()

    x_left = xlim[0]
    x_right = xlim[1]
    if x_left is None:
        print('xleft is None, apply 1/3 qmax')
        x_left = (q_val[0] + 0.33*math.fabs(q_val[-1]-q_val[0]))
    if x_right is None:
        print('xright is None, apply 2/3 qmax')
        x_right = (q_val[-1] - 0.33*math.fabs(q_val[-1]-q_val[0]))
    x_lim = [x_left, x_right]

    xlabel, ylabel = 'q [1/nm]', None
    title = f'm{k} Fit Curve: {filename}\ndrag vertical line with left mouse button to select fit interval\nclick right button to fit'
    if k==2:
        ylabel = 'v₂ [10¹² m⁻²]'
    elif k==4:
        ylabel = 'v₄/q² [10¹² m⁻²]'
    elif k == 3:
        ylabel = 'v₃ [10²¹ m⁻³]'

    moment_type = None
    if k==2:
        moment_type = 'm2'
    if k==3:
        moment_type = 'm3'
    if k==4:
        moment_type = 'm4'
    if k==5:
        moment_type = 'm4_rho2'

    fit_line = Fit_Curve(ax, x_lim, q_val, moments, function_type=moment_type,
                         color_moment=color, color_fit_interval= 'red', color_fitline='black',
                         fitres=300, xlabel=xlabel, ylabel=ylabel, title=title, markersize=1, filename=file_name+'_'+moment_type)
    plt.show()

    rho_star, d_nm = fit_line.rho_d[0], fit_line.rho_d[1]
    fitted_y = fit_line.fitted_y

    if rho_star is not None and d_nm is not None:
        x_lim = [fit_line.x_lim[0], fit_line.x_lim[1]]
    print(f'fit_line x_lim:{x_lim}')

    if k == 3:
        s_2 = fit_line.s2
        return s_2, fitted_y

    return rho_star, d_nm, fitted_y, x_lim

def q_01(x_values, y_values, CoG):
    q_01_left, q_01_right = None, None
    goal_y = 0.1*max(y_values)
    epsilon = max(y_values)
    for x, y in zip(x_values, y_values):
        if x < CoG:
            if math.fabs(y-goal_y) < epsilon:
                epsilon = math.fabs(y-goal_y)
                q_01_left = x
    epsilon = max(y_values)
    for x, y in zip(x_values, y_values):
        if x > CoG:
            if math.fabs(y - goal_y) < epsilon:
                epsilon = math.fabs(y - goal_y)
                q_01_right = x

    if q_01_left is not None and q_01_right is not None:
        print(f' q_01_left={q_01_left}, q_01_right={q_01_right}')
        return round(0.5*(math.fabs(q_01_left) + math.fabs(q_01_right) ), 3)
    else:
        return None

def save_file(file_name, extention, x_values, y_values, moment = None):
    text_data = ""
    if moment == None:
        for x, y in zip(x_values, y_values):
            text_data += f'{x}\t{y}\n'
    else:
        for x, y, m in zip(x_values, y_values,moment):
            text_data += f'{x}\t{y}\t{m}\n'

    with open(file_name + extention, 'w', encoding='utf-8') as file:
        file.write(text_data)

def save_results(file_name, extention, text):
    text_out = file_name + extention + '\n' + text
    with open(file_name + extention, 'w', encoding='utf-8') as file:
        file.write(text_out)

if __name__ == '__main__':
    print('MOMENTS\n')
    file_name = file_path.split('.')[0]
    print(f'filename:"{file_name}"')
    outname = file_name
    filename = file_name.rpartition("/")[2]

    # Adatok beolvasása fájlból, intervallum kiválasztása (lehet teljes pattern is), átváltás K-ba
    x_values, y_values = read_data_file(file_path, wavelength_nm, cut_peak_name_id)

    # Csúcs eltolása, a maximum intenzitás legyen 0-ban
    X_0, Y_min, Y_max = get_peak_data(x_values, y_values)
    print(f'Before Peak Shift: X_0={X_0}, Y_min={Y_min}, Y_max={Y_max}')
    x_values = [x-X_0 for x in x_values]

    # Most még egyszer lehetőség van intervallum kiválasztására, nem muszáj, de ha teljes pattern volt korábban, akkor így finomítható még
    x_lim = select_interval(x_values, y_values, xlabel='q [1/nm]', ylabel='Intensity',
                            title='Select Interval by Dragging Lines\nclick on the line, drag then click again to fix the position',
                            set_log_y=False, linestyle='line', zero_vertical=True, show_grid=False,color='green', vertical_lines=True)
    # zoomolt adatok újraszámolása: új x tartomány (és y is), új Y min és max, és új X_0 (ami persze 0)
    x_values, y_values = new_x_lims(x_values, y_values, x_lim[0], x_lim[1])
    X_0, Y_min, Y_max = get_peak_data(x_values, y_values)
    print(f'After Select Intarval: X_0={X_0}, Y_min={Y_min}, Y_max={Y_max}')

    # kezdeti háttérpontok az adatok szélei
    x_BG = [x_values[0],x_values[-1]]
    y_BG = [y_values[0],y_values[-1]]

    x_values_CoG, y_normalized = None, None
    success = False
    #x_fit_interval_left, x_fit_interval_right = 0.25*x_values[-1], 0.5*x_values[-1]
    x_fit_interval = [None, None]
    while(True):
    # addig megy a ciklus, amíg elégedettek nem vagyunk

        # háttér levonása, az itt kiválaszott háttérpontoktól folyatjuk a következő while-ciklusban majd, tehát az nem vész el
        if subtract_BaseLine:
            y_values_noBG, x_BG, y_BG = subtract_BG(x_values, y_values, x_BG, y_BG)
        else:
            y_values_noBG, x_BG, y_BG = subtract_BG(x_values, y_values, x_BG, [0,0])

        # Center of Gravity (CoG)
            # CoG kiszámítása
        CoG = calc_center_of_gravity(x_values, y_values_noBG, x_values[0], x_values[-1])
        x_values_CoG = [x - CoG for x in x_values]
        print(f'Center of Gravity: {CoG}')
            # CoG korrigálása, ha az szükséges
        CoG = correct_center_of_gravity(x_values_CoG, y_values_noBG)
        print(f'Correction for Center of Gravity: {CoG}')
        x_values_CoG = [x - CoG for x in x_values_CoG]

        # normalizált csúcs
        Area = numerical_integral(x_values_CoG, y_values_noBG)
        print(f'peak area = {Area}')
        y_normalized = [y / Area for y in y_values_noBG]

        # Végső  csúcs ábrázolása: háttér nélküli, normalizált és centralizált
        select_interval(x_values_CoG, y_normalized, xlabel='q [1/nm]', ylabel='Intensity', title='Subtracted, Centralized, Normalized',
                                  set_log_y=False, linestyle='line', zero_vertical=False, enable_zoom=False, color='green', savefile=file_name)

        # momentumok számolása
        print(f'\nCalc. moments...', end='')
        qmax = min((math.fabs(x_values_CoG[0]), math.fabs(x_values_CoG[-1])))
        q_val_4, m4 = momentum(x_values_CoG, y_normalized, 4, qmax = qmax)
        q_val_2, m2 = momentum(x_values_CoG, y_normalized, 2, qmax=qmax)
        print(f' done. qmax={qmax}')

        # q_01 meghatározása (Az a q, ahol a relatív intenzitás 0.1, vagyis  0.1*I_center(q_01)
        if x_fit_interval[0] is None:
            print('\nxleft is None, calculate q_01')
            x_fit_interval[0] = q_01(x_values_CoG, y_normalized, CoG)
            print(f'q_01={x_fit_interval[0]}')

        print(f'\nFIT M4 MOMENT')
        if fit_rho2:
            rho_4, d_4, fitted_y_4, x_fit_interval = fit_momentum(5, q_val_4, m4, x_fit_interval, filename=filename, file_name=file_name)
        else:
            rho_4, d_4, fitted_y_4, x_fit_interval = fit_momentum(4, q_val_4, m4, x_fit_interval, filename=filename, file_name=file_name)
        print(f'm4 fitted results: rho_4={rho_4}, d_4={d_4}')
        print(f'\nFIT M2 MOMENT')
        rho_2, d_2, fitted_y_2, x_fit_interval = fit_momentum(2, q_val_2, m2, x_fit_interval, filename=filename,file_name=file_name)
        result_text = f'\nm4 fitted results: rho_4={rho_4}, d_4={d_4}\nm2 fitted results: rho_2={rho_2}, d_2={d_2}'
        print(result_text)

        q_val_3, m3, fitted_y_3 = [],[],[]
        if calc_m3:
            print(f'\nFIT M3 MOMENT')
            q_val_3, m3 = momentum(x_values_CoG, y_normalized, 3, qmax=qmax)
            s_2, fitted_y_3 = fit_momentum(3, q_val_3, m3, x_fit_interval, filename=filename, file_name=file_name)
            result_text = f'\nm4 fitted results: rho_4={rho_4}, d_4={d_4}\nm2 fitted results: rho_2={rho_2}, d_2={d_2}\nm3 fitted results: s_2={s_2}'
            print(result_text)

        # ciklus megszakítása ha elégedettek vagyunk az eredménnyel
        while(True):
            success = input(f'\nSuccess? (y/n):')
            if success.lower() in ('y', 'yes'):

                # eredmények elmentése fájlba
                print('saving files... ',end='')
                if cut_peak_name_id is not None and len(cut_peak_name_id) > 0:
                    outname += ('_'+ cut_peak_name_id)
                save_results(outname, '_results.dat', result_text)
                save_file(outname, '_peak.dat', x_values_CoG, y_normalized)
                save_file(outname, '_m2.dat', q_val_2, m2, fitted_y_2)
                save_file(outname, '_m4.dat', q_val_4, m4, fitted_y_4)
                if calc_m3:
                    save_file(outname, '_m3.dat', q_val_3, m3, fitted_y_3)
                print('done')

                success = True
                break
            if success.lower() in ('n', 'no'):
                break
        if success == True:
            break


