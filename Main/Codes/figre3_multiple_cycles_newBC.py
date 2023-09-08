######################################################################
#Figure 3 - Multiple ice layers due to cyclic thermal forcing
#Mohammad Afzal Shadab
#Date modified: 05/03/2022
######################################################################


######################################################################
#import libraries
######################################################################
import sys
sys.path.insert(1, '../../solver')
from scipy.optimize import curve_fit

# import personal libraries and class
from classes import *    

from two_components_aux import *
from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells
from comp_face_coords_fun import comp_face_coords
from find_plate_dofs import find_plate_dofs,multiple_plate_dofs
from colliander_data_analysis import spun_up_profile_May2016_Colliander, Qnet_May2Sept2016_Samira
from comp_mean_matrix import comp_mean_matrix
from eval_phase_behavior import eval_phase_behaviorCwH

fit_temp,fit_porosity = spun_up_profile_May2016_Colliander()
day,Qnet,Qnet_func             = Qnet_May2Sept2016_Samira()

#for 2D
from build_gridfun2D import build_grid
from spin_up_firn import spin_up_firn

#Non-dimensional permeability: Harmonic mean
def f_Cm(phi,m):
    fC = np.zeros_like(phi)        
    fC = phi**m / phi_L**m           #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

######################################################################
#parameters 
######################################################################
##simulation

simulation_name = f'paper_original_net_zero'   #left: 25Wto-40W
diffusion = 'yes'
CFL    = 0.1     #CFL number
tilt_angle = 0   #angle of the slope
ncycles = 4           
npp    = 4   #number of positive days [days]          
break_type = 'equal' #equal or unequal
break_length = 5    #number of days for a break         
lag    = 2*npp#npp #number of days to wait after preipitation has happened in the very end[days]
pulse_type = 'sine'  #'sine' wave or 'square' wave  : sine needs equal break

##hydrology
m      = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n      = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr   = 0.0   #Residual water saturation
s_gr   = 0.0   #Residual gas saturation
k_w0   = 1.0  #relative permeability threshold: wetting phase
rho_w  = 1000.0  #density of non-wetting phase
mu_w   = 1e-3 #dynamic viscosity: wetting phase    
grav   = 9.81    #acceleration due to gravity
k0     = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017

[C_L,phi_L] = [0.03,fit_porosity(0.0)] #water volume fraction, porosity at the surface

##thermodynamics
On_flux  =  45  #Flux on hot days [W/m^2]
Off_flux = -45  #Flux on cold days [W/m^2]

T_firn = -10  # temperature of firn [C]
simulation_name = simulation_name+f'T{T_firn}C'+f'LWC{C_L}'+f'npp{npp}'

rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_nw = 1.225  # density of gas [kg/m^3]
phi_nw_init= 0.5# volumetric ratio of gas, decreasing exponentially
cp_nw  = 1003.5 # specific heat of gas at constant pressure [J/(kg K)]
rho_i = 917     # average density of ice cap [kg/m^3]
cp_i  = 2106.1  # specific heat of ice at constant pressure [J/(kg K)]
k_i   = 2.25    # coefficient of thermal conductivity of ice [W / (m K)]
kk    = k_i/(rho_i*cp_i) # thermal diffusivity of ice
Tm    = 273.16  # melting point temperature of ice [K]
L_fusion= 333.55e3# latent heat of fusion of water [J / kg]

#domain details
z0 = 3.75 #characteristic height (m)
fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)
sat_threshold = 1-1e-3 #threshold for saturated region formation

#injection
Param.xleft_inj= 0e3;  Param.xright_inj= 1000e3

#temporal
if break_type == 'equal':
    tf     = npp*(2*ncycles)*day2s + lag*day2s
else:
    tf     = ncycles*(npp + break_length)*day2s + lag*day2s    
    
tmax = tf#0.07#0.0621#2 #5.7#6.98  #time scaling with respect to fc
t_interest = np.linspace(0,tmax,int(tf/day2s*24)+1)   #swr,sgr=0

#tmax = tmax / phi_L**m   #time scaling with respect to K_0 where K_0 = f_c/phi**m
Nt   = 1000
dt = tmax / (Nt)

#######
#analytical solution 
phi_i_init = (1-phi_nw_init)
kbar = 2.22362*(rho_i/rho_w*(phi_i_init))**1.885; npp=4
rhocp_bar = rho_i * cp_i * phi_i_init; alpha_bar = kbar / rhocp_bar; omega = 2*np.pi/(2*npp*day2s)

T_analytical_sine = lambda z,t: T_firn + On_flux / kbar * np.sqrt(alpha_bar/(2 * omega)) * np.exp(-z * np.sqrt(omega/(2*alpha_bar))) * ( np.sin(omega*t - z*np.sqrt(omega/(2*alpha_bar))) - np.cos(omega*t - z*np.sqrt(omega/(2*alpha_bar))) )
T_max    = T_firn + On_flux / kbar * np.sqrt(alpha_bar/(omega))


#Non-dimensional permeability: Harmonic mean
def f_Cm(phi,m):
    fC = np.zeros_like(phi)
    fC = fc* phi**m / phi_L**m           #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

######################################################################
#Defining grid and operators
######################################################################

Grid.xmin =  0*z0; Grid.xmax =1000e3; Grid.Nx = 2;   #Horizontal direction
Grid.ymin =  0*z0; Grid.ymax =5;  Grid.Ny = 200;     #Vertically downward direction
Grid = build_grid(Grid)  #building grid
[D,G,I] = build_ops(Grid) #building divergence, gradient and identity operators
D  = -np.transpose(G)
Avg     = comp_mean_matrix(Grid)   #building mean operator

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

s_wfunc    = lambda phi_w,phi_nw: phi_w  / (phi_w + phi_nw)
s_nwfunc   = lambda phi_w,phi_nw: phi_nw / (phi_w + phi_nw)
T_annual_func_sigmoid = lambda Tbot, Ttop, Yc_col, Y0: Tbot + (Ttop - Tbot)/Y0*(Yc_col) #* 1/(1+np.exp(-(Grid.ymax-Yc_col)/Y0))

######################################################################
##Initial conditions
######################################################################
phi_nw  = phi_nw_init*np.ones_like(Yc_col) #volume fraction of gas 
phi_w   = np.zeros_like(phi_nw) ##volume fraction of water : No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i  #Composition
H       = enthalpyfromT(T_firn*np.ones_like(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion) #Constant Enthalpy


phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()#s_wp *np.ones((grid.N,1))
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation

simulation_name = simulation_name+f'phi{phi_nw[0]}'+f'T{T_firn}'+f'npp{npp}'+f'cycles{ncycles}'+f'lag{lag}days'+f'break{break_type}'+f'length{break_length}'

######################################################################
#initializing arrays
######################################################################
s_w_sol = np.copy(s_w) #for water saturation
H_sol   = np.copy(H)   #for enthalpy
T_sol   = np.copy(T)   #for Temperature
phi_w_sol =np.copy(phi_w) #for wetting phase volume fraction
phi_i_sol=np.copy(phi_i) #for non wetting phase volume fraction
q_w_new_sol = np.zeros((Grid.Nf,1)) #placeholder for wetting face Darcy flux
q_nw_new_sol= np.zeros((Grid.Nf,1)) #placeholder for non-wetting face Darcy flux
phi_w_sol = phi_w.copy()


#injection
dof_inj   = Grid.dof_ymin[  np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]
dof_f_inj = Grid.dof_f_ymin[np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]


######################################################################
#Boundary conditions
######################################################################

#Boundary condition for saturation equation
BC.dof_dir   = np.array([])
BC.dof_f_dir = np.array([])
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])
BC.C_g    = np.array([]) 
[B,N,fn]  = build_bnd(BC, Grid, I)

# BC for Enthalpy equation (total)

dof_fixedH = np.setdiff1d(Grid.dof_ymin,dof_inj)
dof_f_fixedH = np.setdiff1d(Grid.dof_f_ymin,dof_f_inj)

Param.H.dof_dir = np.array(Grid.dof_ymax)
Param.H.dof_f_dir = np.array(Grid.dof_f_ymax)
Param.H.g = H[Grid.dof_ymax-1]

Param.H.dof_neu = np.array([dof_inj])
Param.H.dof_f_neu = np.array([dof_f_inj])
Param.H.qb = 50*np.ones((len(dof_inj),1))
[H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

G_original = G.copy() 
D_original = D.copy() 

t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))

i = 0

######################################################################
#Time loop starts
######################################################################
H_flux_array = np.array([])
while time<tmax:
    if break_type == 'equal':
        if pulse_type =='sine':
         
            mean_flux = (On_flux + Off_flux) /2  
            dflux     = (On_flux - Off_flux) /2 
            
            Param.H.qb = mean_flux + dflux * np.sin(2*np.pi*time/(day2s*npp*2))* np.ones((len(dof_inj),1))
            
        else:    
            if np.floor(((time/day2s)/npp)%2) == 0.0 and time < tf - lag*day2s:# On (Odd cycle)  #time >npp*day2s:
                Param.H.qb = On_flux*np.ones((len(dof_inj),1))

            else: #Off (Even cycle)

                Param.H.qb = Off_flux*np.ones((len(dof_inj),1))

        if time > ncycles*2*npp*day2s: Param.H.qb = np.zeros((len(dof_inj),1))   #making flux zero after cycles
        
    else:

        if ((time/day2s)%(npp+break_length)) <= npp and time < tf - lag*day2s :# On (Odd cycle)  #time >npp*day2s:

            Param.H.qb= On_flux*np.ones((len(dof_inj),1)) 

        else: #Off (Even cycle)

            Param.H.qb= Off_flux*np.ones((len(dof_inj),1)) 

    G = G_original.copy()
    D = D_original.copy()
    ################################################################################################
    non_porous_vol_frac = 1e-2
    if np.any(phi_i<=non_porous_vol_frac):
        dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
        dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
        dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
        ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
        
        Param.H.dof_neu = ytop
        Param.H.dof_f_neu = dof_f_bnd_melt
        
        G = zero_rows(G,dof_f_bnd_melt-1)
        D = -np.transpose(G)
        D = zero_rows(D,dof_complete_melt-1) #
    ################################################################################################   

    [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)
    
    phi_w_old = phi_w.copy() 
    C_old     = C.copy()      
    flux      = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid) @ f_Cn(phi_w_old,phi,n)))*np.cos(tilt_angle*np.pi/180)
    flux_vert = flux.copy()
    flux_vert[Grid.dof_f<=Grid.Nfx,0] = flux_vert[Grid.dof_f<=Grid.Nfx,0]*np.tan(tilt_angle*np.pi/180)  #making gravity based flux in x direction
    
    if i<=0:
        speed = np.min(f_Cm(1-phi_i,m)*f_Cn(np.array([C_L]),np.array([phi_L]),n)[0]/C_L)
    else:
        speed = f_Cm(phi,m)*f_Cn(phi_w,np.array(phi),n)/(phi_w)  
        speed[np.isnan(speed)] = 0
    
    if np.any(speed[speed>0]):
        dt1   = CFL*Grid.dy/np.max(speed[speed>0]) #Calculating the time step from the filling of volume
    else:
        dt1   = 1e16 #Calculating the time step from the filling of volume

    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()

    #Taking out the domain to cut off single phase region
    
    dof_act  = Grid.dof[phi_w_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        
        #Step 2 Eliminate inactive cells by putting them to constraint matrix
        Param.P.dof_dir = (dof_act)           
        Param.P.dof_f_dir= np.array([])
        Param.P.g       =  -Yc_col[dof_act-1]*np.cos(tilt_angle*np.pi/180) \
                           -Xc_col[dof_act-1]*np.sin(tilt_angle*np.pi/180)  
        Param.P.dof_neu = np.array([])
        Param.P.dof_f_neu = np.array([])
        Param.P.qb = np.array([])

        [B_P,N_P,fn_P] = build_bnd(Param.P,Grid,I)
        Kd  = comp_harmonicmean(Avg,f_Cm(phi,m)) * (Avg @ f_Cn(phi_w_old,phi,n))
        
        
        Kd  = sp.dia_matrix((Kd[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
        L = - D @ Kd @ G
        u = solve_lbvp(L,fn_P,B_P,Param.P.g,N_P)   # Non dimensional water potential
        q_w = - Kd @ G @ u
        
        #upwinding boundary y-directional flux
        if tilt_angle != 90:
            #finding boundary faces
            dof_ysat_faces = dof_f_saturated[dof_f_saturated>=Grid.Nfx]
            
            #removing boundary faces
            dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
            
            ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
            q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,phi_w,phi,sat_threshold)
            
        #upwinding boundary x-directional flux   ####new line
        if tilt_angle != 0:
            #finding boundary faces
            dof_xsat_faces = dof_f_saturated[dof_f_saturated<Grid.Nfx]
            
            #removing boundary faces
            dof_xsat_faces = np.setdiff1d(dof_xsat_faces,np.append(Grid.dof_f_xmin,Grid.dof_f_xmax))
            
            xleft,xright            = find_left_right_cells(dof_xsat_faces,D,Grid)
            q_w[dof_xsat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_xsat_faces-1],flux_vert[dof_xsat_faces-1],xright,xleft,phi_w,phi,sat_threshold)
             
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]
        
        res = D @ flux_vert
        
    dt2   = np.abs((phi*sat_threshold - phi*s_gr - phi_w_old)/(res)) #Calculating the time step from the filling of volume
    dt2  =  CFL*np.min(dt2[dt2>1e-4*z0/fc])


    #Time step of diffusion
    dt3  = np.min(CFL*(Grid.dy**2)/(2*(phi_w*k_w + phi_i*k_i)/(rho_i*phi_i*cp_i+rho_w*phi_w*cp_w))) 
    if np.isnan(dt3): dt3=1e10
    dt = np.min([dt1,dt2,dt3])

    if time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time

    #Explicit Enthalpy update
    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,0,cp_i,cp_w,0,C,L_fusion)

    #Enthalpy
    K_bar = 2.22362*(rho_i/rho_w*(phi_i))**1.885 + phi_w*k_w
    K_bar_edge = sp.dia_matrix((np.array(np.transpose(comp_harmonicmean(Avg,K_bar))),  np.array([0])), shape=(Grid.Nf, Grid.Nf))     #average permeability at each interface
    
    
    ##Update
    #Volume fraction
    #RHS = C_old - dt*D@flux  #since the gradients are only zero and 1    
    RHS = C_old - dt*res*rho_w + dt* fn #since the gradients are only zero and 1  
    C   = solve_lbvp(I,RHS,B,BC.C_g,N)

    h_i,h_w,h_nw = eval_h(Tm,T,rho_i,rho_w,0,cp_i,cp_w,0,L_fusion)
    
    if diffusion == 'yes':
        RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)  -K_bar_edge @ G @ T  ) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 
    else:
        RHS    =  H - dt * (D @ (rho_w * flux_vert * comp_harmonicmean(Avg,h_w)) -(fs_theta+H_fn)) # -kappa_edge @ G @ H 

    H      =  solve_lbvp(I,RHS,H_B,Param.H.g,H_N)

    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
    phi    = 1-phi_i
    phi_nw = 1 -phi_i -phi_w
    time = time + dt    

    if np.isin(time,t_interest):
        t.append(time) 
        s_w_sol = np.concatenate((s_w_sol,s_w), axis=1) 
        H_sol   = np.concatenate((H_sol,H), axis=1) 
        T_sol   = np.concatenate((T_sol,T), axis=1) 
        phi_w_sol    = np.concatenate((phi_w_sol,phi_w), axis=1) 
        phi_i_sol   = np.concatenate((phi_i_sol,phi_i), axis=1) 
        q_w_new_sol  = np.concatenate((q_w_new_sol,flux_vert), axis=1) 
        H_flux_array = np.append(H_flux_array,Param.H.qb[0,0])
        
        if len(dof_act)< Grid.N:
            print(i,time/day2s,'Saturated cells',Grid.N-len(dof_act))        
        else:    
            print(i,time/day2s)
    i = i+1
    

t = np.array(t)


######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,H_flux_array=H_flux_array)


'''
######################################################################
#for loading data
######################################################################
data = np.load('paper_original_net_zeroT-10CLWC0.03npp4phi[0.5]T-10npp4cycles4lag8daysbreakequallength5_C0.03_2by200_t3456000.npz')
t=data['t']
phi_w_sol =data['phi_w_sol']
phi_i_sol =data['phi_i_sol']
H_sol =data['H_sol']
T_sol =data['T_sol']
s_w_sol =data['s_w_sol']

phi=data['phi']
Xc=data['Xc']
Yc=data['Yc']
Xc_col=data['Xc_col']
Yc_col=data['Yc_col']
Grid.Nx=data['Grid_Nx']
Grid.Ny=data['Grid_Ny']
Grid.xc=data['Grid_xc']
Grid.yc=data['Grid_yc']
Grid.xf=data['Grid_xf']
Grid.yf=data['Grid_yf']
[dummy,endstop] = np.shape(phi_w_sol)
#C[:,0] = phi_w_sol[:,-1]
'''

tday = t/day2s
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]



#combined 1D plot
phi_sol = 1- phi_i_sol
from matplotlib import rcParams
rcParams.update({'font.size': 22})

Grid.ymax = 3
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True,figsize=(15,15),dpi=100)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_i_sol<non_porous_vol_frac] = np.nan
phi_i_sol[phi_i_sol< non_porous_vol_frac] = np.nan
T_sol[phi_i_sol<non_porous_vol_frac] = np.nan

#####

t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(4,1,1)
plt.plot(tday[:-1],H_flux_array,'r-',label='Data',linewidth=2)
plt.axhline(y=(On_flux+Off_flux)/2,xmin=0, xmax=40,linestyle='--',color='black')
#plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)


plt.subplot(4,1,2)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(1-np.max(phi_i_array[~np.isnan(phi_i_array)]),1-np.min(phi_i_array[~np.isnan(phi_i_array)]),10000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'z [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-100, y=0.5, rotation=90)

plt.subplot(4,1,3)

plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'z [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)

plt.subplot(4,1,4)
plot = [plt.contourf(t_array, depth_array, T_array,cmap="Reds",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'z [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ.pdf',bbox_inches='tight', dpi = 100)



black_array= np.array([[0,0,0], [43,43,43], [85,85,85], [128,128,128] , [170,170,170], [213,213,213]])/255; 
Grid.ymax = 5
from matplotlib import rcParams
rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(5,15) , dpi=100)
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t/day2s == 0)[0][0]],Grid.yc,'k-',label=f'{0*npp} days',color = black_array[5,:])
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t/day2s == 2*npp)[0][0]],Grid.yc,'k-',label=f'{2*npp} days',color = black_array[4,:])
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t/day2s == 4*npp)[0][0]],Grid.yc,'k-',label=f'{4*npp} days',color = black_array[3,:])
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t/day2s == 6*npp)[0][0]],Grid.yc,'k-',label=f'{6*npp} days',color = black_array[2,:])
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t/day2s == 8*npp)[0][0]],Grid.yc,'k-',label=f'{8*npp} days',color = black_array[1,:])
plt.plot(1-phi_i_sol[0:Grid.Ny,np.argwhere(t == tf)[0][0]],Grid.yc,'k-',label=f'{10*npp} days',color = black_array[0,:])
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$\phi$')
#plt.legend(loc='lower right',frameon=False)
plt.xlim([0.2, 1])
plt.ylim([Grid.ymax,Grid.ymin])
plt.tight_layout()
plt.xticks([0.2,0.6,1.0])
plt.savefig(f'./combined_phi.pdf',bbox_inches='tight', dpi = 600)


'''
######################################################################
#Run Reparately to make a Movie
######################################################################
blue = np.array([135, 206, 235])/255
#A fun combined  phi, LWC, T
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
tday = t/day2s
ymax_loc =5
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation

######
#Remove 99% melted ice
phi_w_sol_backup[phi_i_sol<non_porous_vol_frac] = np.nan
phi_w_sol[phi_i_sol<non_porous_vol_frac] = np.nan
phi_i_sol[phi_i_sol< non_porous_vol_frac] = np.nan
T_sol[phi_i_sol<non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan
phi_w_sol[np.isnan(phi_i_sol)] = np.nan

#####

[N,frn] = np.shape(T_sol) # frame number of the animation from the saved file
[Xy,Yf] = np.meshgrid(Grid.xf,Grid.yf)
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
ax3.set_ylim([ymax_loc,Grid.ymin])
ax3.set_xlabel(r'T[$^\circ$C]')


ax1.set_xlim([0,1])
ax1.set_ylim([ymax_loc,Grid.ymin])
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'Depth $[m]$')


#ax2.set_xlim([np.min(phi_w_sol/(1-phi_i_sol)),np.max(phi_w_sol/(1-phi_i_sol))])
ax2.set_xlim([0,0.1])
ax2.set_ylim([ymax_loc,Grid.ymin])
ax2.set_xlabel(r'$LWC$')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

x = Yc[:,1]
y = np.zeros_like(x)
z = np.zeros_like(x)
a = np.zeros_like(Yf[:,1])
face = Yf[:,1]

line1, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-')
line2, = ax2.plot(z, x, lw=2, color= 'k')
line3, = ax3.plot(y, x, lw=2, color= 'k',linestyle='--')
line4, = ax3.plot(z, x, lw=2, color= 'k')
line5, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-',alpha=0.5)


ax2.set_title("Time: %0.2f days" %t[0],loc = 'center', fontsize=18)
plt.tight_layout()
line   = [line1, line2, line3, line4, line5]
 
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
t_dates = []
t_dates = ([dates[850] + timedelta(seconds = i) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])

# initialization function: plot the background of each frame
def init():
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
    line[3].set_data([], [])
    line[4].set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i,u_sol,Sw_sol,phi_w_sol,phi_i_sol,Yc,Yf,time):
    ax1.set_xlim([0,1])
    x = Yc[:,1] 
    y = np.transpose(u_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]-273.16    
    z = np.transpose(phi_w_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz= 1-np.transpose(phi_i_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz0= 1-np.transpose(phi_i_sol[:,0].reshape(Grid.Nx,Grid.Ny))[:,1]                           
    Tmelt = 0.0*Yc[:,1]
    line[0].set_data( zz, x)
    line[1].set_data(z, x)
    line[2].set_data(Tmelt, x)
    line[3].set_data(y, x)
    line[4].set_data( zz0, x)
    if np.any(np.isnan(phi_w_sol[:,i])): 
        ax1.axhline(y=np.max(Yc_col[np.isnan(phi_i_sol[:,i])]), c=blue,linestyle='-',lw=3)
        ax2.axhline(y=np.max(Yc_col[np.isnan(phi_i_sol[:,i])]), c=blue,linestyle='-',lw=3)
        ax3.axhline(y=np.max(Yc_col[np.isnan(phi_i_sol[:,i])]), c=blue,linestyle='-',lw=3)
    else:
        ax1.axhline(y=0, c=blue,linestyle='-',lw=3) 
        ax2.axhline(y=0, c=blue,linestyle='-',lw=3)
        ax3.axhline(y=0, c=blue,linestyle='-',lw=3)
    #plt.tight_layout()
    
    
    ax2.set_title("Time: %0.2f days" %tday[i],loc = 'center', fontsize=18)
    #ax1.legend(loc='lower right', shadow=False, fontsize='medium')
    #ax2.legend(loc='best', shadow=False, fontsize='medium')
    ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
    ax1.set_xlim([0.2,1])
    ax2.set_xlim([0,0.1])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
ii=1
ani = animation.FuncAnimation(fig, animate, frn, init_func=init, fargs=(T_sol[:,::ii],s_w_sol[:,::ii],phi_w_sol[:,::ii],phi_i_sol[:,::ii],Yc[:,:],Yf[:,:],tday[::ii])
                               , interval=1/fps)

ani.save(f"../Figures/{simulation_name}_combined_with_seconds_new.mov", writer='ffmpeg', fps=30)

'''

