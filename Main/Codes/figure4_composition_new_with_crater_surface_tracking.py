######################################################################
#Figure 4 - Dye 2, 2016
#Mohammad Afzal Shadab
#Date modified: 05/03/2022
######################################################################

######################################################################
#import libraries
######################################################################

import sys
sys.path.insert(1, '../../solver')

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
from comp_mean_matrix import comp_mean_matrix
from eval_phase_behavior import eval_phase_behaviorCwH

#Samira Site A actual init temp but RetMIP porosity
from RetMIP_data_analysis import RetMIP_fit_porosity, Samira_data_actual_fit_temp, RetMIP_acc_rate
fit_temp     = Samira_data_actual_fit_temp
fit_porosity = RetMIP_fit_porosity
from colliander_data_analysis import spun_up_profile_May2016_Colliander, Qnet_May2Sept2016_Samira
day,Qnet,Qnet_func             = Qnet_May2Sept2016_Samira() #Samira

from scipy import integrate

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
simulation_name = f'SamiraA_withSamimiflux_lowres_absolute_k0_onetimes_phiexp1.885_cond_'
diffusion = 'yes'
CFL    = 0.1     #CFL number
tilt_angle = 0   #angle of the slope

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
npp    = 4   #number of positive days [days]

##thermodynamics
rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_nw = 1.225  # density of gas [kg/m^3]
phi_nw_init = 0.5    # volumetric ratio of gas, decreasing exponentially
cp_nw  = 1003.5  # specific heat of gas at constant pressure [J/(kg K)]
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
tf     = 120*day2s
tmax = tf#0.07#0.0621#2 #5.7#6.98  #time scaling with respect to fc
#t_interest = [0,0.25,0.5399999999999999 + 0.005,0.6,0.7116505061468059,1.0] #swr,sgr=0.05
t_interest = np.linspace(0,tmax,int(tf/day2s*24)+1)   #swr,sgr=0

Nt   = 1000
dt = tmax / (Nt)

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
Grid.xmin =  0*z0; Grid.xmax =1000e3; Grid.Nx = 2;    #Horizontal direction
Grid.ymin =  0*z0; Grid.ymax =5;  Grid.Ny = 200;      #Vertically downward direction
Grid = build_grid(Grid)   #building grid
[D,G,I] = build_ops(Grid) #building divergence, gradient and identity operators
D  = -np.transpose(G)
Avg     = comp_mean_matrix(Grid)  #building mean operator

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

s_wfunc    = lambda phi_w,phi_nw: phi_w  / (phi_w + phi_nw)
s_nwfunc   = lambda phi_w,phi_nw: phi_nw / (phi_w + phi_nw)
T_annual_func_sigmoid = lambda Tbot, Ttop, Yc_col, Y0: Tbot + (Ttop - Tbot)/Y0*(Yc_col) 

#Initial conditions
phi_nw  = fit_porosity(Yc_col) #volume fraction of gas 

phi_w   = np.zeros_like(phi_nw)  #volume fraction of water phase #No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T       = fit_temp(Yc_col)+Tm
T[Yc_col>3.7] = -15.7 + Tm
H       = enthalpyfromT(fit_temp(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)

phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1))#np.exp(-(1-Yc_col/(grid.ymax-grid.ymin)))#phi*np.ones((grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()#s_wp *np.ones((grid.N,1))
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation

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
#boundary condition for saturation equation
BC.dof_dir   = np.array([])
BC.dof_f_dir = np.array([])
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])
BC.C_g    = np.array([])#C[dof_inj-1] #+ rho_w*C_L*np.ones((len(dof_inj),1))
[B,N,fn]  = build_bnd(BC, Grid, I)
# Enthalpy equation (total)

dof_fixedH = np.setdiff1d(Grid.dof_ymin,dof_inj)
dof_f_fixedH = np.setdiff1d(Grid.dof_f_ymin,dof_f_inj)
#Param.H.dof_dir = np.hstack([dof_fixedH,Grid.dof_ymax,Grid.dof_xmin[1:-1],Grid.dof_xmax[1:-1]])
#Param.H.dof_f_dir = np.hstack([dof_f_fixedH ,Grid.dof_f_ymax,Grid.dof_f_xmin[1:-1],Grid.dof_f_xmax[1:-1]])
#Param.H.g  = np.hstack([H[dof_fixedH-1,0],H[Grid.dof_ymax-1,0],H[Grid.dof_xmin[1:-1]-1,0],H[Grid.dof_xmax[1:-1]-1,0]])#np.hstack([0*np.ones_like(Grid.dof_ymin),LWC_top*rho_w*L_fusion*np.ones_like(Grid.dof_ymin)])


Param.H.dof_dir = np.array([])
Param.H.dof_f_dir = np.array([])
Param.H.g  = np.array([])#np.hstack([0*np.ones_like(Grid.dof_ymin),LWC_top*rho_w*L_fusion*np.ones_like(Grid.dof_ymin)])

Param.H.dof_neu = dof_inj
Param.H.dof_f_neu = dof_f_inj
Param.H.qb = np.zeros((len(dof_inj),1))
[H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))

G_original = G.copy() 
D_original = D.copy()  

i = 0

length_added_final = 0

######################################################################
#Time loop starts
######################################################################
while time<tmax:
    surf_loc = 0
    Param.H.qb= Qnet_func(time/day2s)*np.ones((len(dof_inj),1)) 
    [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)

    #BC.C_g    = C[dof_inj-1] 
    #Param.H.qb= Qnet_func(time/day2s)*np.ones((Grid.Nx,1))

    #Param.H.qb= Qnet_func(time/day2s)*np.transpose([np.concatenate([np.flip(np.linspace(1,1.1,int(Grid.Nx/2))),np.linspace(1,1.1,int(Grid.Nx/2))])])
    G = G_original.copy()
    D = D_original.copy()
    ################################################################################################
    non_porous_vol_frac = 1e-3
    if np.any(phi_i<=non_porous_vol_frac):
        dof_complete_melt  = Grid.dof[phi_i[:,0] < non_porous_vol_frac]
        dof_partial_melt   = np.setdiff1d(Grid.dof,dof_complete_melt) #saturated cells 
        dof_f_bnd_melt     = find_faces(dof_complete_melt,D,Grid) 
        ytop,ybot          = find_top_bot_cells(dof_f_bnd_melt,D,Grid)
        
        Param.H.dof_neu = ytop
        Param.H.dof_f_neu = dof_f_bnd_melt
        Param.H.qb = Qnet_func(time/day2s)*np.ones((Grid.Nx,1))
        [H_B,H_N,H_fn] = build_bnd(Param.H,Grid,I)
        
        G = zero_rows(G,dof_f_bnd_melt-1)
        D = -np.transpose(G)
        D  = zero_rows(D,dof_complete_melt-1) #
    
        ############################################################
        #location of the surface
        surf_loc = (Yc_col[ytop[0]-1] + Yc_col[ybot[0]-1]) / 2
    ################################################################################################     

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

    #flux_vert_mat = sp.dia_matrix((flux_vert[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    #dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < 0.99999]
    dof_act  = Grid.dof[phi_w_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        #Step 1: Modify the gradient to import natural BC at the crater
        #G_small = zero_rows(G,dof_f_saturated-1)
        
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

    #if i<10: 
    #    dt = tmax/(Nt*10)
    if time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time

    #Explicit Enthalpy update
    phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,0,cp_i,cp_w,0,C,L_fusion)

    #Enthalpy
    #calculating K_bar using K_bar = phi_w*k_w + phi_i*k_i
    #K_bar = phi_w=*k_w + phi_i*k_i
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
    
    #adding accumulation
    if (RetMIP_acc_rate(time/day2s) > 0) and (surf_loc > 0):  #if accumulation > 0 and surface is below top
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        #length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        length_added = integrate.qmc_quad(RetMIP_acc_rate, (time-dt)/day2s, (time)/day2s)[0] * day2s * rho_w /315 #'meter' ice equivalent of solid ice accumulation #dt is involved
        
        
        if length_added_final < Grid.dy:  #insufficient to fill one cell
            length_added_final = length_added_final + length_added 
                
        else: #sufficient to fill one cell atleast
            if  length_added_final == 0:   length_added_final =  length_added 

            surf_loc = surf_loc - length_added_final     #new surface location [m]        
            dof_fresh = dof_complete_melt[np.ravel(Yc_col[dof_complete_melt-1] > surf_loc)] #cells of freshly fallen snow
            if np.any(dof_fresh): print(dof_fresh)
            H[dof_fresh-1] = 0.0   #adding temperate ice on top
            C[dof_fresh-1] = 315   #adding temperate, freshly fallen snow on top [kg/m3]
            phi_i[dof_fresh-1]   = 315/rho_i
            phi_w[dof_fresh-1]   = 0 
            phi_nw[dof_fresh-1]  = 1 - 315/rho_i
            print(time,'\t',length_added_final)
            length_added_final = 0  #re-setting the length to zero for next cycle
    ####################################################################################

    if np.isin(time,t_interest):
        t.append(time) 
        s_w_sol = np.concatenate((s_w_sol,s_w), axis=1) 
        H_sol   = np.concatenate((H_sol,H), axis=1) 
        T_sol   = np.concatenate((T_sol,T), axis=1) 
        phi_w_sol    = np.concatenate((phi_w_sol,phi_w), axis=1) 
        phi_i_sol   = np.concatenate((phi_i_sol,phi_i), axis=1) 
        q_w_new_sol  = np.concatenate((q_w_new_sol,flux_vert), axis=1) 
        
        if len(dof_act)< Grid.N:
            print(i,time/day2s,'Saturated cells',Grid.N-len(dof_act))      
        else:    
            print(i,time/day2s)
    i = i+1
    

t = np.array(t)
tday = t/day2s

######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)


np.savez(f'{simulation_name}_T_surface.npz', tday=tday,T_surf=T_sol[0,:])

'''
######################################################################
#for loading data
######################################################################
data = np.load('SamiraA_withSamimiflux_lowres_absolute_k0_onetimes_phiexp1.885_cond__C0.03_2by200_t10368000.npz')
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
phi_sol = 1- phi_i_sol
'''

######################################################################
#plotting
######################################################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

######################################################################
#combined 1D plot with upGPR with actual dates and Temperatures + LWC combined
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
phi_sol = 1 - phi_i_sol
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx',header=(1))


#red_array  = np.array([[255,0,0], [255,43,43], [255,85,85], [255,128,128] , [255,170,170], [255,213,213]])/255;
#black_array= np.array([[0,0,0], [43,43,43], [85,85,85], [128,128,128] , [170,170,170], [213,213,213]])/255; 
#blue_array = np.array([[0,0,255], [43,43,255], [85,85,255], [128,128,255] , [170,170,255], [213,213,255]])/255;

red_array  = np.array([[255,0,0], [255,106.5,106.5] , [255,106.5,106.5] , [255,106.5,106.5] , [255,106.5,106.5], [255,213,213]])/255;
black_array= np.array([[0,0,0], [106.5,106.5,106.5], [106.5,106.5,106.5], [106.5,106.5,106.5] , [106.5,106.5,106.5], [213,213,213]])/255; 
blue_array = np.array([[0,0,255], [106.5,106.5,255], [106.5,106.5,255], [106.5,106.5,255] , [106.5,106.5,255], [213,213,255]])/255;

data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016
Samira_dates = df['Time']
Samira_dates_index = np.where((Samira_dates>=dates[850]) & (Samira_dates<=dates[len(dates)-1]))

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))

from matplotlib import rcParams
rcParams.update({'font.size': 22})
Grid.ymax = 3
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharex=True,figsize=(15,15),dpi=50)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_w_sol[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan
phi_w_sol[np.isnan(phi_i_sol)] = np.nan

#####

t_array=np.kron(tday,np.ones((Grid.Ny,1)))
#Make actual time array
import pandas as pd
t_dates = ([dates[850] + timedelta(seconds = i) for i in t])
t_dates = np.array([[np.datetime64(i) for i in t_dates]])
t_dates = np.repeat(t_dates, Grid.Ny, axis=0)


phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array[T_array>Tm] = Tm
T_array[T_array<Tm-22] = Tm-22

plt.subplot(5,1,1)
plt.plot(([dates[850] + timedelta(days = i) for i in day]),Qnet,'r-',label='Data',linewidth=2)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]',color='red')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
plt.yticks([0,25,50,75])
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax1.xaxis.grid(True, which='major', color='k', linestyle='--')
ax1.spines['left'].set_color('red')
ax1.tick_params(axis='y', colors='red')

ax1new = ax1.twinx()

color = 'tab:blue'
ax1new.spines['right'].set_color('blue')
ax1new.set_ylabel(r'$a$ [mm.w.e./day]', color='blue')  # we already handled the x-label with ax1
ax1new.plot(([dates[850] + timedelta(days = i) for i in day]), RetMIP_acc_rate(day)*1e3*day2s , color='blue')
ax1new.tick_params(axis='y', colors='blue')
plt.yticks([0,10,20])

plt.subplot(5,1,2)
plot = [plt.contourf(t_dates, depth_array, (1-phi_i_array),cmap="Greys",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(1-np.max(phi_i_array[~np.isnan(phi_i_array)]),1-np.min(phi_i_array[~np.isnan(phi_i_array)]),10000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'z [m]')
#plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-100, y=0.5, rotation=90)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax2.xaxis.grid(True, which='major', color='k', linestyle='--')

plt.subplot(5,1,4)
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.1'])[Samira_dates_index[0]],'-',label='0.3 m',color = black_array[0,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.2'])[Samira_dates_index[0]],'-',label='0.6 m',color = black_array[1,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.3'])[Samira_dates_index[0]],'-',label='0.9 m',color = black_array[2,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.4'])[Samira_dates_index[0]],'-',label='1.4 m',color = black_array[3,:])
#plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.5'])[Samira_dates_index[0]],'-',label='1.8 m',color = black_array[4,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.6'])[Samira_dates_index[0]],'-',label='2.1 m',color = black_array[5,:])

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
Samira_siteA_depth_array_iter = np.array([0.3, 0.9, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in Samira_siteA_depth_array_iter:   
    print(i)
    index = np.where(Yc_col<i)[0][-1]
    plt.plot(t_dates[0,:],(T_sol[index,:]+T_sol[index+1,:])/2-273.16,'r-',color=red_array[np.argwhere(i==Samira_siteA_depth_array)][0,0])
#plt.legend(fontsize='medium', ncol=2)
plt.ylabel(r'T[$^\circ$C]')
ax4.xaxis.grid(True, which='major', color='k', linestyle='--')

phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_w_array[phi_w_array>0.06] = 0.06

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-100, y=0.5, rotation=90)


plt.subplot(5,1,3)
T_array[T_array >= Tm]  = np.nan
new1 = ax3.contourf(t_dates, depth_array, T_array-Tm,cmap="Reds",levels=100,ls=None)
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)
ax3.xaxis.grid(True, which='major', color='k', linestyle='--')


T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array[T_array>Tm] = Tm
T_array[T_array<Tm-22] = Tm-22
#phi_w_array[phi_w_array>0.06] = 0.06
phi_w_array[T_array < Tm] = np.nan
new2 = ax3.contourf(t_dates, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)

plt.plot(dates[850:4200],data['height'][850] - data['height'][850:4200],'k-')

plt.subplot(5,1,5)

phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
#phi_w_array[phi_w_array>0.06] = 0.06

data_LWC  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/samimi_LWC.csv')

data_LWC[np.isnan(data_LWC)] = -10

plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.3m']]),data_LWC['0.3m.1']-data_LWC['0.3m.1'][0],'ko',label='0.3 m',  color = black_array[0,:], markersize=4)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.6m']]),data_LWC['0.6m.1']-data_LWC['0.6m.1'][0],'ks',label='0.6 m',color = black_array[1,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.9m']]),data_LWC['0.9m.1']-data_LWC['0.9m.1'][0],'kd',label='0.9 m',color = black_array[2,:], markersize=4)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.4m']]),data_LWC['1.4m.1']-data_LWC['1.4m.1'][0],'k^',label='1.4 m',color = black_array[3,:], markersize=3)
#plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.8m']]),data_LWC['1.8m.1']-data_LWC['1.8m.1'][0],'k>',label='1.8 m',color = black_array[4,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['2.1m']]),data_LWC['2.1m.1']-data_LWC['2.1m.1'][0],'k<',label='2.1 m',color = black_array[5,:], markersize=4)

plt.ylabel(r'LWC')

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in [0,2,5]:   
    index = np.where(Yc_col<Samira_siteA_depth_array[i])[0][-1]
    print(index)
    print(i)
    plt.plot(t_dates[0,:],(phi_w_sol[index,:]+phi_w_sol[index+1,:])/2,'-',color = blue_array[i,:])

plt.ylim(0,0.0275)
plt.xlim()
#plt.legend( ncol=2)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
ax5.xaxis.grid(True, which='major', color='k', linestyle='--')
plt.xlabel(r'Calendar date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withQAcc_actual_dateswithT_TplusLWC.pdf',bbox_inches='tight', dpi = 50)




'''
######################################################################
#Making video
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
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_w_sol[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
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
def animate(i,u_sol,Sw_sol,phi_w_sol,phi_i_sol,Yc,Yf,time,t_dates):
    
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
        ax1.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
        ax2.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
        ax3.axhline(y=np.max(Yc_col[np.isnan(phi_w_sol[:,i])]), c=blue,linestyle='-',lw=3)
    else:
        ax1.axhline(y=0, c=blue,linestyle='-',lw=3) 
        ax2.axhline(y=0, c=blue,linestyle='-',lw=3)
        ax3.axhline(y=0, c=blue,linestyle='-',lw=3)
    #plt.tight_layout()
    
    
    ax2.set_title(f"{np.datetime_as_string(t_dates[0][i], unit='s')}",loc = 'center', fontsize=18)
    #ax1.legend(loc='lower right', shadow=False, fontsize='medium')
    #ax2.legend(loc='best', shadow=False, fontsize='medium')
    ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
    ax1.set_xlim(np.min([1-phi_i_sol[~np.isnan(phi_i_sol)]]),np.max(1-phi_i_sol[~np.isnan(phi_i_sol)]))
    ax2.set_xlim([0,0.1])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
ii=1
ani = animation.FuncAnimation(fig, animate, frn, init_func=init, fargs=(T_sol[:,::ii],s_w_sol[:,::ii],phi_w_sol[:,::ii],phi_i_sol[:,::ii],Yc[:,:],Yf[:,:],tday[::ii],t_dates[::ii])
                               , interval=1/fps)

ani.save(f"../Figures/{simulation_name}_combined_with_seconds_new.mov", writer='ffmpeg', fps=30)

'''


#New plots

fc = k0*k_w0*rho_w*grav/mu_w*phi_L**3 #Infiltration capacity (m/s)

fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

Flux_array = fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3 #Flux evaluation
Flux_array[phi_sol>1-non_porous_vol_frac*100] = np.nan

plot = [plt.contourf(t_dates, depth_array, Flux_array[:200,:],cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Flux_array[~np.isnan(Flux_array)]),np.max(Flux_array[~np.isnan(Flux_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Flux$ [m/s]', labelpad=-100, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_Fluxes.pdf',bbox_inches='tight', dpi = 50)


fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

Flux_array = fc*(1-phi_i_sol)**(m)/phi_L**3 #Flux evaluation
Flux_array[phi_sol>1-non_porous_vol_frac*100] = np.nan

plot = [plt.contourf(t_dates, depth_array, Flux_array[:200,:],cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Flux_array[~np.isnan(Flux_array)]),np.max(Flux_array[~np.isnan(Flux_array)]),10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Flux$ [m/s]', labelpad=-150, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_Flux2saturate.pdf',bbox_inches='tight', dpi = 50)


#Analytical plot Marc for 50% porosity
fig, ax1 = plt.subplots(2,1, sharex=True,figsize=(6,12),dpi=100)
rcParams.update({'font.size': 22})
phi0 =0.5
s_w_analy = np.linspace(0/phi0,0.5/phi0,1000)
Flux_analy= fc*s_w_analy**n*(phi0)**(m)/phi_L**3 #Flux evaluation

plt.subplot(2,1,1)
plt.plot(s_w_analy,Flux_analy*1e5,'b-',color=blue,label='$s_{wr}=0$')

s_w_analy1= np.linspace(0.07,0.5/phi0,1000)
Flux_analy1= fc*(s_w_analy-0.07)**n*(phi0)**(m)/phi_L**3 #Flux evaluation
plt.plot(s_w_analy1,Flux_analy*1e5,'k-',label='$s_{wr}=0.07$')
plt.xlim([0,1])
plt.ylim([-1e-6*1e5,np.max([Flux_analy*1e5,Flux_analy1*1e5])])
plt.ylabel(r'Flux, $I \times 10^5$ [m/s]')
plt.legend(loc='best')
plt.plot(0.07,0,'k+',markersize=10)

plt.axvline(0.5,ymin=0,ymax=2,color='black',linestyle='-.',alpha=0.5)
plt.plot([0,0.5], [0,1.71], color=blue,linestyle="--")
plt.plot([0,0.5], [0,1.43], color='black',linestyle="--")

plt.subplot(2,1,2)
plt.plot(s_w_analy,Flux_analy*1e5,'b-',color=blue,label='$s_{wr}=0$')

s_w_analy1= np.linspace(0.07,0.5/phi0,1000)
Flux_analy1= fc*(s_w_analy-0.07)**n*(phi0)**(m)/phi_L**3 #Flux evaluation
plt.plot(s_w_analy1,Flux_analy*1e5,'k-',label='$s_{wr}=0.07$')
plt.xlim([0,1])
plt.ylim([-1e-6*1e5,np.max([Flux_analy*1e5,Flux_analy1*1e5])])
plt.ylabel(r'Flux, $I \times 10^5$ [m/s]')

plt.xlabel('Water saturation, $LWC/\phi$ [-]')
plt.plot(0.07,0,'k+',markersize=10)
plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

plt.axhline(1.71,xmin=0,xmax=1,color='black',linestyle='-.',alpha=0.5)


plt.plot([0,0.5], [0,1.71], color=blue,linestyle="--")
plt.plot([0,0.532], [0,1.71], color='black',linestyle="--")
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_res_sat_comparison.pdf',bbox_inches='tight', dpi = 50)

#Peclet number estimation for 50% porosity
K_h = k0*k_w0*rho_w*grav/mu_w #Infiltration capacity (m/s)

LWC = 0.03; Tfirn = -10; phi0 =0.5; t_pulse = 4*day2s
K_bar = 2.22362*(rho_i/rho_w*(1-phi0))**1.885
kk_bar = K_bar/(rho_i*cp_i*(1-phi0))
v= (fc*LWC**n*(phi0)**(m-n)/phi_L**3 - 0)/(0.00573*(1-phi0)*(-Tfirn) + LWC - 0)
grain_size= 2e-3 #grainsize 1mm or 2mm Humphrey et al. (2021)
Pe_grain = v*grain_size/kk_bar
Pe_cell =  v*Grid.dy/kk_bar
Pe_cell_shock =  v*(v*4*day2s)/kk_bar


#Peclet number wrong and rough comparison
fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

speed_array = (fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3)/(phi_w_sol + 0.00573*phi_i_sol*(Tm-T_sol)) #Rough speed evaluation
speed_array[phi_sol>0.8] = np.nan
K_bar_array = 2.22362*(rho_i/rho_w*phi_i_sol)**1.885
kk_bar_array = K_bar_array/(rho_i*cp_i*phi_i_sol)

Pe_L_array = speed_array*Grid.dy/kk_bar_array
Pe_grain_array = speed_array*grain_size/kk_bar_array

plot = [plt.contourf(t_dates, depth_array, Pe_L_array[:200,:],cmap="Blues",levels=100,vmin=np.min(Pe_L_array[~np.isnan(Pe_L_array)]),vmax=np.max(Pe_L_array[~np.isnan(Pe_L_array)])/2,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Pe_L_array[~np.isnan(Pe_L_array)]),np.max(Pe_L_array[~np.isnan(Pe_L_array)])/2,10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Pe_{cell}$', labelpad=-105, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_wrong_PeL.pdf',bbox_inches='tight', dpi = 50)


fig, ax1 = plt.subplots(1,1, sharex=True,figsize=(20,6),dpi=50)
rcParams.update({'font.size': 22})
Grid.ymax = 2.6

speed_array = (fc*phi_w_sol**n*(1-phi_i_sol)**(m-n)/phi_L**3)/(phi_w_sol + 0.00573*phi_i_sol*(Tm-T_sol)) #Rough speed evaluation
speed_array[phi_sol>0.8] = np.nan
K_bar_array = 2.22362*(rho_i/rho_w*phi_i_sol)**1.885
kk_bar_array = K_bar_array/(rho_i*cp_i*phi_i_sol)

Pe_L_array = speed_array*Grid.dy/kk_bar_array
Pe_grain_array = speed_array*grain_size/kk_bar_array

plot = [plt.contourf(t_dates, depth_array, Pe_grain_array[:200,:],cmap="Blues",levels=100,vmin=np.min(Pe_grain_array[~np.isnan(Pe_grain_array)]),vmax=np.max(Pe_grain_array[~np.isnan(Pe_grain_array)])/2,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(Pe_grain_array[~np.isnan(Pe_grain_array)]),np.max(Pe_grain_array[~np.isnan(Pe_grain_array)])/2,10000))
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$Pe_{grain}$', labelpad=-150, y=0.5, rotation=90)
plt.ylim([Grid.ymax,Grid.ymin])

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

chosen_dates = [20160524,20160918]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xlim(chosen_dates)

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.ylabel(r'z [m]')
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_only_wrong_Pegrain.pdf',bbox_inches='tight', dpi = 50)


