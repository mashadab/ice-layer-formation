#Coding the Hyperbolic Richard's solver with CH thermodynamics
#Multiple_layers
#Mohammad Afzal Shadab
#Date modified: 05/03/2022

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

#parameters
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
#L  = 90 #length of the domain (m)
#H  = 15 #height of the domain (m)
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

#tmax = tmax / phi_L**m   #time scaling with respect to K_0 where K_0 = f_c/phi**m
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

#spatial
Grid.xmin =  0*z0; Grid.xmax =1000e3; Grid.Nx = 2; 
Grid.ymin =  0*z0; Grid.ymax =5;  Grid.Ny = 200;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
D  = -np.transpose(G)
Avg     = comp_mean_matrix(Grid)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

s_wfunc    = lambda phi_w,phi_nw: phi_w  / (phi_w + phi_nw)
s_nwfunc   = lambda phi_w,phi_nw: phi_nw / (phi_w + phi_nw)
#T_annual_func = lambda Tbot, Ttop, Yc_col, t0: Tbot + (Ttop - Tbot) * np.exp(-(Grid.ymax-Yc_col)*np.sqrt(np.pi/(t0*kk)) ) * np.sin(np.pi/2 - (Grid.ymax-Yc_col)*np.sqrt(np.pi/(t0*kk)) )
T_annual_func_sigmoid = lambda Tbot, Ttop, Yc_col, Y0: Tbot + (Ttop - Tbot)/Y0*(Yc_col) #* 1/(1+np.exp(-(Grid.ymax-Yc_col)/Y0))

#Initial conditions
phi_nw  = fit_porosity(Yc_col)#np.exp(-(Yc_col/(Grid.ymax-Grid.ymin))) #volume fraction of gas np.ones_like(Yc_col)# #*np.exp(-(Yc_col/(Grid.ymax-Grid.ymin)))
#phi_nw  = phi_nw_init*np.ones_like(Yc_col)#np.exp(-(Yc_col/(Grid.ymax-Grid.ymin))) #volume fraction of gas np.ones_like(Yc_col)# #*np.exp(-(Yc_col/(Grid.ymax-Grid.ymin)))

phi_w   = np.zeros_like(phi_nw) #No water
C       = rho_w * phi_w + (1 - phi_w - phi_nw) * rho_i
T       = fit_temp(Yc_col)+Tm
T[Yc_col>3.7] = -15.7 + Tm
H       = enthalpyfromT(fit_temp(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)

#H       = enthalpyfromT(T_firn*((Yc_col/Grid.ymax))+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion) #Linear
#H       = enthalpyfromT(T_firn*np.ones_like(Yc_col)+Tm,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion) #Constant

'''
T_dummy = T_annual_func_sigmoid (Tm,T_firn+Tm, Yc_col, 0.5)
T_dummy[T_dummy<Tm+T_firn] = Tm+T_firn
H       = enthalpyfromT(T_dummy,Tm,rho_i,rho_w,0,cp_i,cp_w,0,phi_w,phi_nw,L_fusion)  #Exponential, analytic
'''

phi_w,phi_i,T,dTdH = eval_phase_behaviorCwH(H,Tm,rho_i,rho_w,rho_nw,cp_i,cp_w,cp_nw,C,L_fusion)
s_w_init= s_wfunc(phi_w,phi_nw)
phi = (phi_w+ phi_nw)*np.ones((Grid.N,1))#np.exp(-(1-Yc_col/(grid.ymax-grid.ymin)))#phi*np.ones((grid.N,1)) #porosity in each cell
s_w = s_w_init.copy()#s_wp *np.ones((grid.N,1))
fs_theta = 0.0*np.ones((Grid.N,1))                     #RHS of heat equation


#initializing arrays
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

##########

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

'''
Param.H.dof_dir = dof_inj
Param.H.dof_f_dir = dof_f_inj
#Param.H.g  = np.zeros((len(dof_inj),1))#np.hstack([0*np.ones_like(Grid.dof_ymin),LWC_top*rho_w*L_fusion*np.ones_like(Grid.dof_ymin)])
Param.H.g = rho_w*C_L*L_fusion*np.ones((Grid.Nx,1))
'''

'''
Param.H.dof_dir = np.hstack([Grid.dof_ymin, Grid.dof_ymax])
Param.H.dof_f_dir = np.hstack([Grid.dof_f_ymin, Grid.dof_f_ymax])
Param.H.g  = np.hstack([C_L*rho_w*L_fusion*np.ones_like(Grid.dof_ymin), H[Grid.dof_ymax-1,0]])
'''

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

#Grid,s_w,time,tf_new,time,t_interest,i,tf,t,s_w_sol,phi_w_sol,phi_w,phi_i_sol,phi_i,H_sol,H,T_sol,T,phi = spin_up_firn(file_name, Grid, tf_new, n, t_interest)
length_added_final = 0
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
        length_added = RetMIP_acc_rate(time/day2s) * dt * rho_w /315 #'meter' ice equivalent of solid ice accumulation
        
        if length_added_final < Grid.dy:  #insufficient to fill one cell
            length_added_final = length_added_final + length_added 
                
        else: #sufficient to fill one cell atleast
            if  length_added_final == 0:   length_added_final =  length_added 
            
            dof_fresh = dof_complete_melt[np.ravel(Yc_col[dof_complete_melt-1] > surf_loc)] #cells of freshly fallen snow
            
            surf_loc = surf_loc - length_added_final
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

#saving the tensors
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,q_w_new_sol=q_w_new_sol,H_sol=H_sol,T_sol=T_sol,s_w_sol=s_w_sol,phi_w_sol =phi_w_sol,phi_i_sol =phi_i_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)


np.savez(f'{simulation_name}_T_surface.npz', tday=tday,T_surf=T_sol[0,:])

'''
#for loading data
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
'''


###############################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

low_phi_dof = np.union1d(np.intersect1d(np.argwhere(Xc_col >=0.5)[:,0],np.argwhere(Xc_col < 0.75 - (1-Yc_col)/3)), \
                         np.intersect1d(np.argwhere(Xc_col < 0.5)[:,0],np.argwhere(Xc_col > 0.25 + (1-Yc_col)/3)))

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose(phi.reshape(Grid.Nx,Grid.Ny)),20,cmap="coolwarm",levels=20,vmin = 0, vmax = 1)]
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$z$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
##plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(C)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\phi$', labelpad=1, y=1.075, rotation=0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_porosity.pdf',bbox_inches='tight', dpi = 600)




'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file
t_hr    = t/hr2s 
def update_plot(frame_number, zarray, plot,t,phi):
    plt.clf()
    plot[0] = plt.contourf(Xc/1e3, Yc, np.transpose((zarray[:,frame_number]).reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1)
    plt.title("t= %0.4f hours" % t[frame_number],loc = 'center', fontsize=18)
    ##plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(np.min(phi_w_sol), np.max(phi_w_sol))
    plt.ylabel(r'$z$ [m]')
    plt.xlabel(r'$x$ [km]')    
    
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(phi_w_sol)
    mm.set_clim(np.min(phi_w_sol), np.max(phi_w_sol))
    clb = plt.colorbar(mm, pad=0.1)
    clb.set_label(r'$\phi_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc/1e3, Yc, np.transpose((phi_w_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [km]')
plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
plt.ylim([Grid.ymax,Grid.ymin])
##plt.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol)
mm.set_clim(np.min(phi_w_sol), np.max(phi_w_sol))
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\phi_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f hours" %(t_hr[-1]),loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,:], plot[:],t_hr[:],1-phi_i_sol[:,:]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)

'''

'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    zarray  = zarray/phi   
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.05, vmax = 1.05)
    plt.title("t= %0.4f" % t[frame_number],loc = 'center')
    #plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(0,1)

fig = plt.figure(figsize=(10,7.5) , dpi=100)
Ind = C/phi
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.0)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
clb.set_label(r'$s_w$')
plt.title("t= %0.2f s" % t[-1],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,:], plot[:],t[:]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


'''
#New Contour plot combined new
Grid.ymax = 2
horizontal_units = 'km' #m or km
fps = 100000 # frame per sec 
t = np.array(t)
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file
t_day = t/day2s
def update_plot(frame_number, phi_w_sol,phi_i_sol,T_sol, plot,t_day):
    fig.suptitle("t= %0.2f days" % t_day[frame_number], fontsize=22)
    plt.subplot(3,1,1)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        plot = [plt.contourf(Xc/1e3, Yc, np.transpose(phi_w_sol[:,frame_number].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol),vmax=np.max(phi_w_sol),levels=100)]
    else:
        plt.xlim([Grid.xmin, Grid.xmax])    
        plot = [plt.contourf(Xc, Yc, np.transpose(phi_w_sol[:,frame_number].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol),vmax=np.max(phi_w_sol),levels=100)]
 
    plt.ylim([Grid.ymax,Grid.ymin])
    ax1.set_ylabel(r'$z$ [m]')
    
    plt.subplot(3,1,2)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((1-phi_i_sol[:,frame_number]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin=np.min(1-phi_i_sol),vmax=np.max(1-phi_i_sol),levels=100)]
    else:
        plt.xlim([Grid.xmin, Grid.xmax])  
        plot1 = [plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol[:,frame_number]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin=np.min(1-phi_i_sol),vmax=np.max(1-phi_i_sol),levels=100)]
    
    plt.ylim([Grid.ymax,Grid.ymin])
    ax2.set_ylabel(r'$z$ [m]')
    
    plt.subplot(3,1,3)
    if horizontal_units == 'km':
        plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
        ax3.set_xlabel(r'$x$ [km]')
        plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((T_sol[:,frame_number]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol)-Tm,vmax=np.max(T_sol)-Tm,levels=100)]
    
    else:
        plt.xlim([Grid.xmin, Grid.xmax])  
        plot1 = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,frame_number]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol)-Tm,vmax=np.max(T_sol)-Tm,levels=100)]
        ax3.set_xlabel(r'$x$ [m]')
    plt.ylim([Grid.ymax,Grid.ymin])
    ax3.set_ylabel(r'$z$ [m]')

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,figsize=(20,20))
plt.subplot(3,1,1)
ax1.set_ylabel(r'$z$')
if horizontal_units == 'km':
    plot = [plt.contourf(Xc/1e3, Yc, np.transpose(phi_w_sol[:,0].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol),vmax=np.max(phi_w_sol),levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
else:
    plot = [plt.contourf(Xc, Yc, np.transpose(phi_w_sol[:,0].reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=np.min(phi_w_sol),vmax=np.max(phi_w_sol),levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax]) 
plt.ylim([Grid.ymax,Grid.ymin])
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol)
mm.set_clim(np.min(phi_w_sol), np.max(phi_w_sol))
clb = plt.colorbar(mm, orientation='vertical',ax=[ax1],aspect=10,pad=0.01)
clb.set_label(r'$\phi_w$')
ax1.set_ylabel(r'$z$ [m]')

plt.subplot(3,1,2)
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
ax2.set_ylabel(r'$z$')
mm.set_array(1-phi_i_sol)
mm.set_clim(np.max(1-phi_i_sol),np.min(1-phi_i_sol))
clb = plt.colorbar(mm, orientation='vertical',ax=[ax2],aspect=10,pad=0.01)
if horizontal_units == 'km':
    plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((1-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin=np.min(1-phi_i_sol),vmax=np.max(1-phi_i_sol),levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
else:
    plot1 = [plt.contourf(Xc, Yc, np.transpose((1-phi_i_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin=np.min(1-phi_i_sol),vmax=np.max(1-phi_i_sol),levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax]) 
plt.ylim([Grid.ymax,Grid.ymin])
clb.set_label(r'$\varphi$')
ax2.set_ylabel(r'$z$ [m]')

plt.subplot(3,1,3)
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(T_sol-Tm)
mm.set_clim(np.min(T_sol)-Tm, np.max(T_sol)-Tm)
clb = plt.colorbar(mm, orientation='vertical',ax=[ax3],aspect=10,pad=0.01)
clb.set_label(r'T $[^\circ C]$')
if horizontal_units == 'km':
    plot1 = [plt.contourf(Xc/1e3, Yc, np.transpose((T_sol[:,0]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol)-Tm,vmax=np.max(T_sol)-Tm,levels=100)]
    plt.xlim([Grid.xmin/1e3, Grid.xmax/1e3])
    ax3.set_xlabel(r'$x$ [km]')
else:
    plot1 = [plt.contourf(Xc, Yc, np.transpose((T_sol[:,0]-Tm).reshape(Grid.Nx,Grid.Ny)),cmap="Reds",vmin=np.min(T_sol)-Tm,vmax=np.max(T_sol)-Tm,levels=100)]
    plt.xlim([Grid.xmin, Grid.xmax])  
    ax3.set_xlabel(r'$x$ [m]')
plt.ylim([Grid.ymax,Grid.ymin])

ax3.set_ylabel(r'$z$ [m]')

fig.suptitle("t= %0.2f days" % t_day[0], fontsize=22)

#ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,:],phi_i_sol[:,:],T_sol[:,:], plot[:0],t_day[:]), interval=1/fps)

ii = int(tf/(2*day2s))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,::ii],phi_i_sol[:,::ii],T_sol[:,::ii], plot[::ii],t_day[::ii]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_new_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

'''
#New Contour plot combined with new mesh
print('New Contour plot combined with new mesh')
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(phi_w_sol) # frame number of the animation from the saved file

[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)
[X_plate,Y_plate] = comp_face_coords(dof_f_plate_bnd,Grid)

def update_plot(frame_number, zarray, plot,t):
    plt.cla()
    fig.suptitle("t= %0.4f" % t[frame_number], fontsize=22)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    zarray  = zarray/phi 
    ax1.set_label(r'$x$')
    plt.subplot(1,2,1)
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.0005, vmax = 1.0005)
    plt.ylim([Grid.ymax,Grid.ymin])
    plt.subplot(1,2,2)
    zarray[zarray<sat_threshold] = 0
    zarray[zarray>=sat_threshold] = 1
    #plot1[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Greys",vmin = -0.0005, vmax = 1.0005)
    ax2.set_label(r'$x$')
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(Ind)
    mm.set_clim(0., 1.)
    #clb = plt.colorbar(mm, pad=0.05,orientation='horizontal',ax=[ax1,ax2],aspect=50)
    #plt.clim(0,1)
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    ax1.axis('scaled')
    ax2.axis('scaled')
    ax1.set_xlim([Grid.xmin, Grid.xmax])
    ax1.set_ylim([Grid.ymax,Grid.ymin])
    ax2.set_xlim([Grid.xmin, Grid.xmax])
    ax2.set_ylim([Grid.ymax,Grid.ymin])
    ax1.set_xlabel(r'$x$')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$z$')
    bnd = np.ones_like(Grid.xc)
    bnd[Grid.xc < 0.75] = 1 + 3*( Grid.xc[Grid.xc < 0.75] - 0.75 )
    bnd[Grid.xc < 0.5]  = 1 + 3*(-Grid.xc[Grid.xc < 0.5 ] + 0.25 )
    bnd[Grid.xc < 0.25] = 1
    #ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    #ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    #ax2.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    ax1.plot(Xc_col[np.argwhere(zarray[:,frame_number]<0)][:,0],Yc_col[np.argwhere(zarray[:,frame_number]<0)][:,0],'ro-')
    #print(np.shape(zarray[:,frame_number]), np.shape(phi[:,0]))
    ax1.plot(Xc_col[np.argwhere(zarray[:,frame_number]>1)][:,0],Yc_col[np.argwhere(zarray[:,frame_number]>1)][:,0],'gX')  
    ax2.plot(X_all,Y_all,'k-',linewidth=0.4)
    
    dof_inact= Grid.dof[zarray[:,frame_number] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
    if np.any(dof_inact):
        dof_f_saturated = find_faces(dof_inact,D,Grid) 
        dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
        [X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
        [X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
        
        ax2.plot(X_sat,Y_sat,'r-',linewidth=0.4)
        ax2.plot(X_bnd,Y_bnd,'r-',linewidth=2)
        
        ax1.plot(X_plate,Y_plate,'k-',linewidth=2)
        ax2.plot(X_plate,Y_plate,'k-',linewidth=2)
    

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,20)) 
Ind = phi_w_sol[:,0]/phi[:,0]
plt.subplot(1,2,1)
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.0005, vmax = 1.0005)]
ax1.set_ylabel(r'$z$')
ax1.set_label(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
ax1.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
#clb = fig.colorbar(plot[0],orientation='horizontal', orientation='horizontal',aspect=50,pad=0.05)
#clb.set_label(r'$s_w$')
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.set_label(r'$x$')
ax2.set_ylabel(r'$z$')
plt.subplot(1,2,2)
Ind = phi_w_sol[:,0]/phi[:,0]
Ind[Ind<sat_threshold] = 0
Ind[Ind>=sat_threshold] = 1.0
plot1 = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = -0.0005, vmax = 1.0005)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
ax1.axis('scaled')
ax2.axis('scaled')
ax1.set_xlabel(r'$x$')
ax2.set_xlabel(r'$x$')
clb = plt.colorbar(mm, orientation='horizontal',ax=[ax1,ax2],aspect=50,pad=0.13)
clb.set_label(r'$s_w$')
fig.suptitle("t= %0.2f s" % t[0], fontsize=22)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax1.axis('scaled')
ax2.axis('scaled')
bnd = np.ones_like(Grid.xc)
bnd[Grid.xc < 0.75] = 1 + 3*( Grid.xc[Grid.xc < 0.75] - 0.75 )
bnd[Grid.xc < 0.5]  = 1 + 3*(-Grid.xc[Grid.xc < 0.5 ] + 0.25 )
bnd[Grid.xc < 0.25] = 1
#ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
#ax2.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
#ax1.plot(Xc_col[np.argwhere(phi_w_sol[:,0]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,0]<0)[:,0]],'ro-')
#ax1.plot(Xc_col[np.argwhere(phi_w_sol[:,]>phi[:,0]),0][:,0],Yc_col[np.argwhere(phi_w_sol[:,0]>phi[:,0]),0][:,0],'gX') 
ax2.plot(X_all,Y_all,'k-',linewidth=0.4)
ax1.plot(X_plate,Y_plate,'k-',linewidth=2)
ax2.plot(X_plate,Y_plate,'k-',linewidth=2)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_w_sol[:,::10], plot[::10],t[::10]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w_mesh.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

dof_inact= Grid.dof[phi_w_sol[:,-1] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
[X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
[X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.00001, vmax = 1.00001)]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)

plt.colorbar()
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((phi_w/(1-phi_i)).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.00001, vmax = 1.00001)]
#plt.plot(X_plate,Y_plate,'k-',linewidth=2)
plt.colorbar()
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
#plt.axis('scaled')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw_withoutmesh.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
phi_copy = phi.copy()
plot = [plt.contourf(Xc, Yc, np.transpose(phi_copy.reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = np.min(phi_copy), vmax = np.max(phi_copy),levels=1000)]
plt.colorbar()
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
#plt.axis('scaled')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_porosity.pdf',bbox_inches='tight', dpi = 600)

[Xc_flux,Yf_flux] = np.meshgrid(Grid.xc,Grid.yf)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc_flux, Yf_flux, np.transpose(flux_vert[Grid.Nfx:Grid.Nf,-1].reshape(Grid.Nx,Grid.Ny+1)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
#plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_verticalflux.pdf',bbox_inches='tight', dpi = 600)

[Xf_flux,Yc_flux] = np.meshgrid(Grid.xf,Grid.yc)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xf_flux, Yc_flux, np.transpose(flux_vert[0:Grid.Nfx,-1].reshape(Grid.Nx+1,Grid.Ny)),cmap="Blues",vmin = -1, vmax = 1)]
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$x$ [m]')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
#plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(phi_w_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_horizontalflux.pdf',bbox_inches='tight', dpi = 600)


#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()

#phi_w_sol_backup[phi_w_sol_backup>LWC_top] = np.nan
t = np.array(t)
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100)]
shock     = np.genfromtxt('./Colliander_data/Shock.csv', delimiter=',')
#plt.plot(shock[:-4,0],shock[:-4,1],'ro',label='Shock, Colliander et al. (2022)',markersize=10)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol_backup)
#mm.set_clim(0, LWC_top)
clb = plt.colorbar(mm, orientation='vertical',aspect=50,pad=0.13)
clb.set_label(r'$LWC$', labelpad=-40, y=1.1, rotation=0)
plt.xlabel(r'$t$ [days]')
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.clim(0.000000, 1.0000000)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_combined.pdf',bbox_inches='tight', dpi = 600)


'''
#combined
Grid.ymax = 2
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,figsize=(15,15))
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,1)
plt.plot(day,Qnet,'ro',label='Data',markersize=10)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.plot(fit_time,Qnet_func(fit_time),'k-',label='Fit',markersize=10)
plt.plot(fit_time,1.1*Qnet_func(fit_time),'b--',label='Ends',markersize=10)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol_backup)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
clb = plt.colorbar(mm, orientation='vertical',aspect=50,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.1, rotation=0)
plt.ylabel(r'Q$_{net}$ [W/m$^2$]')


plt.subplot(3,1,2)
plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phi_w_sol_backup)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)

plt.subplot(3,1,3)
T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
plot = [plt.contourf(t_array, depth_array, T_array-Tm,cmap="Reds",levels=100)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(T_sol-Tm)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined.pdf',bbox_inches='tight', dpi = 600)

'''

#combined
Grid.ymax = 3
from matplotlib import rcParams
rcParams.update({'font.size': 22})
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True, sharex=True,figsize=(15,15))
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,1)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",levels=100)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(0.28882281204111515,0.5,1000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-40, y=1.18, rotation=0)

plt.subplot(3,1,2)
plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(0.0,0.030341777509989266,1000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)

plt.subplot(3,1,3)
T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
plot = [plt.contourf(t_array, depth_array, T_array-Tm,cmap="Reds",levels=100)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(-10,0,1000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ.pdf',bbox_inches='tight', dpi = 600)



#combined 1D plot

from matplotlib import rcParams
rcParams.update({'font.size': 22})
Grid.ymax = 3
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True, sharex=True,figsize=(15,15),dpi=100)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,1)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",vmin=np.min(1-phi_i_sol),vmax=np.max(1-phi_i_sol),ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
#mm.set_array(np.linspace(0.28882281204111515,0.5,10000))
mm.set_array(1-phi_i_sol)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-40, y=1.18, rotation=0)

plt.subplot(3,1,2)
plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100,vmin=np.min(phi_w_sol),vmax=np.max(phi_w_sol),ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
#mm.set_array(np.linspace(0.0,0.030341777509989266,10000))
mm.set_array(phi_w_sol_backup)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)

plt.subplot(3,1,3)
T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
plot = [plt.contourf(t_array, depth_array, T_array-Tm,cmap="Reds",levels=100,vmin=np.min(T_array-Tm),vmax=np.max(T_array-Tm),ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(-10,0,10000))
mm.set_array(T_sol-Tm)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout(w_pad=0.5, h_pad=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ.pdf',bbox_inches='tight', dpi = 600)

#combined 1D plot with Sw
Grid.ymax = 3
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True, sharex=True,figsize=(15,15),dpi=100)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))
t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,1)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",levels=100)]#, vmin = 0.28882281204111515,vmax = 0.5)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
#mm.set_array(1-phi_i_sol)
#mm.set_array(np.linspace(0.28882281204111515,0.5,1000))
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-40, y=1.18, rotation=0)

plt.subplot(3,1,2)
plot = [plt.contourf(t_array, depth_array, phi_w_array/(1-phi_i_array),cmap="Blues",levels=100)]#, vmin=0.0,vmax=0.06544608643314336)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
#mm.set_array(np.linspace(0.0,0.06544608643314336,1000))
#mm.set_array(phi_w_array/(1-phi_i_array))
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$S_w$', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)

plt.subplot(3,1,3)
T_array = T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
plot = [plt.contourf(t_array, depth_array, T_array-Tm,cmap="Reds",levels=100)]#, vmin = -10,vmax = 0)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
#mm.set_array(np.linspace(-10,0,1000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ_WithSw.pdf',bbox_inches='tight', dpi = 600)


'''
#A fun combined  phi, S, T
tday = t/day2s
ymax_loc =5
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(T_sol) # frame number of the animation from the saved file
[Xy,Yf] = np.meshgrid(Grid.xf,Grid.yf)
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
ax3.set_ylim([ymax_loc,Grid.ymin])
ax3.set_xlabel(r'$T[C]$')

#ax1.set_xlim(np.min([1-phi_i_sol]),np.max(1-phi_i_sol))
ax1.set_xlim(np.min([1-phi_i_sol[~np.isnan(phi_i_sol)]]),np.max(1-phi_i_sol[~np.isnan(phi_i_sol)]))
ax1.set_ylim([ymax_loc,Grid.ymin])
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$y[m]$')


#ax2.set_xlim([np.min(phi_w_sol/(1-phi_i_sol)),np.max(phi_w_sol/(1-phi_i_sol))])
ax2.set_xlim([0,0.1])
ax2.set_ylim([ymax_loc,Grid.ymin])
ax2.set_xlabel(r'$S_w$')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

x = Yc[:,1]
y = np.zeros_like(x)
z = np.zeros_like(x)
a = np.zeros_like(Yf[:,1])
face = Yf[:,1]

line1, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-')
line2, = ax2.plot(z, x, lw=2, color= 'b')
line3, = ax3.plot(y, x, lw=2, color= 'b',linestyle='--')
line4, = ax3.plot(z, x, lw=2, color= 'r')
ax2.set_title("Time: %0.2f days" %t[0],loc = 'center', fontsize=18)
plt.tight_layout()
line   = [line1, line2, line3, line4]

# initialization function: plot the background of each frame
def init():
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
    line[3].set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i,u_sol,Sw_sol,phi_w_sol,phi_i_sol,Yc,Yf,time):
    x = Yc[:,1] 
    Sw = np.transpose((phi_w_sol[:,i]/phi_i_sol[:,i]).reshape(Grid.Nx,Grid.Ny))[:,1]
    y = np.transpose(u_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]-273.16    
    z = np.transpose(phi_w_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz= 1-np.transpose(phi_i_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]                           
    Tmelt = 0.0*Yc[:,1]
    line[0].set_data( zz, x)
    line[1].set_data(Sw, x)
    line[2].set_data(Tmelt, x)
    line[3].set_data(y, x)
    #plt.tight_layout()
    ax2.set_title("Time: %0.2f days" %time[i],loc = 'center', fontsize=18)
    #ax1.legend(loc='lower right', shadow=False, fontsize='medium')
    #ax2.legend(loc='best', shadow=False, fontsize='medium')
    ax3.set_xlim([-30,0])#np.max(T_sol)-273.16])
    ax1.set_xlim(np.min([1-phi_i_sol[~np.isnan(phi_i_sol)]]),np.max(1-phi_i_sol[~np.isnan(phi_i_sol)]))
    ax2.set_xlim([0,0.1])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
ii=1
ani = animation.FuncAnimation(fig, animate, frn, init_func=init, fargs=(T_sol[:,::ii],s_w_sol[:,::ii],phi_w_sol[:,::ii],phi_i_sol[:,::ii],Yc[:,:],Yf[:,:],tday[::ii])
                               , interval=1/fps)

ani.save(f"../Figures/{simulation_name}_combined.mov", writer='ffmpeg', fps=30)



# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The codec argument ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, codec='libx264')
'''



'''
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



'''
#A fun combined  phi, S, T image
ii=1
tday = t/day2s
ymax_loc =5
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(T_sol) # frame number of the animation from the saved file
[Xy,Yf] = np.meshgrid(Grid.xf,Grid.yf)
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax3.set_xlim([np.min(T_sol)-273.16,np.max(T_sol)-273.16])
ax3.set_ylim([ymax_loc,Grid.ymin])
ax3.set_xlabel(r'$T[C]$')

ax1.set_xlim(np.min([1-phi_i_sol]),np.max(1-phi_i_sol))
ax1.set_ylim([ymax_loc,Grid.ymin])
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$y[m]$')


ax2.set_xlim([np.min(phi_w_sol/phi_i_sol),np.max(phi_w_sol/phi_i_sol)])
ax2.set_ylim([ymax_loc,Grid.ymin])
ax2.set_xlabel(r'$S_w$')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

x = Yc[:,1]
y = np.zeros_like(x)
z = np.zeros_like(x)
a = np.zeros_like(Yf[:,1])
face = Yf[:,1]

line1, = ax1.plot(y, x, lw=2, color= 'k',linestyle='-')
line2, = ax2.plot(z, x, lw=2, color= 'b')
line3, = ax3.plot(y, x, lw=2, color= 'b',linestyle='--')
line4, = ax3.plot(z, x, lw=2, color= 'r')
plt.tight_layout()
line   = [line1, line2, line3, line4]

# initialization function: plot the background of each frame
line[0].set_data([], [])
line[1].set_data([], [])
line[2].set_data([], [])
line[3].set_data([], [])

u_sol = T_sol[:,::ii]
Sw_sol = s_w_sol

Ind_array = [0,60,96,108,120,240,480]

for i in Ind_array:
    print(i,tday[i])
    x = Yc[:,1] 
    Sw = np.transpose((phi_w_sol[:,i]/phi_i_sol[:,i]).reshape(Grid.Nx,Grid.Ny))[:,1]
    y = np.transpose(u_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]-273.16    
    z = np.transpose(phi_w_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1] 
    zz= 1-np.transpose(phi_i_sol[:,i].reshape(Grid.Nx,Grid.Ny))[:,1]                           
    Tmelt = 0.0*Yc[:,1]
    line[0].set_data(zz, x)
    line[1].set_data(Sw, x)
    line[2].set_data(Tmelt, x)
    line[3].set_data(y, x)
    
    transparency =0.05+0.95*i/len(t_interest)
    
    line1, = ax1.plot(zz, x, lw=2, color= 'k',linestyle='-', alpha=transparency)
    line2, = ax2.plot(Sw, x, lw=2, color= 'b', alpha=transparency,label="%0.1f days" %tday[i])
    line3, = ax3.plot(Tmelt, x, lw=2, color= 'b',linestyle='--')
    line4, = ax3.plot(y, x, lw=2, color= 'r', alpha=transparency)
    
    
    plt.tight_layout()
    #ax1.legend(loc='lower right', shadow=False, fontsize='medium')
    ax2.legend(loc='best', shadow=False, fontsize='medium',frameon=False)

plt.savefig(f'../Figures/{simulation_name}_combined.pdf',bbox_inches='tight', dpi = 600)
'''

phi_sol = 1- phi_i_sol

'''

#combined 1D plot
phi_sol = 1- phi_i_sol
from matplotlib import rcParams
rcParams.update({'font.size': 22})
T_array[T_array>Tm] = Tm
Grid.ymax = 2.5
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True, sharex=True,figsize=(15,15),dpi=100)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan

#####

t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(3,1,1)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(1-np.max(phi_i_array[~np.isnan(phi_i_array)]),1-np.min(phi_i_array[~np.isnan(phi_i_array)]),10000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-40, y=1.18, rotation=0)

plt.subplot(3,1,2)

plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)

plt.subplot(3,1,3)
plot = [plt.contourf(t_array, depth_array, T_array,cmap="Reds",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'Depth [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ.pdf',bbox_inches='tight', dpi = 100)



fig = plt.figure(figsize=(5,10) , dpi=100)
plt.plot(1-phi_i_sol[0:Grid.Ny,0],Grid.yc,'r',label=f'Initial',linewidth=2)
plt.plot(1-phi_i_sol[0:Grid.Ny,-1],Grid.yc,'k--',label=f'Final',linewidth=2)
plt.ylabel(r'$z$ [m]')
plt.xlabel(r'$\phi$')
plt.legend(fancybox=True, framealpha=0.5,loc='lower right')
plt.xlim([0,1])
plt.ylim([Grid.ymax,Grid.ymin])
plt.tight_layout()
plt.savefig(f'../Figures/{simulation_name}_combined_phi_variation.pdf',bbox_inches='tight', dpi = 600)



#combined 1D plot with upGPR
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
 
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))


#Make actual time array
t_dates = []
for i in range(0,len(t)):
    t_dates.append(dates[850] + timedelta(seconds = t[i]))
    

from matplotlib import rcParams
rcParams.update({'font.size': 22})
T_array[T_array>Tm] = Tm
Grid.ymax = 2.5
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True,figsize=(15,15),dpi=100)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan
phi_w_sol[np.isnan(phi_i_sol)] = np.nan
#####

t_array=np.kron(tday,np.ones((Grid.Ny,1)))
phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
phi_i_array = phi_i_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
T_array =         T_sol[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]

plt.subplot(4,1,1)
plt.plot(day,Qnet,'r-',label='Data',markersize=10)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
plt.xlim([tday[0],tday[-1]])

plt.subplot(4,1,2)
plot = [plt.contourf(t_array, depth_array, (1-phi_i_array),cmap="Greys",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
mm.set_array(np.linspace(1-np.max(phi_i_array[~np.isnan(phi_i_array)]),1-np.min(phi_i_array[~np.isnan(phi_i_array)]),10000))
#mm.set_array(1-phi_i_sol)
plt.ylabel(r'z [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$\phi$', labelpad=-40, y=1.18, rotation=0)

plt.subplot(4,1,3)

plot = [plt.contourf(t_array, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'z [m]')
plt.xlim([tday[0],tday[-1]])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(samira_heilig_time_days,data['height'][850] - data['height'][850:4200],'k-')
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-40, y=1.18, rotation=0)
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
clb.set_label(r'T[$^\circ$C]', labelpad=-40, y=1.18, rotation=0)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Time [days]')

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ.pdf',bbox_inches='tight', dpi = 100)



######################################################################
#combined 1D plot with upGPR with actual dates
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
 
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))

from matplotlib import rcParams
rcParams.update({'font.size': 22})
T_array[T_array>Tm] = Tm
Grid.ymax = 2.6
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True,figsize=(15,15),dpi=50)
#Neww plot sidebysidesnapshot
phi_w_sol_backup = phi_w_sol.copy()
tday = t/day2s
depth_array=np.kron(np.ones(len(tday)),np.transpose([Grid.yc]))


######
#Remove 99% melted ice
phi_w_sol_backup[phi_sol>1-non_porous_vol_frac] = np.nan
phi_i_sol[phi_sol>1- non_porous_vol_frac] = np.nan
T_sol[phi_sol>1-non_porous_vol_frac] = np.nan
T_sol[np.isnan(phi_i_sol)] = np.nan
phi_w_sol_backup[np.isnan(phi_i_sol)] = np.nan

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


plt.subplot(4,1,1)
plt.plot(([dates[850] + timedelta(days = i) for i in day]),Qnet,'r-',label='Data',linewidth=2)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)



import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees

plt.subplot(4,1,2)
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

plt.subplot(4,1,3)

plot = [plt.contourf(t_dates, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(dates[850:4200],data['height'][850] - data['height'][850:4200],'k-')
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-100, y=0.5, rotation=90)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees


plt.subplot(4,1,4)
plot = [plt.contourf(t_dates, depth_array, T_array,cmap="Reds",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)
plt.xlabel(r'Calendar date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withoutQ_actual_dates.pdf',bbox_inches='tight', dpi = 100)
'''

'''
######################################################################
#combined 1D plot with upGPR with actual dates and Temperatures
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
 
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))

from matplotlib import rcParams
rcParams.update({'font.size': 22})
Grid.ymax = 2.6
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex=True,figsize=(15,15),dpi=50)
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

plt.subplot(6,1,1)
plt.plot(([dates[850] + timedelta(days = i) for i in day]),Qnet,'r-',label='Data',linewidth=2)
fit_time = np.linspace(tday[0],tday[-1],1000)
plt.ylim([1.1*np.min(Qnet_func(fit_time)),1.1*np.max(Qnet_func(fit_time))])
plt.ylabel(r'Q$_{net}$ [W/m$^2$]',color='red')
mm = plt.cm.ScalarMappable(cmap=cm.Greys)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)

import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax1.xaxis.grid(True, which='major', color='k', linestyle='--')
ax1.spines['left'].set_color('red')
ax1.tick_params(axis='y', colors='red')

ax1new = ax1.twinx()

color = 'tab:blue'
ax1new.spines['right'].set_color('blue')
ax1new.set_ylabel(r'$a$ [m.w.eq/yr]', color='blue')  # we already handled the x-label with ax1
ax1new.plot(([dates[850] + timedelta(days = i) for i in day]), RetMIP_acc_rate(day)*yr2s , color='blue')
ax1new.tick_params(axis='y', colors='blue')

plt.subplot(6,1,2)
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

plt.subplot(6,1,3)
phi_w_array[phi_w_array>0.06] = 0.06
plot = [plt.contourf(t_dates, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(phi_w_array[~np.isnan(phi_w_array)]),np.max(phi_w_array[~np.isnan(phi_w_array)]),10000))
#mm.set_array(phi_w_sol_backup)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(dates[850:4200],data['height'][850] - data['height'][850:4200],'k-')
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'$LWC$', labelpad=-100, y=0.5, rotation=90)
ax3.xaxis.grid(True, which='major', color='k', linestyle='--')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees


plt.subplot(6,1,4)
plot = [plt.contourf(t_dates, depth_array, T_array,cmap="Reds",levels=100,ls=None)]
mm = plt.cm.ScalarMappable(cmap=cm.Reds)
mm.set_array(np.linspace(np.min(T_array[~np.isnan(T_array)]-Tm),np.max(T_array[~np.isnan(T_array)]-Tm),10000))
#mm.set_array(T_sol-Tm)
plt.ylabel(r'z [m]')
plt.ylim([Grid.ymax,Grid.ymin])
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
clb.set_label(r'T[$^\circ$C]', labelpad=-100, y=0.5, rotation=90)
#plt.clim(0.000000, 1.0000000)
ax4.xaxis.grid(True, which='major', color='k', linestyle='--')

plt.subplot(6,1,5)
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.1'])[Samira_dates_index[0]],'-',label='0.3 m',color = red_array[0,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.2'])[Samira_dates_index[0]],'-',label='0.6 m',color = red_array[1,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.3'])[Samira_dates_index[0]],'-',label='0.9 m',color = red_array[2,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.4'])[Samira_dates_index[0]],'-',label='1.4 m',color = red_array[3,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.5'])[Samira_dates_index[0]],'-',label='1.8 m',color = red_array[4,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.6'])[Samira_dates_index[0]],'-',label='2.1 m',color = red_array[5,:])

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in Samira_siteA_depth_array:   
    index = np.where(Yc_col<i)[0][-1]
    plt.plot(t_dates[0,:],(T_sol[index,:]+T_sol[index+1,:])/2-273.16,'k-',alpha = alpha_array[np.argwhere(i==Samira_siteA_depth_array)][0,0])
#plt.legend(fontsize='medium', ncol=2)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
plt.ylabel(r'T[$^\circ$C]')
ax5.xaxis.grid(True, which='major', color='k', linestyle='--')

plt.subplot(6,1,6)
data_LWC  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/samimi_LWC.csv')

data_LWC[np.isnan(data_LWC)] = -10

plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.3m']]),data_LWC['0.3m.1']-data_LWC['0.3m.1'][0],'bo',label='0.3 m',  color = blue_array[0,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.6m']]),data_LWC['0.6m.1']-data_LWC['0.6m.1'][0],'bs',label='0.6 m',color = blue_array[1,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.9m']]),data_LWC['0.9m.1']-data_LWC['0.9m.1'][0],'bd',label='0.9 m',color = blue_array[2,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['1.4m']]),data_LWC['1.4m.1']-data_LWC['1.4m.1'][0],'b^',label='1.4 m',color = blue_array[3,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['1.8m']]),data_LWC['1.8m.1']-data_LWC['1.8m.1'][0],'b>',label='1.8 m',color = blue_array[4,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['2.1m']]),data_LWC['2.1m.1']-data_LWC['2.1m.1'][0],'b<',label='2.1 m',color = blue_array[5,:], markersize=3)

plt.ylabel(r'LWC')

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in range(0,len(Samira_siteA_depth_array)):   
    index = np.where(Yc_col<Samira_siteA_depth_array[i])[0][-1]
    plt.plot(t_dates[0,:],(phi_w_sol[index,:]+phi_w_sol[index+1,:])/2,'-',color = black_array[i,:])

plt.ylim(0,0.0275)
plt.xlim()
#plt.legend( ncol=2)
clb = plt.colorbar(mm, orientation='vertical',aspect=10,pad=0.05)
ax6.xaxis.grid(True, which='major', color='k', linestyle='--')
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
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_super_combined_withQAcc_actual_dateswithT.pdf',bbox_inches='tight', dpi = 50)

'''

'''
######################################################################
#combined 1D plot with upGPR with actual dates and Temperatures + LWC combined
from datetime import datetime 
from datetime import timedelta
import pandas as pd 
phi_sol = 1 - phi_i_sol
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx',header=(1))
Samira_dates = df['Time']
Samira_dates_index = np.where((Samira_dates>=dates[850]) & (Samira_dates<=dates[len(dates)-1]))

red_array  = np.array([[255,0,0], [255,43,43], [255,85,85], [255,128,128] , [255,170,170], [255,213,213]])/255;
black_array= np.array([[0,0,0], [43,43,43], [85,85,85], [128,128,128] , [170,170,170], [213,213,213]])/255; 
blue_array = np.array([[0,0,255], [43,43,255], [85,85,255], [128,128,255] , [170,170,255], [213,213,255]])/255;

data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016

# format the string in the given format : day/month/year 
# hours/minutes/seconds-micro seconds
#format_data = "'%d-%b-%Y %H:%M:%S.%f'"
#for i in data['date']:
#    print(datetime.strptime(i, format_data))

from matplotlib import rcParams
rcParams.update({'font.size': 22})
Grid.ymax = 2.6
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
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.2'])[Samira_dates_index[0]],'-',label='0.6 m',color = black_array[1,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.3'])[Samira_dates_index[0]],'-',label='0.9 m',color = black_array[2,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.4'])[Samira_dates_index[0]],'-',label='1.4 m',color = black_array[3,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.5'])[Samira_dates_index[0]],'-',label='1.8 m',color = black_array[4,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.6'])[Samira_dates_index[0]],'-',label='2.1 m',color = black_array[5,:])

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in Samira_siteA_depth_array:   
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
phi_w_array[phi_w_array>0.06] = 0.06
phi_w_array[T_array < Tm] = np.nan
new2 = ax3.contourf(t_dates, depth_array, phi_w_array,cmap="Blues",levels=100,ls=None)

plt.plot(dates[850:4200],data['height'][850] - data['height'][850:4200],'k-')

plt.subplot(5,1,5)

phi_w_array = phi_w_sol_backup[int(Grid.Nx*Grid.Ny/2+0):int(Grid.Nx*Grid.Ny/2+Grid.Ny),:]
#phi_w_array[phi_w_array>0.06] = 0.06

data_LWC  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/samimi_LWC.csv')

data_LWC[np.isnan(data_LWC)] = -10

plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.3m']]),data_LWC['0.3m.1']-data_LWC['0.3m.1'][0],'ko',label='0.3 m',  color = black_array[0,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.6m']]),data_LWC['0.6m.1']-data_LWC['0.6m.1'][0],'ks',label='0.6 m',color = black_array[1,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['0.9m']]),data_LWC['0.9m.1']-data_LWC['0.9m.1'][0],'kd',label='0.9 m',color = black_array[2,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.4m']]),data_LWC['1.4m.1']-data_LWC['1.4m.1'][0],'k^',label='1.4 m',color = black_array[3,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['1.8m']]),data_LWC['1.8m.1']-data_LWC['1.8m.1'][0],'k>',label='1.8 m',color = black_array[4,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i-3.5) for i in data_LWC['2.1m']]),data_LWC['2.1m.1']-data_LWC['2.1m.1'][0],'k<',label='2.1 m',color = black_array[5,:], markersize=3)

plt.ylabel(r'LWC')

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in range(0,len(Samira_siteA_depth_array)):   
    index = np.where(Yc_col<Samira_siteA_depth_array[i])[0][-1]
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




#Importing data from temperature
import pandas as pd
from datetime import datetime
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
samira_heilig_time_days =np.array([(dates[i] - dates[850]).total_seconds() for i in range(850,4200)])/day2s   #time in days May 24 through September 18, 2016


#Site A: 0.3, 0.6, 0.9, 1.4, 1.8, 2.1, 2.8, 3.7
df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx',header=(1))
Samira_dates = df['Time']
Samira_dates_index = np.where((Samira_dates>=dates[850]) & (Samira_dates<=dates[len(dates)-1]))

red_array  = np.array([[255,0,0], [255,43,43], [255,85,85], [255,128,128] , [255,170,170], [255,213,213]])/255;
black_array= np.array([[0,0,0], [43,43,43], [85,85,85], [128,128,128] , [170,170,170], [213,213,213]])/255; 
blue_array = np.array([[0,0,255], [43,43,255], [85,85,255], [128,128,255] , [170,170,255], [213,213,255]])/255;

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True,figsize=(15,15),dpi=100)
plt.subplot(2,1,1)
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.1'])[Samira_dates_index[0]],'-',label='0.3 m',color = red_array[0,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.2'])[Samira_dates_index[0]],'-',label='0.6 m',color = red_array[1,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.3'])[Samira_dates_index[0]],'-',label='0.9 m',color = red_array[2,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.4'])[Samira_dates_index[0]],'-',label='1.4 m',color = red_array[3,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.5'])[Samira_dates_index[0]],'-',label='1.8 m',color = red_array[4,:])
plt.plot(Samira_dates[Samira_dates_index[0]],(df['degC.6'])[Samira_dates_index[0]],'-',label='2.1 m',color = red_array[5,:])

Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in Samira_siteA_depth_array:   
    index = np.where(Yc_col<i)[0][-1]
    plt.plot(t_dates[0,:],(T_sol[index,:]+T_sol[index+1,:])/2-273.16,'k-',alpha = alpha_array[np.argwhere(i==Samira_siteA_depth_array)][0,0])
plt.legend(fontsize='medium', ncol=2)

plt.ylabel(r'T[$^\circ$C]')

plt.subplot(2,1,2)
data_LWC  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/samimi_LWC.csv')

data_LWC[np.isnan(data_LWC)] = -10

plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.3m']]),data_LWC['0.3m.1']-data_LWC['0.3m.1'][0],'bo',label='0.3 m',  color = blue_array[0,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.6m']]),data_LWC['0.6m.1']-data_LWC['0.6m.1'][0],'bs',label='0.6 m',color = blue_array[1,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['0.9m']]),data_LWC['0.9m.1']-data_LWC['0.9m.1'][0],'bd',label='0.9 m',color = blue_array[2,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['1.4m']]),data_LWC['1.4m.1']-data_LWC['1.4m.1'][0],'b^',label='1.4 m',color = blue_array[3,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['1.8m']]),data_LWC['1.8m.1']-data_LWC['1.8m.1'][0],'b>',label='1.8 m',color = blue_array[4,:], markersize=3)
plt.plot(([dates[850] + timedelta(days = i) for i in data_LWC['2.1m']]),data_LWC['2.1m.1']-data_LWC['2.1m.1'][0],'b<',label='2.1 m',color = blue_array[5,:], markersize=3)

plt.ylabel(r'LWC')

'''
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%'])[Samira_dates_index[0]],0),'bo',label='z=0.3 m',  color = blue_array[0,:], markersize=3)
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%.1'])[Samira_dates_index[0]],0),'bs',label='z=0.6 m',color = blue_array[1,:], markersize=3)
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%.2'])[Samira_dates_index[0]],0),'bd',label='z=0.9 m',color = blue_array[2,:], markersize=3)
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%.3'])[Samira_dates_index[0]],0),'b^',label='z=1.4 m',color = blue_array[3,:], markersize=3)
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%.4'])[Samira_dates_index[0]],0),'b>',label='z=1.8 m',color = blue_array[4,:], markersize=3)
plt.plot(Samira_dates[Samira_dates_index[0]],np.maximum((df['%.5'])[Samira_dates_index[0]],0),'b<',label='z=2.1 m',color = blue_array[5,:], markersize=3)
'''
Samira_siteA_depth_array = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1])
alpha_array = np.array([1.0, 0.75,0.6,0.4,0.2,0.1])
for i in range(0,len(Samira_siteA_depth_array)):   
    index = np.where(Yc_col<Samira_siteA_depth_array[i])[0][-1]
    plt.plot(t_dates[0,:],(phi_w_sol[index,:]+phi_w_sol[index+1,:])/2,'-',color = black_array[i,:])

plt.ylim(0,0.025)
plt.legend(fontsize='medium', ncol=2)


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
#yyyymmdd format
chosen_dates = [20160529,20160623,20160718,20160812,20160906]
chosen_dates = [datetime.strptime(str(int(dateee)),'%Y%m%d') for dateee in chosen_dates]

plt.xticks(chosen_dates,rotation = 0) # Rotates X-Axis Ticks by 45-degrees
plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_rhow{rho_w}_LWC_actual_dateswithT.pdf',bbox_inches='tight', dpi = 100)
  