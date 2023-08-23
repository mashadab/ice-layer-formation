######################################################################
#Figure 2 - Regime diagram analyzed
#Mohammad Afzal Shadab
#Date modified: 05/03/2022
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 25})


col_red    = np.array([190,30,45])/255
col_blue   = np.array([ 39,170,225])/255; 
col_green  = np.array([  0,166,81])/255;
col_orange = np.array([247,148,30])/255;
col_purple = np.array([102,45,145])/255;
col_brown  = np.array([117, 76,36])/255;
col_tan    = np.array([199,178,153])/255;


##### loglog function

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

target_func = func_powerlaw

fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)

######################################################################
#LWC 0.03
######################################################################
porosity_drop11   = [0.21117718795888407,0.23937429919338105,0.15309057248259195, 0.21478788881673605, 0.2428220808786894,0.1998938509587419,0.21628571309201694,0.24651734064228037,0.21169315549476986,0.1731761137361466,0.22112574627490744]
pene_depth11      = [2.0625,1.2374999999999998,4.0375,2.7624999999999997, 1.5875,1.8375,1.3624999999999998,1.5124999999999997,2.2875,3.4625,2.5875]
T_firn11          = np.array([-10,-15,-5,-7.5,-12.5,-11,-14,-13,-9,-6,-8])

plot = plt.plot(np.log(np.abs(-T_firn11)),np.log(pene_depth11),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.03')

z11 = np.polyfit(np.log(np.abs(-T_firn11)),np.log(pene_depth11), 1)
T_firn_syn11 = np.log(np.linspace(np.min(-T_firn11), np.max(-T_firn11),1000))
plt.plot(T_firn_syn11, z11[0] * T_firn_syn11 + z11[1],'r--',color=col_red)
print(z11)

######################################################################
#LWC 0.04
######################################################################
porosity_drop22   = [0.2874013652842401, 0.35397675000873585, 0.19277217625488152, 0.2613433159984344, 0.29367159828135225, 0.3177815834605371, 0.3380628301465024, 0.312852457512136, 0.2512281765724733,0.23274098362772655, 0.2579781441164293]
pene_depth22      = [3.8625000000000003, 2.5375000000000005,  7.737500000000001, 5.1375, 3.0875000000000004, 3.5125, 2.7375000000000003, 2.9625000000000004, 4.3125,6.4625, 4.8375]
T_firn22          = np.array([-10,-15,-5,-7.5,-12.5,-11,-14,-13,-9,-6,-8])

plot = plt.plot(np.log(np.abs(-T_firn22)),np.log(pene_depth22),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.04')

z22 = np.polyfit(np.log(np.abs(-T_firn22)),np.log(pene_depth22), 1)
T_firn_syn22 = np.log(np.linspace(np.min(-T_firn22), np.max(-T_firn22),1000))
plt.plot(T_firn_syn22, z22[0] * T_firn_syn22 + z22[1],'g--',color=col_green)
print(z22)

######################################################################
#LWC 0.05
######################################################################
porosity_drop33   = [0.298531657095896 , 0.3768719715066855, 0.059969876974321656, 0.28744362498539044, 0.3670061812919956 ,0.3670061812919956,                        0.38909156189190863,     0.3811919314600054,       0.30354245162672977, 0.23723633442420833, 0.29577387387393195 ]
pene_depth33      = [6.262500000000001 , 4.1375,             12.4375,               8.362499999999999, 4.9875,      5.687500000000001, 4.4375,                         4.7875000000000005,      6.9625,                  10.4625, 7.8375 ]
T_firn33          = np.array([-10,-15,-5,-7.5,-12.5,-11,-14,-13,-9,-6,-8])

plot = plt.plot(np.log(np.abs(-T_firn33)),np.log(pene_depth33),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.05')

z33 = np.polyfit(np.log(np.abs(-T_firn33)),np.log(pene_depth33), 1)
T_firn_syn33 = np.log(np.linspace(np.min(-T_firn33), np.max(-T_firn33),1000))
plt.plot(T_firn_syn33, z33[0] * T_firn_syn33 + z33[1],'b--',color=col_blue)
print(z33)

######################################################################
#LWC 0.06
######################################################################
porosity_drop44   = [0.3761469494487789,0.41129845458718883, 0.1070833271700744,0.30091200474092683, 0.4045887821188816, 0.3880534553361251, 0.4001309364399459, 0.35764604332941086, 0.31862138680069074, 0.2983588915296106, 0.2786545684314431]
pene_depth44      = [9.237499999999999,6.112500000000001,18.0875, 12.3625, 7.362500000000001, 8.3875, 6.562500000000001, 7.0875, 10.2875, 15.4625, 11.5875]
T_firn44          = np.array([-10,-15,-5,-7.5,-12.5,-11,-14,-13,-9,-6,-8])

plot = plt.plot(np.log(np.abs(-T_firn44)),np.log(pene_depth44),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.06')

z44 = np.polyfit(np.log(np.abs(-T_firn44)),np.log(pene_depth44), 1)
T_firn_syn44 = np.log(np.linspace(np.min(-T_firn44), np.max(-T_firn44),1000))
plt.plot(T_firn_syn44, z44[0] * T_firn_syn44 + z44[1],'k--')
print(z44)

######################################################################
#LWC 0.07
######################################################################
porosity_drop55   = [0.20568006415283946, 0.39634452843405354, 0.43728458523539926, 0.325464011682832, 0.39366606345021415, 0.3607223360279611, 0.4273213458559417, 0.3914314030260778, 0.3658972299983648, 0.3017444261187623, 0.34689150912138744]
pene_depth55      = [25.7375, 12.8375      , 8.487499999999999, 17.1125, 10.2375, 11.6625, 9.112499999999999, 9.8375, 14.2875,21.4875, 16.0875]
T_firn55          = np.array([-5, -10               ,-15,-7.5,-12.5,-11,-14,-13,-9,-6,-8])
plot = plt.plot(np.log(np.abs(-T_firn55)),np.log(pene_depth55),'o',markeredgecolor=[0.5,0.5,0.5],markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.07')

z55 = np.polyfit(np.log(np.abs(-T_firn55)),np.log(pene_depth55), 1)
T_firn_syn55 = np.log(np.linspace(np.min(-T_firn55), np.max(-T_firn55),1000))
plt.plot(T_firn_syn55, z55[0] * T_firn_syn55 + z55[1],'k--', markeredgecolor=[0.5,0.5,0.5])
print(z55)
plt.ylabel(r'log(Penetration depth) [log m]')
plt.legend(loc='best')
plt.xlabel(r'$log|T_{0}|$ [log C]')
plt.ylim([np.log(20),0])
plt.savefig(f'../Figures/T_firnvspene_loglog.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(12,8) , dpi=100)
AAA = np.array([[-1.07597175 ,3.17091429],
[-1.01069191 , 3.67657848],
[-1.00596788 , 4.14790244],
[-0.9979044  , 4.51860119],
[-0.94196477 , 4.70811631]])
LWC_array = [0.03, 0.04, 0.05, 0.06, 0.07]
z_data    = np.polyfit(LWC_array, AAA[:,1], 2)
z_func    = np.poly1d(np.polyfit(LWC_array, AAA[:,1], 2))
plt.ylabel(r'$c$')
plt.xlabel(r'LWC')
plt.plot(LWC_array,AAA[:,1],'ro',markeredgecolor=col_red)
plt.plot(np.linspace(0.02,0.08,100),z_func(np.linspace(0.02,0.08,100)),'k--')
plt.savefig(f'../Figures/c_vs_LWC.pdf',bbox_inches='tight', dpi = 600)


######################################################################
# Supplementary Figure
######################################################################
fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)

######################################################################
#LWC 0.03
#T_firn = -30 C
#porosity = 0.4
######################################################################
npp             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop   = [0.09062271723696291, 0.12374952963537411, 0.1626440933242712, 0.23852054940751444, 0.24973241450836525, 0.29839798158644537, 0.27763601544191796, 0.3215083767431941, 0.33145438832928376]
pene_depth      = [0.2625, 0.5125, 1.1125, 1.7625, 2.4125, 3.0625, 3.7125, 4.3875, 5.062500000000001]

plot = plt.plot((npp),(pene_depth),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')
npp_syn = np.log(np.linspace(np.min(npp), np.max(npp),1000))

######################################################################
#T_firn = -30 C
#porosity = 0.5
######################################################################
npp             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop   = [0.12922863664210815, 0.16918960543770367, 0.2111771931499642, 0.22427603762831938, 0.28392571820402046, 0.31861976045493245, 0.34058411615497775, 0.3183586683266787, 0.3529811698448252]
pene_depth      = [0.4875, 0.9875, 2.0625, 3.1375, 4.2625, 5.4125, 6.5625, 7.7125, 8.8875]


plot = plt.plot((npp),(pene_depth),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')

######################################################################
#T_firn = -30 C
#porosity = 0.6
######################################################################
npp             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop   = [0.14329550105856603, 0.18140313652936801,0.234550610092216, 0.22623988229547476, 0.23189760717340435, 0.2766599775408536, 0.2916552035366988, 0.3133280319148, 0.28317879417464187]
pene_depth      = [0.7875, 1.5875,3.2625, 4.9875, 6.7625, 8.5875, 10.4125, 12.2375, 14.1125]


plot = plt.plot((npp),(pene_depth),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


plt.ylim([20,10**(-1.5)])
plt.ylabel(r'Penetration depth, $z_{p}$ [m]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')

plt.savefig(f'../Figures/fig2gh_nppvspene.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)

######################################################################
#LWC 0.03
#T_firn = -30 C
#porosity = 0.4
######################################################################
npp1             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop1   = [0.09062271723696291, 0.12374952963537411, 0.1626440933242712, 0.23852054940751444, 0.24973241450836525, 0.29839798158644537, 0.27763601544191796, 0.3215083767431941, 0.33145438832928376]
pene_depth1      = [0.2625, 0.5125, 1.1125, 1.7625, 2.4125, 3.0625, 3.7125, 4.3875, 5.062500000000001]

plot1 = plt.plot(np.log(npp1),np.log(pene_depth1),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')
npp_syn1 = np.log(np.linspace(np.min(npp), np.max(npp),1000))

z1 = np.polyfit(np.log(npp1),np.log(pene_depth1), 1)

plt.plot(npp_syn1, z1[0] * npp_syn1 + z1[1],'r--',color=col_red)
print(z1)

######################################################################
#T_firn = -30 C
#porosity = 0.5
######################################################################
npp2             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop2   = [0.12922863664210815, 0.16918960543770367, 0.2111771931499642, 0.22427603762831938, 0.28392571820402046, 0.31861976045493245, 0.34058411615497775, 0.3183586683266787, 0.3529811698448252]
pene_depth2      = [0.4875, 0.9875, 2.0625, 3.1375, 4.2625, 5.4125, 6.5625, 7.7125, 8.8875]


plot2 = plt.plot(np.log(npp2),np.log(pene_depth2),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
npp_syn2 = np.log(np.linspace(np.min(npp2), np.max(npp2),1000))

z2 = np.polyfit(np.log(npp2),np.log(pene_depth2), 1)

plt.plot(npp_syn2, z2[0] * npp_syn2 + z2[1],'g--',color=col_green)
print(z2)


######################################################################
#T_firn = -30 C
#porosity = 0.6
######################################################################
npp3             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop3   = [0.14329550105856603, 0.18140313652936801,0.234550610092216, 0.22623988229547476, 0.23189760717340435, 0.2766599775408536, 0.2916552035366988, 0.3133280319148, 0.28317879417464187]
pene_depth3      = [0.7875, 1.5875,3.2625, 4.9875, 6.7625, 8.5875, 10.4125, 12.2375, 14.1125]


plot3 = plt.plot(np.log(npp3),np.log(pene_depth3),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')
npp_syn3 = np.log(np.linspace(np.min(npp3), np.max(npp3),1000))

z3 = np.polyfit(np.log(npp3),np.log(pene_depth3), 1)

plt.plot(npp_syn3, z3[0] * npp_syn3 + z3[1],'b--',color=col_blue)
print(z3)

######################################################################
#porosity = 0.7
######################################################################
npp4             = [1, 2, 4, 6,14, 8, 10, 12, 16]
porosity_drop4   = [0.11837568185517078, 0.16704739293940318, 0.20883611347729147,0.2358209447478946, 0.30615178210953864, 0.24945399122592582, 0.27458685211098344, 0.21825030817019386, 0.26852472662458127]
pene_depth4      = [1.2375, 2.5125, 5.2125,8.0125, 18.8375, 10.8625, 13.7625, 16.6875, 22.4375]


plot = plt.plot(np.log(npp4),np.log(pene_depth4),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')
npp_syn4 = np.log(np.linspace(np.min(npp4), np.max(npp4),1000))

z4 = np.polyfit(np.log(npp4),np.log(pene_depth4), 1)

plt.plot(npp_syn4, z4[0] * npp_syn4 + z4[1],'k--')
print(z4)

plt.ylim([np.log(20),-1.5])
plt.ylabel(r'log(Penetration depth) [log m]')
plt.legend(loc='best')
plt.xlabel(r'$log(npp)|$ [log days]')
plt.savefig(f'../Figures/fig2gh_nppvspene_loglog.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
AAA = np.array([[ 1.07992587,-1.37344008],
[ 1.0496292,-0.7297281],
[ 1.04332973,-0.25339728],
[1.0446195 , 0.20717396]])
phi_array1 = [0.4, 0.5, 0.6, 0.7]
z_data1    = np.polyfit(phi_array1, AAA[:,1], 1)
print(z_data1)
z_func1    = np.poly1d(np.polyfit(phi_array1, AAA[:,1], 1))
plt.ylabel(r'$d$')
plt.xlabel(r'$\phi$')
plt.plot(phi_array1,AAA[:,1],'ro',markeredgecolor=col_red)
plt.plot(np.linspace(0.3,0.7,100),z_func1(np.linspace(0.3,0.7,100)),'k--')
plt.savefig(f'../Figures/_d_vs_phi.pdf',bbox_inches='tight', dpi = 600)



fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)

######################################################################
#LWC 0.03
#T_firn = -30 C
#porosity = 0.4
######################################################################
plot1 = plt.plot(npp1,pene_depth1,'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')
npp_syn1 = (np.linspace(np.min(npp1), np.max(npp1),1000))

plt.plot(npp_syn1, np.exp(z1[0] * np.log(npp_syn1) + z1[1]),'r--',color=col_red)

######################################################################
#T_firn = -30 C
#porosity = 0.5
######################################################################
npp2             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop2   = [0.12922863664210815, 0.16918960543770367, 0.2111771931499642, 0.22427603762831938, 0.28392571820402046, 0.31861976045493245, 0.34058411615497775, 0.3183586683266787, 0.3529811698448252]
pene_depth2      = [0.4875, 0.9875, 2.0625, 3.1375, 4.2625, 5.4125, 6.5625, 7.7125, 8.8875]


plot2 = plt.plot((npp2),(pene_depth2),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
npp_syn2 = (np.linspace(np.min(npp2), np.max(npp2),1000))

z2 = np.polyfit(np.log(npp2),np.log(pene_depth2), 1)

plt.plot(npp_syn2, np.exp(z2[0] * np.log(npp_syn2) + z2[1]),'g--',color=col_green)

print(z2)

######################################################################
#T_firn = -30 C
#porosity = 0.6
######################################################################
npp3             = [1, 2, 4, 6, 8, 10, 12, 14, 16]
porosity_drop3   = [0.14329550105856603, 0.18140313652936801,0.234550610092216, 0.22623988229547476, 0.23189760717340435, 0.2766599775408536, 0.2916552035366988, 0.3133280319148, 0.28317879417464187]
pene_depth3      = [0.7875, 1.5875,3.2625, 4.9875, 6.7625, 8.5875, 10.4125, 12.2375, 14.1125]


plot3 = plt.plot((npp3),(pene_depth3),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')
npp_syn3 = (np.linspace(np.min(npp3), np.max(npp3),1000))

z3 = np.polyfit(np.log(npp3),np.log(pene_depth3), 1)

plt.plot(npp_syn3, np.exp(z3[0] * np.log(npp_syn3) + z3[1]),'b--',color=col_blue)
print(z3)

######################################################################
#porosity = 0.7
######################################################################
npp4             = [1, 2, 4, 6,14, 8, 10, 12, 16]
porosity_drop4   = [0.11837568185517078, 0.16704739293940318, 0.20883611347729147,0.2358209447478946, 0.2792962020982275, 0.24945399122592582, 0.27458685211098344, 0.21825030817019386, 0.26852472662458127]
pene_depth4      = [1.2375, 2.5125, 5.2125,8.0125, 19.6375, 10.8625, 13.7625, 16.6875, 22.4375]

plot = plt.plot((npp4),(pene_depth4),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')
npp_syn4 = (np.linspace(np.min(npp4), np.max(npp4),1000))

z4 = np.polyfit(np.log(npp4),np.log(pene_depth4), 1)

plt.plot(npp_syn4, np.exp(z4[0] * np.log(npp_syn4) + z4[1]),'k--')
print(z4)

plt.ylim([23,0])
plt.ylabel(r'Penetration depth [m]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.savefig(f'../Figures/fig2gh_nppvspene_new.pdf',bbox_inches='tight', dpi = 600)


######################################################################
#scale z
######################################################################

##hydrology and thermodynamic parameters
m      = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n      = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr   = 0.0   #Residual water saturation
s_gr   = 0.0   #Residual gas saturation
k_w0   = 1.0  #relative permeability threshold: wetting phase
rho_w  = 1000.0  #density of non-wetting phase
mu_w   = 1e-3 #dynamic viscosity: wetting phase    
grav   = 9.81    #acceleration due to gravity
k0     = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017
rho_w = 1000    # density of water [kg/m^3]
cp_w  = 4186    # specific heat of water at constant pressure [J/(kg K)]
k_w   = 0.606   # coefficient of thermal conductivity of water [W / (m K)]
rho_nw = 1.225  # density of gas [kg/m^3]
phi_nw_init = 0.7#0.5    # volumetric ratio of gas, decreasing exponentially
cp_nw  = 1003.5  # specific heat of gas at constant pressure [J/(kg K)]
rho_i = 917     # average density of ice cap [kg/m^3]
cp_i  = 2106.1  # specific heat of ice at constant pressure [J/(kg K)]
k_i   = 2.25    # coefficient of thermal conductivity of ice [W / (m K)]
kk    = k_i/(rho_i*cp_i) # thermal diffusivity of ice
Tm    = 273.16  # melting point temperature of ice [K]
L_fusion= 333.55e3# latent heat of fusion of water [J / kg]
day2s = 60*60*24  #seconds in a day


K_h0 = k0*k_w0*rho_w*grav/mu_w

prefactor = K_h0 * day2s* L_fusion / (cp_i)

fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
#T_firn = -10 C
#porosity = 0.4
LWC = 0.03; Tfirn = -(-10); phi = 0.4
plot1 = plt.plot(npp1,(pene_depth1)/((prefactor*phi/(1-phi))/Tfirn*LWC**2*np.array(npp1)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')



#T_firn = -10 C
#porosity = 0.5
LWC = 0.03; Tfirn = -(-10); phi = 0.5
plot2 = plt.plot((npp2),(pene_depth2)/((prefactor*phi/(1-phi))/Tfirn*LWC**2*np.array(npp2)),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
#plot1 = plt.plot(npp1,(pene_depth1)/(((1-phi-2106.1*10/333.55e3)/(1-phi))*1/Tfirn*LWC**2*np.array(npp2)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

print(z2)


#T_firn = -30 C
#porosity = 0.6
LWC = 0.03; Tfirn = -(-10); phi = 0.6
plot3 = plt.plot((npp3),(pene_depth3)/((prefactor*phi/(1-phi))/Tfirn*LWC**2*np.array(npp3)),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


print(z3)

#porosity = 0.7
LWC = 0.03; Tfirn = -(-10); phi = 0.7
plot = plt.plot((npp4),(pene_depth4)/((prefactor*phi/(1-phi))/Tfirn*LWC**2*np.array(npp4)),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')

print(z4)

#plt.yscale("log")
plt.ylim([0, 1])
#plt.ylim([23,0])
plt.ylabel(r'$\frac{z_{p}}{\frac{I \cdot L}{(1-\phi){c_{p}\cdot(T_m - {T_{0}})}}\cdot t_{pulse}}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.tight_layout()
plt.savefig(f'../Figures/fig2gh_nppvspene_new_scale_z.pdf',bbox_inches='tight', dpi = 600)



fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
LWC = 0.03; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn11)),(pene_depth11)/(prefactor*phi/(1-phi)/(-T_firn11)*LWC**2*npp),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.03')

#LWC 0.04
LWC = 0.04; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn22)),(pene_depth22)/(prefactor*phi/(1-phi)/(-T_firn22)*LWC**2*npp),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.04')

#LWC 0.05
LWC = 0.05; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn33)),(pene_depth33)/(prefactor*phi/(1-phi)/(-T_firn33)*LWC**2*npp),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.05')

#LWC 0.06
LWC = 0.06; phi = 0.5; npp =4
lot = plt.plot((np.abs(-T_firn44)),(pene_depth44)/(prefactor*phi/(1-phi)/(-T_firn44)*LWC**2*npp),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.06')

#LWC 0.07
LWC = 0.07; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn55)),(pene_depth55)/(prefactor*phi/(1-phi)/(-T_firn55)*LWC**2*npp),'o',markeredgecolor=[0.5,0.5,0.5],markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.07')


#plt.plot(np.linspace(-15,-5,100),z_p(4*day2s,2,0.06,0.5,np.linspace(-15,-5,100)+273.16),'k--')
#plt.yscale("log")
#plt.ylim([1e-1, 1e2])
plt.ylim([0, 1])
plt.ylabel(r'$\frac{z_{p}}{\frac{I \cdot L}{(1-\phi){c_{p}\cdot(T_m - {T_{0}})}}\cdot t_{pulse}}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$T_m-T_{0}[^\circ C]$')
plt.tight_layout()
#plt.axis('scaled')
#plt.ylim([np.log(20),0])
plt.savefig(f'../Figures/T_firnvspene_new_scale_z.pdf',bbox_inches='tight', dpi = 600)

#Dimensional

fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
#T_firn = -10 C
#porosity = 0.4
LWC = 0.03; Tfirn = -(-10); phi = 0.4
plot1 = plt.plot(npp1,(pene_depth1),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')


#T_firn = -10 C
#porosity = 0.5
LWC = 0.03; Tfirn = -(-10); phi = 0.5
plot2 = plt.plot((npp2),(pene_depth2),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
#plot1 = plt.plot(npp1,(pene_depth1)/(((1-phi-2106.1*10/333.55e3)/(1-phi))*1/Tfirn*LWC**2*np.array(npp2)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

print(z2)


#T_firn = -30 C
#porosity = 0.6
LWC = 0.03; Tfirn = -(-10); phi = 0.6
plot3 = plt.plot((npp3),(pene_depth3),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


print(z3)

#porosity = 0.7
LWC = 0.03; Tfirn = -(-10); phi = 0.7
plot = plt.plot((npp4),(pene_depth4),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')

print(z4)
plt.ylim([23,0])
plt.ylabel(r'Penetration depth, $z_{p}$ [m]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.tight_layout()
plt.savefig(f'../Figures/fig2gh_nppvspene_new_scale_z_dim.pdf',bbox_inches='tight', dpi = 600)



fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
LWC = 0.03; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn11)),(pene_depth11),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.03')

#LWC 0.04
LWC = 0.04; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn22)),(pene_depth22),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.04')

#LWC 0.05
LWC = 0.05; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn33)),(pene_depth33),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.05')

#LWC 0.06
LWC = 0.06; phi = 0.5; npp =4
lot = plt.plot((np.abs(-T_firn44)),(pene_depth44),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.06')

#LWC 0.07
LWC = 0.07; phi = 0.5; npp =4
plot = plt.plot((np.abs(-T_firn55)),(pene_depth55),'o',markeredgecolor=[0.5,0.5,0.5],markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.07')


#plt.plot(np.linspace(-15,-5,100),z_p(4*day2s,2,0.06,0.5,np.linspace(-15,-5,100)+273.16),'k--')
plt.ylabel(r'Penetration depth, $z_{p}$ [m]')
plt.legend(loc='best')
plt.xlabel(r'$T_m-T_{0}[^\circ C]$')
#plt.axis('scaled')
plt.ylim([23,0])
plt.tight_layout()
plt.savefig(f'../Figures/T_firnvspene_new_scale_z_dim.pdf',bbox_inches='tight', dpi = 600)



#new porosity
K_h0 = k0*k_w0*rho_w*grav/mu_w

prefactor = K_h0 * day2s* L_fusion

fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
#T_firn = -10 C
#porosity = 0.4
LWC = 0.03; Tfirn = -(-10); phi = 0.4
plot1 = plt.plot(npp1,pene_depth1/((prefactor*phi/((1-phi)*cp_i*Tfirn + 0 *LWC * L_fusion)*LWC**2*np.array(npp1))),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')


#T_firn = -10 C
#porosity = 0.5
LWC = 0.03; Tfirn = -(-10); phi = 0.5
plot2 = plt.plot((npp2),(pene_depth2)/((prefactor*phi/((1-phi)*cp_i*Tfirn + 0 *LWC * L_fusion)*LWC**2*np.array(npp2))),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
#plot1 = plt.plot(npp1,(pene_depth1)/(((1-phi-2106.1*10/333.55e3)/(1-phi))*1/Tfirn*LWC**2*np.array(npp2)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

print(z2)


#T_firn = -30 C
#porosity = 0.6
LWC = 0.03; Tfirn = -(-10); phi = 0.6
plot3 = plt.plot((npp3),(pene_depth3)/((prefactor*phi/((1-phi)*cp_i*Tfirn + 0 *LWC * L_fusion )*LWC**2*np.array(npp3))),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


print(z3)

#porosity = 0.7
LWC = 0.03; Tfirn = -(-10); phi = 0.7
plot = plt.plot((npp4),(pene_depth4)/((prefactor*phi/((1-phi)*cp_i*Tfirn + 0 *LWC * L_fusion)*LWC**2*np.array(npp4))),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')

print(z4)

#plt.yscale("log")
plt.ylim([0, 1])
#plt.ylim([23,0])
plt.ylabel(r'$\frac{z_{p}}{\frac{I \cdot L}{(1-\phi){c_{p}\cdot(T_m - {T_{0}})}}\cdot t_{pulse}}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.tight_layout()
plt.savefig(f'../Figures/fig2gh_nppvspene_new_scale_z_new_scaling.pdf',bbox_inches='tight', dpi = 600)



fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
LWC = 0.03; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*(np.abs(T_firn11)/Tm) + LWC))
ti  =npp * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot = plt.plot((np.abs(-T_firn11)),(pene_depth11)/(Lambda*ti),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.03')

#LWC 0.04
LWC = 0.04; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*(np.abs(T_firn22)/Tm) + LWC))
ti  = npp * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot = plt.plot((np.abs(-T_firn22)),(pene_depth22)/(Lambda*ti),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.04')

#LWC 0.05
LWC = 0.05; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*(np.abs(T_firn33)/Tm) + LWC))
ti  = npp * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot = plt.plot((np.abs(-T_firn33)),(pene_depth33)/(Lambda*ti),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.05')

#LWC 0.06
LWC = 0.06; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*(np.abs(T_firn44)/Tm) + LWC))
ti  = npp * (1/(1- (Lambda/(2*fc/LWC))))* day2s
lot = plt.plot((np.abs(-T_firn44)),(pene_depth44)/(Lambda*ti),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.06')

#LWC 0.07
LWC = 0.07; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*(np.abs(T_firn55)/Tm) + LWC))
ti  = npp * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot = plt.plot((np.abs(-T_firn55)),(pene_depth55)/(Lambda*ti),'o',markeredgecolor=[0.5,0.5,0.5],markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.07')


#plt.plot(np.linspace(-15,-5,100),z_p(4*day2s,2,0.06,0.5,np.linspace(-15,-5,100)+273.16),'k--')
#plt.yscale("log")
#plt.ylim([1e-1, 1e2])
#plt.ylim([0, 1])
plt.ylabel(r'$\frac{z_{p}}{z_i}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$T_m-T_{0}[^\circ C]$')
plt.tight_layout()
#plt.axis('scaled')
#plt.ylim([np.log(20),0])
plt.savefig(f'../Figures/T_firnvspene_final_scale_z.pdf',bbox_inches='tight', dpi = 600)



#####Final scaling
#new porosity

fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
#T_firn = -10 C
#porosity = 0.4
LWC = 0.03; Tfirn = -(-10); phi = 0.4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*((Tfirn)/Tm) + LWC))
ti  = np.array(npp1) * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot1 = plt.plot(npp1,pene_depth1/(Lambda*ti),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')


#T_firn = -10 C
#porosity = 0.5
LWC = 0.03; Tfirn = -(-10); phi = 0.5
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*((Tfirn)/Tm) + LWC))
ti  = np.array(npp2) * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot2 = plt.plot((npp2),(pene_depth2)/(Lambda*ti),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
#plot1 = plt.plot(npp1,(pene_depth1)/(((1-phi-2106.1*10/333.55e3)/(1-phi))*1/Tfirn*LWC**2*np.array(npp2)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

print(z2)


#T_firn = -30 C
#porosity = 0.6
LWC = 0.03; Tfirn = -(-10); phi = 0.6
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*((Tfirn)/Tm) + LWC))
ti  = np.array(npp3) * (1/(1- (Lambda/(2*fc/LWC))))* day2s
plot3 = plt.plot((npp3),(pene_depth3)/(Lambda*ti),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


print(z3)

#porosity = 0.7
LWC = 0.03; Tfirn = -(-10); phi = 0.7
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*((Tfirn)/Tm) + LWC))
ti  = np.array(npp4) * (1/(1- (Lambda/(2*fc/LWC)))) * day2s
plot = plt.plot((npp4),(pene_depth4)/(Lambda*ti),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')

print(z4)

#plt.yscale("log")
#plt.ylim([0, 1])
#plt.ylim([23,0])
plt.ylabel(r'$\frac{z_{p}}{z_i}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.tight_layout()
plt.savefig(f'../Figures/fig2gh_nppvspene_new_scale_z_final_scaling.pdf',bbox_inches='tight', dpi = 600)


#####Scaling with LWC
fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
LWC = 0.03; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/( LWC))
ti  =npp * day2s
plot = plt.plot((np.abs(-T_firn11)),np.array(pene_depth11)/(Lambda*ti),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.03')

#LWC 0.04
LWC = 0.04; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/( LWC))
ti  = npp * day2s
plot = plt.plot((np.abs(-T_firn22)),np.array(pene_depth22)/(Lambda*ti),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.04')

#LWC 0.05
LWC = 0.05; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/( LWC))
ti  = npp * day2s
plot = plt.plot((np.abs(-T_firn33)),np.array(pene_depth33)/(Lambda*ti),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.05')

#LWC 0.06
LWC = 0.06; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/(LWC))
ti  = npp * day2s
lot = plt.plot((np.abs(-T_firn44)),np.array(pene_depth44)/(Lambda*ti),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.06')

#LWC 0.07
LWC = 0.07; phi = 0.5; npp =4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/( LWC))
ti  = npp * day2s
plot = plt.plot((np.abs(-T_firn55)),np.array(pene_depth55)/(Lambda*ti),'o',markeredgecolor=[0.5,0.5,0.5],markersize=18, markerfacecolor='white',markeredgewidth=3,label='LWC=0.07')


plt.ylabel(r'$\frac{z_{p}}{I~/~LWC}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$T_m-T_{0}[^\circ C]$')
plt.tight_layout()
plt.savefig(f'../Figures/T_firnvspene_final_scale_z_tentative.pdf',bbox_inches='tight', dpi = 600)


######################################################################
#####Simple scaling
#new porosity
######################################################################
fig = plt.figure(figsize=(12,8) , dpi=100)
ax = fig.add_subplot(1, 1, 1)
#LWC 0.03
#T_firn = -10 C
#porosity = 0.4
LWC = 0.03; Tfirn = -(-10); phi = 0.4
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/(LWC))
ti  = np.array(npp1) * day2s
plot1 = plt.plot(npp1,np.array(pene_depth1)/(Lambda*ti),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

######################################################################
#T_firn = -10 C
#porosity = 0.5
######################################################################
LWC = 0.03; Tfirn = -(-10); phi = 0.5
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/((1-phi)*cp_i*Tm/L_fusion*((Tfirn)/Tm) + LWC))
ti  = np.array(npp2) * day2s
plot2 = plt.plot((npp2),np.array(pene_depth2)/(Lambda*ti),'go',markeredgecolor=col_green,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.5$')
#plot1 = plt.plot(npp1,(pene_depth1)/(((1-phi-2106.1*10/333.55e3)/(1-phi))*1/Tfirn*LWC**2*np.array(npp2)),'ro',markeredgecolor=col_red,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.4$')

print(z2)

######################################################################
#T_firn = -30 C
#porosity = 0.6
######################################################################
LWC = 0.03; Tfirn = -(-10); phi = 0.6
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/( LWC))
ti  = np.array(npp3) * day2s
plot3 = plt.plot((npp3),np.array(pene_depth3)/(Lambda*ti),'bo',markeredgecolor=col_blue,markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.6$')


print(z3)
######################################################################
#porosity = 0.7
######################################################################
LWC = 0.03; Tfirn = -(-10); phi = 0.7
fc  = k0*k_w0*rho_w*grav/mu_w*phi**3*(LWC/phi)**2; Lambda = ((fc)/(LWC))
ti  = np.array(npp4) * day2s
plot = plt.plot((npp4),np.array(pene_depth4)/(Lambda*ti),'ko',markersize=18, markerfacecolor='white',markeredgewidth=3,label=r'$\phi_0=0.7$')

print(z4)

plt.ylabel(r'$\frac{z_{p}}{I~/~LWC}$ [-]')
plt.legend(loc='best')
plt.xlabel(r'$t_{pulse}$ [days]')
plt.tight_layout()
plt.savefig(f'../Figures/fig2gh_nppvspene_new_scale_z_final_scaling_tentative.pdf',bbox_inches='tight', dpi = 600)
