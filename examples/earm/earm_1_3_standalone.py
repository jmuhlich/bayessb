"""
EARM 1.3 (extrinsic apoptosis reaction model)

Gaudet S, Spencer SL, Chen WW, Sorger PK (2012) Exploring the Contextual
Sensitivity of Factors that Determine Cell-to-Cell Variability in
Receptor-Mediated Apoptosis. PLoS Comput Biol 8(4): e1002482.
doi:10.1371/journal.pcbi.1002482

http://www.ploscompbiol.org/article/info:doi/10.1371/journal.pcbi.1002482
"""

# exported from PySB model 'earm_1_3'

import numpy
import scipy.weave, scipy.integrate
import collections
import itertools
import distutils.errors


_use_inline = False
# try to inline a C statement to see if inline is functional
try:
    scipy.weave.inline('int i;', force=1)
    _use_inline = True
except distutils.errors.CompileError:
    pass

Parameter = collections.namedtuple('Parameter', 'name value')
Observable = collections.namedtuple('Observable', 'name species coefficients')
Initial = collections.namedtuple('Initial', 'param_index species_index')


class Model(object):
    
    def __init__(self):
        self.y = None
        self.yobs = None
        self.integrator = scipy.integrate.ode(self.ode_rhs, )
        self.integrator.set_integrator('vode', method='bdf', with_jacobian=True, rtol=1e-3, atol=1e-6)
        self.y0 = numpy.empty(60)
        self.ydot = numpy.empty(60)
        self.sim_param_values = numpy.empty(206)
        self.parameters = [None] * 206
        self.observables = [None] * 6
        self.initial_conditions = [None] * 19
    
        self.parameters[0] = Parameter('L_0', 3000)
        self.parameters[1] = Parameter('pR_0', 1000)
        self.parameters[2] = Parameter('flip_0', 2000)
        self.parameters[3] = Parameter('pC8_0', 10000)
        self.parameters[4] = Parameter('BAR_0', 1000)
        self.parameters[5] = Parameter('pC3_0', 10000)
        self.parameters[6] = Parameter('pC6_0', 10000)
        self.parameters[7] = Parameter('XIAP_0', 100000)
        self.parameters[8] = Parameter('PARP_0', 1e+06)
        self.parameters[9] = Parameter('Bid_0', 60000)
        self.parameters[10] = Parameter('Mcl1_0', 20000)
        self.parameters[11] = Parameter('Bax_0', 80000)
        self.parameters[12] = Parameter('Bcl2_0', 30000)
        self.parameters[13] = Parameter('Mito_0', 500000)
        self.parameters[14] = Parameter('mCytoC_0', 500000)
        self.parameters[15] = Parameter('mSmac_0', 100000)
        self.parameters[16] = Parameter('pC9_0', 100000)
        self.parameters[17] = Parameter('Apaf_0', 100000)
        self.parameters[18] = Parameter('kf1', 4e-07)
        self.parameters[19] = Parameter('kr1', 1e-06)
        self.parameters[20] = Parameter('kc1', 0.01)
        self.parameters[21] = Parameter('kf2', 1e-06)
        self.parameters[22] = Parameter('kr2', 0.001)
        self.parameters[23] = Parameter('kf3', 1e-07)
        self.parameters[24] = Parameter('kr3', 0.001)
        self.parameters[25] = Parameter('kc3', 1)
        self.parameters[26] = Parameter('kf4', 1e-06)
        self.parameters[27] = Parameter('kr4', 0.001)
        self.parameters[28] = Parameter('kf5', 1e-07)
        self.parameters[29] = Parameter('kr5', 0.001)
        self.parameters[30] = Parameter('kc5', 1)
        self.parameters[31] = Parameter('kf6', 1e-07)
        self.parameters[32] = Parameter('kr6', 0.001)
        self.parameters[33] = Parameter('kc6', 1)
        self.parameters[34] = Parameter('kf7', 1e-07)
        self.parameters[35] = Parameter('kr7', 0.001)
        self.parameters[36] = Parameter('kc7', 1)
        self.parameters[37] = Parameter('kf8', 2e-06)
        self.parameters[38] = Parameter('kr8', 0.001)
        self.parameters[39] = Parameter('kc8', 0.1)
        self.parameters[40] = Parameter('kf9', 1e-06)
        self.parameters[41] = Parameter('kr9', 0.001)
        self.parameters[42] = Parameter('kc9', 20)
        self.parameters[43] = Parameter('kf10', 1e-07)
        self.parameters[44] = Parameter('kr10', 0.001)
        self.parameters[45] = Parameter('kc10', 1)
        self.parameters[46] = Parameter('kf11', 1e-06)
        self.parameters[47] = Parameter('kr11', 0.001)
        self.parameters[48] = Parameter('kf12', 1e-07)
        self.parameters[49] = Parameter('kr12', 0.001)
        self.parameters[50] = Parameter('kc12', 1)
        self.parameters[51] = Parameter('kf13', 0.01)
        self.parameters[52] = Parameter('kr13', 1)
        self.parameters[53] = Parameter('kf14', 0.0001)
        self.parameters[54] = Parameter('kr14', 0.001)
        self.parameters[55] = Parameter('kf15', 0.0002)
        self.parameters[56] = Parameter('kr15', 0.001)
        self.parameters[57] = Parameter('kf16', 0.0001)
        self.parameters[58] = Parameter('kr16', 0.001)
        self.parameters[59] = Parameter('kf17', 0.0002)
        self.parameters[60] = Parameter('kr17', 0.001)
        self.parameters[61] = Parameter('kf18', 0.0001)
        self.parameters[62] = Parameter('kr18', 0.001)
        self.parameters[63] = Parameter('kf19', 0.0001)
        self.parameters[64] = Parameter('kr19', 0.001)
        self.parameters[65] = Parameter('kc19', 1)
        self.parameters[66] = Parameter('kf20', 0.0002)
        self.parameters[67] = Parameter('kr20', 0.001)
        self.parameters[68] = Parameter('kc20', 10)
        self.parameters[69] = Parameter('kf21', 0.0002)
        self.parameters[70] = Parameter('kr21', 0.001)
        self.parameters[71] = Parameter('kc21', 10)
        self.parameters[72] = Parameter('kf22', 1)
        self.parameters[73] = Parameter('kr22', 0.01)
        self.parameters[74] = Parameter('kf23', 5e-07)
        self.parameters[75] = Parameter('kr23', 0.001)
        self.parameters[76] = Parameter('kc23', 1)
        self.parameters[77] = Parameter('kf24', 5e-08)
        self.parameters[78] = Parameter('kr24', 0.001)
        self.parameters[79] = Parameter('kf25', 5e-09)
        self.parameters[80] = Parameter('kr25', 0.001)
        self.parameters[81] = Parameter('kc25', 1)
        self.parameters[82] = Parameter('kf26', 1)
        self.parameters[83] = Parameter('kr26', 0.01)
        self.parameters[84] = Parameter('kf27', 2e-06)
        self.parameters[85] = Parameter('kr27', 0.001)
        self.parameters[86] = Parameter('kf28', 7e-06)
        self.parameters[87] = Parameter('kr28', 0.001)
        self.parameters[88] = Parameter('kf31', 0.001)
        self.parameters[89] = Parameter('kdeg_Mcl1', 0.0001)
        self.parameters[90] = Parameter('kdeg_AMito', 0.0001)
        self.parameters[91] = Parameter('kdeg_C3_U', 0)
        self.parameters[92] = Parameter('ks_L', 0)
        self.parameters[93] = Parameter('kdeg_L', 2.9e-06)
        self.parameters[94] = Parameter('kdeg_pR', 2.9e-06)
        self.parameters[95] = Parameter('ks_pR', 0.000435)
        self.parameters[96] = Parameter('kdeg_flip', 2.9e-06)
        self.parameters[97] = Parameter('ks_flip', 0.00087)
        self.parameters[98] = Parameter('kdeg_pC8', 2.9e-06)
        self.parameters[99] = Parameter('ks_pC8', 0.00435)
        self.parameters[100] = Parameter('kdeg_BAR', 2.9e-06)
        self.parameters[101] = Parameter('ks_BAR', 0.000435)
        self.parameters[102] = Parameter('kdeg_pC3', 2.9e-06)
        self.parameters[103] = Parameter('ks_pC3', 0.00435)
        self.parameters[104] = Parameter('kdeg_pC6', 2.9e-06)
        self.parameters[105] = Parameter('ks_pC6', 0.00435)
        self.parameters[106] = Parameter('kdeg_XIAP', 2.9e-06)
        self.parameters[107] = Parameter('ks_XIAP', 0.0435)
        self.parameters[108] = Parameter('kdeg_PARP', 2.9e-06)
        self.parameters[109] = Parameter('ks_PARP', 0.435)
        self.parameters[110] = Parameter('kdeg_Bid', 2.9e-06)
        self.parameters[111] = Parameter('ks_Bid', 0.0261)
        self.parameters[112] = Parameter('ks_Mcl1', 0.3)
        self.parameters[113] = Parameter('kdeg_Bax', 2.9e-06)
        self.parameters[114] = Parameter('ks_Bax', 0.0348)
        self.parameters[115] = Parameter('kdeg_Bcl2', 2.9e-06)
        self.parameters[116] = Parameter('ks_Bcl2', 0.01305)
        self.parameters[117] = Parameter('kdeg_Mito', 2.9e-06)
        self.parameters[118] = Parameter('ks_Mito', 0.2175)
        self.parameters[119] = Parameter('kdeg_mCytoC', 2.9e-06)
        self.parameters[120] = Parameter('ks_mCytoC', 0.2175)
        self.parameters[121] = Parameter('kdeg_mSmac', 2.9e-06)
        self.parameters[122] = Parameter('ks_mSmac', 0.0435)
        self.parameters[123] = Parameter('kdeg_Apaf', 2.9e-06)
        self.parameters[124] = Parameter('ks_Apaf', 0.0435)
        self.parameters[125] = Parameter('kdeg_pC9', 2.9e-06)
        self.parameters[126] = Parameter('ks_pC9', 0.0435)
        self.parameters[127] = Parameter('kdeg_L_pR', 2.9e-06)
        self.parameters[128] = Parameter('ks_L_pR', 0)
        self.parameters[129] = Parameter('kdeg_DISC', 2.9e-06)
        self.parameters[130] = Parameter('ks_DISC', 0)
        self.parameters[131] = Parameter('kdeg_DISC_flip', 2.9e-06)
        self.parameters[132] = Parameter('ks_DISC_flip', 0)
        self.parameters[133] = Parameter('kdeg_DISC_pC8', 2.9e-06)
        self.parameters[134] = Parameter('ks_DISC_pC8', 0)
        self.parameters[135] = Parameter('kdeg_C8', 2.9e-06)
        self.parameters[136] = Parameter('ks_C8', 0)
        self.parameters[137] = Parameter('kdeg_BAR_C8', 2.9e-06)
        self.parameters[138] = Parameter('ks_BAR_C8', 0)
        self.parameters[139] = Parameter('kdeg_C8_pC3', 2.9e-06)
        self.parameters[140] = Parameter('ks_C8_pC3', 0)
        self.parameters[141] = Parameter('kdeg_Bid_C8', 2.9e-06)
        self.parameters[142] = Parameter('ks_Bid_C8', 0)
        self.parameters[143] = Parameter('kdeg_C3', 2.9e-06)
        self.parameters[144] = Parameter('ks_C3', 0)
        self.parameters[145] = Parameter('kdeg_tBid', 2.9e-06)
        self.parameters[146] = Parameter('ks_tBid', 0)
        self.parameters[147] = Parameter('kdeg_C3_pC6', 2.9e-06)
        self.parameters[148] = Parameter('ks_C3_pC6', 0)
        self.parameters[149] = Parameter('kdeg_C3_XIAP', 2.9e-06)
        self.parameters[150] = Parameter('ks_C3_XIAP', 0)
        self.parameters[151] = Parameter('kdeg_C3_PARP', 2.9e-06)
        self.parameters[152] = Parameter('ks_C3_PARP', 0)
        self.parameters[153] = Parameter('kdeg_Mcl1_tBid', 2.9e-06)
        self.parameters[154] = Parameter('ks_Mcl1_tBid', 0)
        self.parameters[155] = Parameter('kdeg_Bax_tBid', 2.9e-06)
        self.parameters[156] = Parameter('ks_Bax_tBid', 0)
        self.parameters[157] = Parameter('kdeg_C6', 2.9e-06)
        self.parameters[158] = Parameter('ks_C6', 0)
        self.parameters[159] = Parameter('ks_C3_U', 0)
        self.parameters[160] = Parameter('kdeg_CPARP', 2.9e-06)
        self.parameters[161] = Parameter('ks_CPARP', 0)
        self.parameters[162] = Parameter('kdeg_aBax', 2.9e-06)
        self.parameters[163] = Parameter('ks_aBax', 0)
        self.parameters[164] = Parameter('kdeg_C6_pC8', 2.9e-06)
        self.parameters[165] = Parameter('ks_C6_pC8', 0)
        self.parameters[166] = Parameter('kdeg_MBax', 2.9e-06)
        self.parameters[167] = Parameter('ks_MBax', 0)
        self.parameters[168] = Parameter('kdeg_Bcl2_MBax', 2.9e-06)
        self.parameters[169] = Parameter('ks_Bcl2_MBax', 0)
        self.parameters[170] = Parameter('kdeg_Bax2', 2.9e-06)
        self.parameters[171] = Parameter('ks_Bax2', 0)
        self.parameters[172] = Parameter('kdeg_Bax2_Bcl2', 2.9e-06)
        self.parameters[173] = Parameter('ks_Bax2_Bcl2', 0)
        self.parameters[174] = Parameter('kdeg_Bax4', 2.9e-06)
        self.parameters[175] = Parameter('ks_Bax4', 0)
        self.parameters[176] = Parameter('kdeg_Bax4_Bcl2', 2.9e-06)
        self.parameters[177] = Parameter('ks_Bax4_Bcl2', 0)
        self.parameters[178] = Parameter('kdeg_Bax4_Mito', 2.9e-06)
        self.parameters[179] = Parameter('ks_Bax4_Mito', 0)
        self.parameters[180] = Parameter('ks_AMito', 0)
        self.parameters[181] = Parameter('kdeg_AMito_mCytoC', 2.9e-06)
        self.parameters[182] = Parameter('ks_AMito_mCytoC', 0)
        self.parameters[183] = Parameter('kdeg_AMito_mSmac', 2.9e-06)
        self.parameters[184] = Parameter('ks_AMito_mSmac', 0)
        self.parameters[185] = Parameter('kdeg_ACytoC', 2.9e-06)
        self.parameters[186] = Parameter('ks_ACytoC', 0)
        self.parameters[187] = Parameter('kdeg_ASmac', 2.9e-06)
        self.parameters[188] = Parameter('ks_ASmac', 0)
        self.parameters[189] = Parameter('kdeg_cCytoC', 2.9e-06)
        self.parameters[190] = Parameter('ks_cCytoC', 0)
        self.parameters[191] = Parameter('kdeg_cSmac', 2.9e-06)
        self.parameters[192] = Parameter('ks_cSmac', 0)
        self.parameters[193] = Parameter('kdeg_Apaf_cCytoC', 2.9e-06)
        self.parameters[194] = Parameter('ks_Apaf_cCytoC', 0)
        self.parameters[195] = Parameter('kdeg_XIAP_cSmac', 2.9e-06)
        self.parameters[196] = Parameter('ks_XIAP_cSmac', 0)
        self.parameters[197] = Parameter('kdeg_aApaf', 2.9e-06)
        self.parameters[198] = Parameter('ks_aApaf', 0)
        self.parameters[199] = Parameter('kdeg_Apop', 2.9e-06)
        self.parameters[200] = Parameter('ks_Apop', 0)
        self.parameters[201] = Parameter('kdeg_Apop_pC3', 2.9e-06)
        self.parameters[202] = Parameter('ks_Apop_pC3', 0)
        self.parameters[203] = Parameter('kdeg_Apop_XIAP', 2.9e-06)
        self.parameters[204] = Parameter('ks_Apop_XIAP', 0)
        self.parameters[205] = Parameter('__source_0', 1)

        self.observables[0] = Observable('Bid_unbound', [9], [1])
        self.observables[1] = Observable('PARP_unbound', [8], [1])
        self.observables[2] = Observable('mSmac_unbound', [15], [1])
        self.observables[3] = Observable('tBid_total', [29, 33, 34], [1, 1, 1])
        self.observables[4] = Observable('CPARP_total', [37], [1])
        self.observables[5] = Observable('cSmac_total', [53, 55], [1, 1])

        self.initial_conditions[0] = Initial(0, 0)
        self.initial_conditions[1] = Initial(1, 1)
        self.initial_conditions[2] = Initial(2, 2)
        self.initial_conditions[3] = Initial(3, 3)
        self.initial_conditions[4] = Initial(4, 4)
        self.initial_conditions[5] = Initial(5, 5)
        self.initial_conditions[6] = Initial(6, 6)
        self.initial_conditions[7] = Initial(7, 7)
        self.initial_conditions[8] = Initial(8, 8)
        self.initial_conditions[9] = Initial(9, 9)
        self.initial_conditions[10] = Initial(10, 10)
        self.initial_conditions[11] = Initial(11, 11)
        self.initial_conditions[12] = Initial(12, 12)
        self.initial_conditions[13] = Initial(13, 13)
        self.initial_conditions[14] = Initial(14, 14)
        self.initial_conditions[15] = Initial(15, 15)
        self.initial_conditions[16] = Initial(17, 16)
        self.initial_conditions[17] = Initial(16, 17)
        self.initial_conditions[18] = Initial(205, 18)

    if _use_inline:
        
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            scipy.weave.inline(r'''                
                ydot[0] = -p[93]*y[0] - p[18]*y[0]*y[1] + p[88]*y[21] + p[19]*y[19] + p[92]*y[18];
                ydot[1] = -p[94]*y[1] - p[18]*y[0]*y[1] + p[88]*y[21] + p[19]*y[19] + p[95]*y[18];
                ydot[2] = -p[96]*y[2] - p[21]*y[2]*y[21] + p[22]*y[22] + p[97]*y[18];
                ydot[3] = -p[98]*y[3] - p[23]*y[21]*y[3] - p[34]*y[3]*y[35] + p[24]*y[23] + p[35]*y[39] + p[99]*y[18];
                ydot[4] = -p[100]*y[4] - p[26]*y[24]*y[4] + p[27]*y[25] + p[101]*y[18];
                ydot[5] = -p[102]*y[5] - p[79]*y[5]*y[57] - p[28]*y[24]*y[5] + p[80]*y[58] + p[29]*y[26] + p[103]*y[18];
                ydot[6] = -p[104]*y[6] - p[31]*y[28]*y[6] + p[32]*y[30] + p[105]*y[18];
                ydot[7] = p[39]*y[31] - p[106]*y[7] - p[84]*y[57]*y[7] - p[86]*y[53]*y[7] - p[37]*y[28]*y[7] + p[85]*y[59] + p[87]*y[55] + p[38]*y[31] + p[107]*y[18];
                ydot[8] = -p[108]*y[8] - p[40]*y[28]*y[8] + p[41]*y[32] + p[109]*y[18];
                ydot[9] = -p[110]*y[9] - p[43]*y[24]*y[9] + p[44]*y[27] + p[111]*y[18];
                ydot[10] = -p[89]*y[10] - p[46]*y[10]*y[29] + p[47]*y[33] + p[112]*y[18];
                ydot[11] = -p[113]*y[11] - p[48]*y[11]*y[29] + p[49]*y[34] + p[114]*y[18];
                ydot[12] = -p[115]*y[12] - p[53]*y[12]*y[40] - p[57]*y[12]*y[42] - p[61]*y[12]*y[44] + p[54]*y[41] + p[58]*y[43] + p[62]*y[45] + p[116]*y[18];
                ydot[13] = p[90]*y[47] - p[117]*y[13] - p[63]*y[13]*y[44] + p[64]*y[46] + p[118]*y[18];
                ydot[14] = -p[119]*y[14] - p[66]*y[14]*y[47] + p[67]*y[48] + p[120]*y[18];
                ydot[15] = -p[121]*y[15] - p[69]*y[15]*y[47] + p[70]*y[49] + p[122]*y[18];
                ydot[16] = -p[123]*y[16] - p[74]*y[16]*y[52] + p[75]*y[54] + p[124]*y[18];
                ydot[17] = -p[125]*y[17] - p[77]*y[17]*y[56] + p[78]*y[57] + p[126]*y[18];
                ydot[18] = 0;
                ydot[19] = -p[20]*y[19] - p[127]*y[19] + p[18]*y[0]*y[1] - p[19]*y[19] + p[128]*y[18];
                ydot[20] = p[185]*y[50] + p[181]*y[48] + p[183]*y[49] + p[187]*y[51] + p[123]*y[16] + p[193]*y[54] + p[199]*y[57] + p[203]*y[59] + p[201]*y[58] + p[100]*y[4] + p[137]*y[25] + p[113]*y[11] + p[170]*y[42] + p[172]*y[43] + p[174]*y[44] + p[176]*y[45] + p[178]*y[46] + p[155]*y[34] + p[115]*y[12] + p[168]*y[41] + p[110]*y[9] + p[141]*y[27] + p[143]*y[28] + p[151]*y[32] + p[91]*y[36] + p[149]*y[31] + p[147]*y[30] + p[157]*y[35] + p[164]*y[39] + p[135]*y[24] + p[139]*y[26] + p[160]*y[37] + p[131]*y[22] + p[133]*y[23] + p[93]*y[0] + p[127]*y[19] + p[166]*y[40] + p[89]*y[10] + p[153]*y[33] + p[117]*y[13] + p[108]*y[8] + p[106]*y[7] + p[195]*y[55] + p[197]*y[56] + p[162]*y[38] + p[189]*y[52] + p[191]*y[53] + p[96]*y[2] + p[119]*y[14] + p[121]*y[15] + p[102]*y[5] + p[104]*y[6] + p[98]*y[3] + p[125]*y[17] + p[94]*y[1] + p[145]*y[29];
                ydot[21] = p[20]*y[19] + p[25]*y[23] - p[21]*y[2]*y[21] - p[23]*y[21]*y[3] - p[88]*y[21] + p[22]*y[22] + p[24]*y[23] + p[130]*y[18];
                ydot[22] = -p[131]*y[22] + p[21]*y[2]*y[21] - p[22]*y[22] + p[132]*y[18];
                ydot[23] = -p[25]*y[23] - p[133]*y[23] + p[23]*y[21]*y[3] - p[24]*y[23] + p[134]*y[18];
                ydot[24] = p[45]*y[27] + p[25]*y[23] + p[30]*y[26] + p[36]*y[39] - p[135]*y[24] - p[43]*y[24]*y[9] - p[26]*y[24]*y[4] - p[28]*y[24]*y[5] + p[44]*y[27] + p[27]*y[25] + p[29]*y[26] + p[136]*y[18];
                ydot[25] = -p[137]*y[25] + p[26]*y[24]*y[4] - p[27]*y[25] + p[138]*y[18];
                ydot[26] = -p[30]*y[26] - p[139]*y[26] + p[28]*y[24]*y[5] - p[29]*y[26] + p[140]*y[18];
                ydot[27] = -p[45]*y[27] - p[141]*y[27] + p[43]*y[24]*y[9] - p[44]*y[27] + p[142]*y[18];
                ydot[28] = p[81]*y[58] + p[30]*y[26] + p[33]*y[30] + p[42]*y[32] - p[143]*y[28] - p[31]*y[28]*y[6] - p[37]*y[28]*y[7] - p[40]*y[28]*y[8] + p[32]*y[30] + p[38]*y[31] + p[41]*y[32] + p[144]*y[18];
                ydot[29] = p[45]*y[27] + p[50]*y[34] - p[145]*y[29] - p[46]*y[10]*y[29] - p[48]*y[11]*y[29] + p[47]*y[33] + p[49]*y[34] + p[146]*y[18];
                ydot[30] = -p[33]*y[30] - p[147]*y[30] + p[31]*y[28]*y[6] - p[32]*y[30] + p[148]*y[18];
                ydot[31] = -p[39]*y[31] - p[149]*y[31] + p[37]*y[28]*y[7] - p[38]*y[31] + p[150]*y[18];
                ydot[32] = -p[42]*y[32] - p[151]*y[32] + p[40]*y[28]*y[8] - p[41]*y[32] + p[152]*y[18];
                ydot[33] = -p[153]*y[33] + p[46]*y[10]*y[29] - p[47]*y[33] + p[154]*y[18];
                ydot[34] = -p[50]*y[34] - p[155]*y[34] + p[48]*y[11]*y[29] - p[49]*y[34] + p[156]*y[18];
                ydot[35] = p[33]*y[30] + p[36]*y[39] - p[157]*y[35] - p[34]*y[3]*y[35] + p[35]*y[39] + p[158]*y[18];
                ydot[36] = p[39]*y[31] - p[91]*y[36] + p[159]*y[18];
                ydot[37] = p[42]*y[32] - p[160]*y[37] + p[161]*y[18];
                ydot[38] = p[50]*y[34] - p[162]*y[38] - p[51]*y[38] + p[52]*y[40] + p[163]*y[18];
                ydot[39] = -p[36]*y[39] - p[164]*y[39] + p[34]*y[3]*y[35] - p[35]*y[39] + p[165]*y[18];
                ydot[40] = -p[166]*y[40] + p[51]*y[38] - p[53]*y[12]*y[40] - 1.0*p[55]*pow(y[40], 2) - p[52]*y[40] + p[54]*y[41] + 2*p[56]*y[42] + p[167]*y[18];
                ydot[41] = -p[168]*y[41] + p[53]*y[12]*y[40] - p[54]*y[41] + p[169]*y[18];
                ydot[42] = -p[170]*y[42] + 0.5*p[55]*pow(y[40], 2) - p[57]*y[12]*y[42] - 1.0*p[59]*pow(y[42], 2) - p[56]*y[42] + p[58]*y[43] + 2*p[60]*y[44] + p[171]*y[18];
                ydot[43] = -p[172]*y[43] + p[57]*y[12]*y[42] - p[58]*y[43] + p[173]*y[18];
                ydot[44] = -p[174]*y[44] + 0.5*p[59]*pow(y[42], 2) - p[61]*y[12]*y[44] - p[63]*y[13]*y[44] - p[60]*y[44] + p[62]*y[45] + p[64]*y[46] + p[175]*y[18];
                ydot[45] = -p[176]*y[45] + p[61]*y[12]*y[44] - p[62]*y[45] + p[177]*y[18];
                ydot[46] = -p[65]*y[46] - p[178]*y[46] + p[63]*y[13]*y[44] - p[64]*y[46] + p[179]*y[18];
                ydot[47] = p[65]*y[46] + p[68]*y[48] + p[71]*y[49] - p[90]*y[47] - p[66]*y[14]*y[47] - p[69]*y[15]*y[47] + p[67]*y[48] + p[70]*y[49] + p[180]*y[18];
                ydot[48] = -p[68]*y[48] - p[181]*y[48] + p[66]*y[14]*y[47] - p[67]*y[48] + p[182]*y[18];
                ydot[49] = -p[71]*y[49] - p[183]*y[49] + p[69]*y[15]*y[47] - p[70]*y[49] + p[184]*y[18];
                ydot[50] = p[68]*y[48] - p[185]*y[50] - p[72]*y[50] + p[73]*y[52] + p[186]*y[18];
                ydot[51] = p[71]*y[49] - p[187]*y[51] - p[82]*y[51] + p[83]*y[53] + p[188]*y[18];
                ydot[52] = p[76]*y[54] - p[189]*y[52] + p[72]*y[50] - p[74]*y[16]*y[52] - p[73]*y[52] + p[75]*y[54] + p[190]*y[18];
                ydot[53] = -p[191]*y[53] + p[82]*y[51] - p[86]*y[53]*y[7] - p[83]*y[53] + p[87]*y[55] + p[192]*y[18];
                ydot[54] = -p[76]*y[54] - p[193]*y[54] + p[74]*y[16]*y[52] - p[75]*y[54] + p[194]*y[18];
                ydot[55] = -p[195]*y[55] + p[86]*y[53]*y[7] - p[87]*y[55] + p[196]*y[18];
                ydot[56] = p[76]*y[54] - p[197]*y[56] - p[77]*y[17]*y[56] + p[78]*y[57] + p[198]*y[18];
                ydot[57] = p[81]*y[58] - p[199]*y[57] + p[77]*y[17]*y[56] - p[79]*y[5]*y[57] - p[84]*y[57]*y[7] - p[78]*y[57] + p[80]*y[58] + p[85]*y[59] + p[200]*y[18];
                ydot[58] = -p[81]*y[58] - p[201]*y[58] + p[79]*y[5]*y[57] - p[80]*y[58] + p[202]*y[18];
                ydot[59] = -p[203]*y[59] + p[84]*y[57]*y[7] - p[85]*y[59] + p[204]*y[18];
                ''', ['ydot', 't', 'y', 'p'])
            return ydot
        
    else:
        
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            ydot[0] = -p[93]*y[0] - p[18]*y[0]*y[1] + p[88]*y[21] + p[19]*y[19] + p[92]*y[18]
            ydot[1] = -p[94]*y[1] - p[18]*y[0]*y[1] + p[88]*y[21] + p[19]*y[19] + p[95]*y[18]
            ydot[2] = -p[96]*y[2] - p[21]*y[2]*y[21] + p[22]*y[22] + p[97]*y[18]
            ydot[3] = -p[98]*y[3] - p[23]*y[21]*y[3] - p[34]*y[3]*y[35] + p[24]*y[23] + p[35]*y[39] + p[99]*y[18]
            ydot[4] = -p[100]*y[4] - p[26]*y[24]*y[4] + p[27]*y[25] + p[101]*y[18]
            ydot[5] = -p[102]*y[5] - p[79]*y[5]*y[57] - p[28]*y[24]*y[5] + p[80]*y[58] + p[29]*y[26] + p[103]*y[18]
            ydot[6] = -p[104]*y[6] - p[31]*y[28]*y[6] + p[32]*y[30] + p[105]*y[18]
            ydot[7] = p[39]*y[31] - p[106]*y[7] - p[84]*y[57]*y[7] - p[86]*y[53]*y[7] - p[37]*y[28]*y[7] + p[85]*y[59] + p[87]*y[55] + p[38]*y[31] + p[107]*y[18]
            ydot[8] = -p[108]*y[8] - p[40]*y[28]*y[8] + p[41]*y[32] + p[109]*y[18]
            ydot[9] = -p[110]*y[9] - p[43]*y[24]*y[9] + p[44]*y[27] + p[111]*y[18]
            ydot[10] = -p[89]*y[10] - p[46]*y[10]*y[29] + p[47]*y[33] + p[112]*y[18]
            ydot[11] = -p[113]*y[11] - p[48]*y[11]*y[29] + p[49]*y[34] + p[114]*y[18]
            ydot[12] = -p[115]*y[12] - p[53]*y[12]*y[40] - p[57]*y[12]*y[42] - p[61]*y[12]*y[44] + p[54]*y[41] + p[58]*y[43] + p[62]*y[45] + p[116]*y[18]
            ydot[13] = p[90]*y[47] - p[117]*y[13] - p[63]*y[13]*y[44] + p[64]*y[46] + p[118]*y[18]
            ydot[14] = -p[119]*y[14] - p[66]*y[14]*y[47] + p[67]*y[48] + p[120]*y[18]
            ydot[15] = -p[121]*y[15] - p[69]*y[15]*y[47] + p[70]*y[49] + p[122]*y[18]
            ydot[16] = -p[123]*y[16] - p[74]*y[16]*y[52] + p[75]*y[54] + p[124]*y[18]
            ydot[17] = -p[125]*y[17] - p[77]*y[17]*y[56] + p[78]*y[57] + p[126]*y[18]
            ydot[18] = 0
            ydot[19] = -p[20]*y[19] - p[127]*y[19] + p[18]*y[0]*y[1] - p[19]*y[19] + p[128]*y[18]
            ydot[20] = p[185]*y[50] + p[181]*y[48] + p[183]*y[49] + p[187]*y[51] + p[123]*y[16] + p[193]*y[54] + p[199]*y[57] + p[203]*y[59] + p[201]*y[58] + p[100]*y[4] + p[137]*y[25] + p[113]*y[11] + p[170]*y[42] + p[172]*y[43] + p[174]*y[44] + p[176]*y[45] + p[178]*y[46] + p[155]*y[34] + p[115]*y[12] + p[168]*y[41] + p[110]*y[9] + p[141]*y[27] + p[143]*y[28] + p[151]*y[32] + p[91]*y[36] + p[149]*y[31] + p[147]*y[30] + p[157]*y[35] + p[164]*y[39] + p[135]*y[24] + p[139]*y[26] + p[160]*y[37] + p[131]*y[22] + p[133]*y[23] + p[93]*y[0] + p[127]*y[19] + p[166]*y[40] + p[89]*y[10] + p[153]*y[33] + p[117]*y[13] + p[108]*y[8] + p[106]*y[7] + p[195]*y[55] + p[197]*y[56] + p[162]*y[38] + p[189]*y[52] + p[191]*y[53] + p[96]*y[2] + p[119]*y[14] + p[121]*y[15] + p[102]*y[5] + p[104]*y[6] + p[98]*y[3] + p[125]*y[17] + p[94]*y[1] + p[145]*y[29]
            ydot[21] = p[20]*y[19] + p[25]*y[23] - p[21]*y[2]*y[21] - p[23]*y[21]*y[3] - p[88]*y[21] + p[22]*y[22] + p[24]*y[23] + p[130]*y[18]
            ydot[22] = -p[131]*y[22] + p[21]*y[2]*y[21] - p[22]*y[22] + p[132]*y[18]
            ydot[23] = -p[25]*y[23] - p[133]*y[23] + p[23]*y[21]*y[3] - p[24]*y[23] + p[134]*y[18]
            ydot[24] = p[45]*y[27] + p[25]*y[23] + p[30]*y[26] + p[36]*y[39] - p[135]*y[24] - p[43]*y[24]*y[9] - p[26]*y[24]*y[4] - p[28]*y[24]*y[5] + p[44]*y[27] + p[27]*y[25] + p[29]*y[26] + p[136]*y[18]
            ydot[25] = -p[137]*y[25] + p[26]*y[24]*y[4] - p[27]*y[25] + p[138]*y[18]
            ydot[26] = -p[30]*y[26] - p[139]*y[26] + p[28]*y[24]*y[5] - p[29]*y[26] + p[140]*y[18]
            ydot[27] = -p[45]*y[27] - p[141]*y[27] + p[43]*y[24]*y[9] - p[44]*y[27] + p[142]*y[18]
            ydot[28] = p[81]*y[58] + p[30]*y[26] + p[33]*y[30] + p[42]*y[32] - p[143]*y[28] - p[31]*y[28]*y[6] - p[37]*y[28]*y[7] - p[40]*y[28]*y[8] + p[32]*y[30] + p[38]*y[31] + p[41]*y[32] + p[144]*y[18]
            ydot[29] = p[45]*y[27] + p[50]*y[34] - p[145]*y[29] - p[46]*y[10]*y[29] - p[48]*y[11]*y[29] + p[47]*y[33] + p[49]*y[34] + p[146]*y[18]
            ydot[30] = -p[33]*y[30] - p[147]*y[30] + p[31]*y[28]*y[6] - p[32]*y[30] + p[148]*y[18]
            ydot[31] = -p[39]*y[31] - p[149]*y[31] + p[37]*y[28]*y[7] - p[38]*y[31] + p[150]*y[18]
            ydot[32] = -p[42]*y[32] - p[151]*y[32] + p[40]*y[28]*y[8] - p[41]*y[32] + p[152]*y[18]
            ydot[33] = -p[153]*y[33] + p[46]*y[10]*y[29] - p[47]*y[33] + p[154]*y[18]
            ydot[34] = -p[50]*y[34] - p[155]*y[34] + p[48]*y[11]*y[29] - p[49]*y[34] + p[156]*y[18]
            ydot[35] = p[33]*y[30] + p[36]*y[39] - p[157]*y[35] - p[34]*y[3]*y[35] + p[35]*y[39] + p[158]*y[18]
            ydot[36] = p[39]*y[31] - p[91]*y[36] + p[159]*y[18]
            ydot[37] = p[42]*y[32] - p[160]*y[37] + p[161]*y[18]
            ydot[38] = p[50]*y[34] - p[162]*y[38] - p[51]*y[38] + p[52]*y[40] + p[163]*y[18]
            ydot[39] = -p[36]*y[39] - p[164]*y[39] + p[34]*y[3]*y[35] - p[35]*y[39] + p[165]*y[18]
            ydot[40] = -p[166]*y[40] + p[51]*y[38] - p[53]*y[12]*y[40] - 1.0*p[55]*pow(y[40], 2) - p[52]*y[40] + p[54]*y[41] + 2*p[56]*y[42] + p[167]*y[18]
            ydot[41] = -p[168]*y[41] + p[53]*y[12]*y[40] - p[54]*y[41] + p[169]*y[18]
            ydot[42] = -p[170]*y[42] + 0.5*p[55]*pow(y[40], 2) - p[57]*y[12]*y[42] - 1.0*p[59]*pow(y[42], 2) - p[56]*y[42] + p[58]*y[43] + 2*p[60]*y[44] + p[171]*y[18]
            ydot[43] = -p[172]*y[43] + p[57]*y[12]*y[42] - p[58]*y[43] + p[173]*y[18]
            ydot[44] = -p[174]*y[44] + 0.5*p[59]*pow(y[42], 2) - p[61]*y[12]*y[44] - p[63]*y[13]*y[44] - p[60]*y[44] + p[62]*y[45] + p[64]*y[46] + p[175]*y[18]
            ydot[45] = -p[176]*y[45] + p[61]*y[12]*y[44] - p[62]*y[45] + p[177]*y[18]
            ydot[46] = -p[65]*y[46] - p[178]*y[46] + p[63]*y[13]*y[44] - p[64]*y[46] + p[179]*y[18]
            ydot[47] = p[65]*y[46] + p[68]*y[48] + p[71]*y[49] - p[90]*y[47] - p[66]*y[14]*y[47] - p[69]*y[15]*y[47] + p[67]*y[48] + p[70]*y[49] + p[180]*y[18]
            ydot[48] = -p[68]*y[48] - p[181]*y[48] + p[66]*y[14]*y[47] - p[67]*y[48] + p[182]*y[18]
            ydot[49] = -p[71]*y[49] - p[183]*y[49] + p[69]*y[15]*y[47] - p[70]*y[49] + p[184]*y[18]
            ydot[50] = p[68]*y[48] - p[185]*y[50] - p[72]*y[50] + p[73]*y[52] + p[186]*y[18]
            ydot[51] = p[71]*y[49] - p[187]*y[51] - p[82]*y[51] + p[83]*y[53] + p[188]*y[18]
            ydot[52] = p[76]*y[54] - p[189]*y[52] + p[72]*y[50] - p[74]*y[16]*y[52] - p[73]*y[52] + p[75]*y[54] + p[190]*y[18]
            ydot[53] = -p[191]*y[53] + p[82]*y[51] - p[86]*y[53]*y[7] - p[83]*y[53] + p[87]*y[55] + p[192]*y[18]
            ydot[54] = -p[76]*y[54] - p[193]*y[54] + p[74]*y[16]*y[52] - p[75]*y[54] + p[194]*y[18]
            ydot[55] = -p[195]*y[55] + p[86]*y[53]*y[7] - p[87]*y[55] + p[196]*y[18]
            ydot[56] = p[76]*y[54] - p[197]*y[56] - p[77]*y[17]*y[56] + p[78]*y[57] + p[198]*y[18]
            ydot[57] = p[81]*y[58] - p[199]*y[57] + p[77]*y[17]*y[56] - p[79]*y[5]*y[57] - p[84]*y[57]*y[7] - p[78]*y[57] + p[80]*y[58] + p[85]*y[59] + p[200]*y[18]
            ydot[58] = -p[81]*y[58] - p[201]*y[58] + p[79]*y[5]*y[57] - p[80]*y[58] + p[202]*y[18]
            ydot[59] = -p[203]*y[59] + p[84]*y[57]*y[7] - p[85]*y[59] + p[204]*y[18]
            return ydot
        
    
    def simulate(self, tspan, param_values=None, view=False):
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.parameters):
                raise Exception("param_values must have length %d" % len(self.parameters))
            self.sim_param_values[:] = param_values
        else:
            # create parameter vector from the values in the model
            self.sim_param_values[:] = [p.value for p in self.parameters]
        self.y0.fill(0)
        for ic in self.initial_conditions:
            self.y0[ic.species_index] = self.sim_param_values[ic.param_index]
        if self.y is None or len(tspan) != len(self.y):
            self.y = numpy.empty((len(tspan), len(self.y0)))
            if len(self.observables):
                self.yobs = numpy.ndarray(len(tspan), zip((obs.name for obs in self.observables),
                                                          itertools.repeat(float)))
            else:
                self.yobs = numpy.ndarray((len(tspan), 0))
            self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        # perform the actual integration
        self.integrator.set_initial_value(self.y0, tspan[0])
        self.integrator.set_f_params(self.sim_param_values)
        self.y[0] = self.y0
        t = 1
        while self.integrator.successful() and self.integrator.t < tspan[-1]:
            self.y[t] = self.integrator.integrate(tspan[t])
            t += 1
        for i, obs in enumerate(self.observables):
            self.yobs_view[:, i] = \
                (self.y[:, obs.species] * obs.coefficients).sum(1)
        if view:
            y_out = self.y.view()
            yobs_out = self.yobs.view()
            for a in y_out, yobs_out:
                a.flags.writeable = False
        else:
            y_out = self.y.copy()
            yobs_out = self.yobs.copy()
        return (y_out, yobs_out)
    

