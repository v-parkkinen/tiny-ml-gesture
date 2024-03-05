// weights.c
#include "weights.h"

const float lstmIxWeights[60] =  {
	0.43965396, 0.47559077, 0.69509387, 0.4423266, -0.42232025, 0.07028187,
	0.22728375, 0.14091884, -0.15634318, -0.337196, -0.2673805, -0.11738481,
	-0.12262166, -1.2072598, 1.611636, -1.787389, 2.0912087, 0.6040567,
	0.1684131, 0.39664719, -0.23882768, 0.23432311, 0.2016639, 1.0307496,
	0.11760235, 0.28191164, 0.52992254, 0.0018549277, 0.47652274, 0.13112551,
	-0.27026325, 0.5472681, -0.21899678, -0.39778978, 0.17108138, -0.53280514,
	0.82363164, -0.07236903, 0.5292191, -0.05029857, 0.24231346, 0.057234295,
	-1.4710636, -0.38895023, -0.6585408, 1.1192017, 0.52386045, 0.57578367,
	-0.8323637, -0.15875734, -1.3786144, 0.7098528, 0.4973351, 1.1319013,
	0.14892818, -0.004936642, 0.5325055, 0.06328532, 0.10429661, 0.086079836,
};

const float lstmFxWeights[60] =  {
	0.0328486, -0.2841842, -0.112918645, -0.2601974, 0.7297854, 0.24329019,
	0.2422807, -0.35134315, -0.25349072, 0.011229498, 0.121656455, -0.64461523,
	-0.5091318, 0.17160113, -0.7668856, 0.52186996, 0.39309156, -0.25562716,
	0.5017784, 0.18475215, 0.114735894, 0.17742877, 0.37692732, 0.40807828,
	-0.21844223, -0.029596094, 0.3132249, 0.2135463, -0.4175031, -0.07525957,
	-0.023764348, 0.06511904, -0.04942755, -0.59933484, -0.4097763, -0.31890523,
	0.4341163, -0.20594802, 0.5318456, 0.17240842, 0.4763246, 0.069145136,
	0.3012623, 0.35813457, 0.47470942, 0.09124264, 0.007705075, 0.25806087,
	0.062432658, 0.13489784, -0.23901069, -0.4595085, -0.15502891, 0.42191613,
	-0.025682496, 0.02007595, -0.16276918, -0.30441687, 0.13506655, 0.034254555,
};

const float lstmCxWeights[60] =  {
	-0.08788545, -0.1282722, 0.32726377, -0.214763, 0.5350797, -0.08164684,
	-0.10794756, -0.3260845, 0.18905032, 0.28986385, -0.41405445, 0.20473662,
	0.17820117, 0.14762911, -0.5854956, -0.02257999, 0.51540613, -0.28565097,
	0.047431447, -0.45987707, 0.5708667, 0.22832413, -0.057391934, -0.2994082,
	0.031339586, 0.03480517, -0.51704836, -0.22110394, -0.24349791, 0.056664076,
	-0.29365745, 0.3870997, 0.111456946, 0.09076977, -0.4340704, -0.13812548,
	0.12696733, -0.074774936, 0.33827114, -0.25518548, -0.25203374, 0.030662196,
	-0.23835537, -0.28326374, -0.2866637, 0.40312287, 0.21336262, -0.11567502,
	0.3306459, 0.19197106, 0.25241297, -0.22121309, 0.11641898, -0.32070607,
	0.08356104, -0.2835451, -0.04644113, 0.029088493, 0.38972715, -0.20487812,
};

const float lstmOxWeights[60] =  {
	0.8273176, 0.37921348, 0.48903024, 0.40455544, 0.8538869, 0.5950067,
	0.48827043, 0.3544749, 0.55287975, -0.28772455, 0.40303722, -0.070115864,
	0.13807404, 0.57088983, 0.42226058, 0.04121788, 1.1181219, 0.7052599,
	0.17630722, 0.20675561, -0.33273417, 0.73827857, 0.053995796, 0.72686553,
	-0.08127929, 0.3419906, 0.4831055, 0.56092983, 0.051491234, -0.046821248,
	0.31487897, 0.41942704, 0.55173063, 0.4490584, 0.4750696, 0.009493106,
	0.51786727, 0.468441, 0.50190014, 0.59794277, 0.7654834, 0.4460939,
	-0.036955476, 0.80093586, 0.3316873, 0.24213488, 0.46462086, 0.8207914,
	0.13150008, 0.70643854, -0.27588585, 0.04027075, 0.5902691, 1.0919068,
	0.30857462, -0.10744411, -0.013215093, -0.09812185, 0.10478987, 0.3178348,
};

const float lstmIhWeights[100] =  {
	-0.026632877, -1.1276668, -0.6844141, 0.5496217, -0.5458459, 0.1202477, 0.7374089, -0.8875149, -0.0066115325, 0.82982177,
	0.60375535, -0.7295035, -1.1017334, 0.50175196, -0.12198326, 0.7646216, 0.3785622, -0.18393332, -0.9231097, 0.079263076,
	-2.0710714, -1.0802283, 2.4615786, -0.62708336, 0.5078054, 1.3942143, -0.055650968, -0.22225285, 0.5754334, 0.035872232,
	-0.13068682, 0.0009460865, 0.78981453, 0.5152808, -0.79560614, -0.4666304, 0.11540856, 0.3969605, 0.33097482, 1.0890126,
	0.16834009, -0.42771912, 0.41865733, -0.6079555, -0.2739364, -1.380394, -0.04689636, -0.75447834, 0.8190516, -0.5009806,
	-0.39650926, -1.6381489, -0.046983823, 0.3096363, -1.2404512, -0.53722906, 1.9588003, -0.7295155, 0.39790446, 0.2102918,
	-0.26463908, 0.45611832, 0.5342677, -0.20574714, 0.14262992, -0.8260613, 0.22348236, 0.12621239, 0.350476, -0.5704923,
	0.49378836, -0.4436001, 0.367731, -0.07718632, -0.18841773, -1.4299605, 0.29861262, -0.8077052, 1.2916713, -0.25495255,
	-0.47870466, -0.1741363, 0.7088969, -0.08263372, -0.39090917, 0.41223508, 0.4570172, -0.50650513, 0.3825946, -0.16876708,
	-0.58299553, -0.27507278, 1.3033653, 0.11461291, -0.42713955, -0.9722952, 0.43934515, -0.26462996, 1.0366386, -0.2928399,
};

const float lstmFhWeights[100] =  {
	-0.4195019, -0.75471383, -0.3868568, -0.027233912, -0.1323355, 0.63083243, 0.13735071, -0.61591786, -0.58368844, 0.094163865,
	0.7074776, -0.42424187, -1.6490195, 0.15201394, 0.08431933, 0.8858149, 0.45893383, -0.49520755, -1.1171489, -0.2507646,
	0.4360335, -1.7442385, 0.41813213, -0.25204512, -0.37965664, -1.5122836, 0.9573789, -1.0271188, 0.46355373, 0.9215596,
	0.22154754, -0.5715017, 0.46260282, -0.057016987, -0.6031876, -1.132571, 0.66951865, -0.019513201, 0.44569343, 0.60254806,
	0.32687736, -0.123024076, -0.08235629, 0.5426991, -0.24820967, 0.2069301, 0.021852221, -0.2331604, 0.03754573, 0.26589224,
	-0.4092683, -1.2354683, -0.14980641, -0.49429175, -0.246263, -0.4458709, 1.034864, -0.091686144, 0.055840883, 0.43953568,
	-0.03484419, 0.44415897, 0.116347246, 0.09239594, 0.031212633, 0.2239915, 0.3289725, -0.47801384, -0.31088755, 0.8516881,
	0.86333287, -0.07194049, 0.10561666, 0.28847367, -0.32898, -0.7184405, 0.36945474, -0.3480034, 0.26680478, -0.027508108,
	0.35742044, -0.6752348, 0.968208, -0.20645687, -0.31653234, -0.618865, 0.06740541, 0.06362126, 0.090527385, 0.22619341,
	-0.21062694, 0.5674184, 2.193504, -0.23965657, 0.016957734, -1.5123875, -0.052206535, 0.20453882, 1.38006, 0.32732904,
};

const float lstmChWeights[100] =  {
	0.38752913, -0.018273944, -0.20036626, -0.1755863, 0.09891274, -0.27506527, -0.03665865, 0.19812952, -0.1722558, 0.048319817,
	0.20929757, 0.23578244, 0.3444208, -0.109004684, -0.047302045, -0.1910108, -0.4488206, -0.026247194, 0.49441788, -0.022601936,
	0.07253188, 0.23754963, 0.50594735, 0.19173692, -0.11568351, -0.09250567, 0.19004177, -0.1587484, 0.0026353758, -0.065610796,
	0.041360375, -0.0461984, -0.41819826, 0.27238202, -0.08922018, -0.4361396, 0.124648824, 0.11038764, -0.010887369, 0.53991467,
	-0.71440065, 0.21823786, 0.35883912, -0.36691388, -0.04600015, -0.47778365, -0.027341956, 0.7546463, -0.09963419, -0.27120626,
	-0.07349484, 0.4160925, -0.9025017, 0.23945424, -0.26180658, 0.32799447, -0.24550374, -0.36715394, -0.026046578, 0.09249222,
	0.08137363, 0.10328532, -0.19355863, -0.10450659, -0.59804684, 0.46944252, 0.2632117, -0.55778366, -0.17249598, -0.044251394,
	0.06872027, -0.17905092, 0.18857503, -0.21539885, -0.07989759, -0.44115457, 0.23145974, 0.6612503, -0.29719266, 0.12338465,
	-0.042173926, 0.11544053, 0.986131, 0.00010096688, 0.121180944, -0.34581277, 0.13243632, 0.10229056, 0.152692, 0.0087883305,
	-0.4064169, 0.25031224, 0.19039571, 0.3115665, -0.027436037, 0.0053862343, 0.021151267, 0.6102761, 0.20134036, 0.60593,
};

const float lstmOhWeights[100] =  {
	0.11664009, -0.9553003, -0.75481296, 0.8247659, -0.49892798, -0.19793105, 0.9661773, -0.552287, -0.25508237, 0.5726966,
	0.6753, -0.9175064, -0.82993335, 0.44233418, -0.2372429, 0.29407668, 0.6147393, -0.7409263, -1.0545758, 0.2027206,
	-0.26540872, -1.5777198, 0.7659384, 0.02510476, -0.8235958, -0.5711707, 1.4475224, -0.75960404, 0.59024787, 0.6479946,
	0.23720023, 0.21918061, 0.6374895, -0.020519897, -0.43517518, -0.91496444, 0.23551181, 1.0076506, 0.5894269, 0.63940966,
	-0.9607456, 0.19192019, -0.40844405, -0.19624986, -0.23089424, 0.02781661, 0.0371752, -0.65491456, 1.1413417, -0.4306535,
	-0.111995675, -2.0823393, 0.4911032, -0.19920768, -1.0812359, -0.50899124, 1.4810964, -0.52179646, 0.38075337, 0.1612977,
	0.1598754, -0.38470182, 0.7181372, 0.060683236, -0.39304015, -0.78560543, 0.68077046, 0.09944561, 0.6436247, 0.40695962,
	1.4010828, -0.4544612, 0.0005935238, 0.19068222, -0.112665676, -1.7099881, 0.25382286, -0.41654763, 1.0033301, -0.25318846,
	-0.29859272, -0.99597687, 1.0792366, -0.094331115, -0.5068735, -0.5236026, 0.39217627, -0.6247829, 0.7004015, 0.090398565,
	-0.1511121, 0.12851113, 2.1370645, 0.14340022, -0.33012423, -1.1888568, 0.3746724, 0.09694301, 1.8273826, 0.14378427,
};

const float lstmIBiases[10] =  {
	0.28857303, 0.026164437, -0.7898776, 0.173426, 0.18269718, 0.021516686, 0.2161042, -0.2621396, -0.38048637, 0.21796171
};

const float lstmFBiases[10] =  {
	0.98431987, 0.9414051, 0.9734748, 1.4259613, 1.0979005, 0.9319043, 0.9858405, 1.198943, 1.1014191, 0.96414953
};

const float lstmCBiases[10] =  {
	-0.048795722, -0.04002289, -0.04997169, -0.040592737, -0.13011214, 0.012730412, 0.021609394, 0.0038579535, -0.013268159, 0.026937002
};

const float lstmOBiases[10] =  {
	0.48437738, 0.20588754, 0.36976114, 0.3116437, 0.090299994, 0.5340924, 0.510798, 0.20219705, 0.09862397, 0.25589007
};


const float denseLayerWeights[50] =  {
	0.29967806, -0.63463724, 0.48571938, -0.56329376, -0.29211962, -0.3572412, 0.44024512, -1.8896052, 0.6336459, -0.9999278,
	-0.009298813, 2.137521, 0.9188927, -1.7734332, 1.3229415, -0.9621918, -1.764653, 2.176186, 0.818727, -0.7706533,
	-0.9661794, -0.33369663, -2.59372, 0.14801098, -0.81929815, 3.3395102, 0.4322836, -0.4540193, -2.119241, -0.16766553,
	2.077508, -0.66999775, -1.7421995, 1.0864286, -0.1587485, -0.46537605, 0.56803334, 1.7178185, -1.5218928, 0.8726441,
	-0.66785353, 1.3701563, 1.1861542, 1.240917, -0.12745555, -0.3861306, 0.029908128, 0.43190935, 1.2214818, 2.2842596,
};

const float denseLayerBiases[5] =  {
	-0.0064469296, 0.106943354, 0.1396327, 0.27801067, -0.3613181
};


