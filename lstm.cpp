#include <iostream>

#include "lstmCell.h"
#include "denseLayer.h"


int main() {
    int input_size =  6;
    int hidden_size =  10;
    int num_classes = 5;

    float input[60] = {
        0.4764710566127876, 0.31938762043024943, 0.6389980931754514, 0.5968458834687843, 0.5034561684596017, 0.4994125276569772,
        0.4768691774822836, 0.3164840966081563, 0.6377739588973375, 0.5974595971806363, 0.5040817883573663, 0.49910734721904326,
        0.4767099291344852, 0.3178038801636532, 0.636079003743026, 0.596604117461085, 0.5031967650873579, 0.4997177080949111,
        0.4761525599171908, 0.317012010030355, 0.6373973021963795, 0.5964181436090086, 0.5033951323720149, 0.49915312428473335,
        0.47599331156939245, 0.3181998152303022, 0.6372089738459004, 0.5970504547060683, 0.5040207522697795, 0.49948882276646067,
        0.47623218409109, 0.31806783687475254, 0.6373973021963795, 0.5968272860835766, 0.5036392767223621, 0.499320973525597,
        0.4760729357432916, 0.3174079450970041, 0.6371148096706608, 0.5965855200758773, 0.5034103913939116, 0.49950408178835737,
        0.476630304960586, 0.316616074963706, 0.6369264813201818, 0.597013259935653, 0.5035324635690852, 0.49916838330663005,
        0.47941715104705784, 0.3200475122079979, 0.6372089738459004, 0.5969574677800301, 0.5032883192187381, 0.49489585717555507,
        0.4306075324468508, 0.457041045268576, 0.6321241083829657, 0.5987800115303789, 0.4981765468833448, 0.44972915236133365
    };

    // Create an LSTMCell object
    LSTMCell lstmCell(input_size, hidden_size);
    DenseLayer denseLayer(input_size, num_classes);

    int seq_len = 10;
    for (int t = 0; t < seq_len; t++) {
        lstmCell.forward(input + t*input_size);
    }
    
    float output[num_classes];
    denseLayer.forward(lstmCell.h_t, output);

    for (int i = 0; i < num_classes; i++) {
        std::cout << output[i] << std::endl;
    }
    
    return 0;
}