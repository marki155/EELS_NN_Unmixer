import copy
import glob
import os
import time


import hyperspy.api as hs
import pandas as pd
import tensorflow as tf
import copy

from utility import make_dir
from conv_net_unmixer import ConvNetUnmixer
from spectral_unmixer_mixed import SpectralUnmixer
from multiprocessing import Process



def p_alive(p_list):
    counter = 0
    for p in p_list:
        if p.is_alive():
            counter += 1
    return counter


def task(N, data_pack, R, learning_rate, end_act, layer_activation, layer_units, ab_rate, neg_rate, dropout, batch_size, epochs, tqdm_callback=None):
    NN = N
    curr_epoch = 0
    end_act_name, end_activation = end_act
    MAIN_DIR_sample, PREPROCESS_MAIN_DIR, unmixer, fusion_data_binned, save_name = data_pack
    nn = ConvNetUnmixer(fusion_data_binned, fusion_data_binned, save_name)
    nn.build_model(R=R, g=layer_activation, end_act=end_activation,
                   learning_rate=learning_rate,
                   dropout_rate=dropout, ab_rate=ab_rate, neg_rate=neg_rate, layer_units=layer_units)

    for epoch in epochs:
        if tqdm_callback is None:
            callbacks = []
        else:
            callbacks = tqdm_callback
        loss, err = nn.training(epochs=epoch, batch_size=batch_size, all_callbacks=callbacks)
        curr_epoch += epoch

        #N = len(df.index)
        saver = "{},{},{},{},{},{},{},{},{},{}".format(R, learning_rate, end_act_name,
                                                    layer_activation, layer_units, ab_rate, neg_rate,
                                                    dropout, batch_size, curr_epoch)
        unmixer.unmixer = nn

        _, _, loss_T, err_T, cross_abundances = unmixer.predict(MAIN_DIR_sample,
                                                                save_path=PREPROCESS_MAIN_DIR + saver + "\\",
                                                                overview_image=PREPROCESS_MAIN_DIR + str(NN) + "_Result_" + saver + "_overview.png",
                                                                remove_eels_background=False, use_refs=False)
        spec_dist = nn.calc_endmember_distances()

        dd = {"R": R, "learning_rate": learning_rate, "end_activation": end_act_name,
              "layer_activation": layer_activation, "ab_rate": ab_rate, "neg_rate": neg_rate,
              "layer_units": str(layer_units),
              "dropout": dropout, "batch_size": batch_size, "epoch": curr_epoch,
              "loss": loss,
              "error": err, "loss_pred": loss_T, "err_pred": err_T, "adundance_diff": cross_abundances,
              "spec_dist": spec_dist}
        entry = pd.DataFrame(dd, index=[NN])

        df = pd.read_pickle(PREPROCESS_MAIN_DIR + "tuning.pkl")
        df = pd.concat([df, entry])
        df.to_pickle(PREPROCESS_MAIN_DIR + "tuning.pkl")
        NN += 1
    del nn


if __name__ == "__main__":
    ELEMENTS = ["Li"]

    COMP_NAMES_ALL = ["comp" + str(i) for i in range(10)]

    parameters = {"component_names": [""],
                  "omegas": [None, 1, None],
                  "end_eels_cl": ("custom", 52, 80)}

    unmixer = SpectralUnmixer(ELEMENTS, parameters)

    folder = "SI data (1)"
    print("-----------------------------")
    print(folder)
    print("-----------------------------")
    SUB_DIR = "SI data (1)\\"
    MAIN_DIR = "C:\\Users\\KoegelMarco\\Documents\\Ergebnisse\\DataShare\\PhD\\Li Unmixing Test\\"
    PREPROCESS_MAIN_DIR = MAIN_DIR + "preprocess_tuning_Li-edge_no-binning\\"
    make_dir(PREPROCESS_MAIN_DIR + SUB_DIR)

    sample = MAIN_DIR
    splits = sample.split("\\")
    #main = splits[-4] + "\\"
    sample = folder + "\\"
    print("PREPROCESS data file in " + sample)
    make_dir(PREPROCESS_MAIN_DIR + sample)


    unmixer.add_data(MAIN_DIR + sample, remove_eels_background=False, use_refs=False, binning_factor=1)


    unmixer.shuffle()



    Rs = [4, 5, 6, 7, 8, 9, 10, 11]#,7,8,9]
    batch_sizes = [20, 50, 100]#, 10, 300]
    epochs = [500, 500]#, 400] # additive
    learning_rates = [0.005]
    dropouts = [0.01]
    layer_activations = ["sigmoid"]
    end_activations = [("sig", tf.keras.activations.sigmoid)]
    ab_rates = [1e-2, 1e-1, 1, 10]
    neg_rates = [1e-1]
    layer_unitss = [[9, 6, 3, 1]]#, [16, 8, 4, 1]]



    if not os.path.isfile(PREPROCESS_MAIN_DIR + "tuning.pkl") :
        df = pd.DataFrame(columns=["R", "learning_rate", "end_activation",
                                              "layer_activation", "ab_rate", "neg_rate", "layer_units",
                                              "dropout", "batch_size", "epoch",
                                              "loss",
                                              "error", "loss_pred", "err_pred", "adundance_diff", "spec_dist"])
        df.to_pickle(PREPROCESS_MAIN_DIR + "tuning.pkl")
        N_start = 0
    else:
        df = pd.read_pickle(PREPROCESS_MAIN_DIR + "tuning.pkl")
        N_start = len(df.index) + 1

    N_curr = N_start

    save_name=None
    #comp_names = parameters["component_names"]
    fusion_data_binned = unmixer.data_collection

    pre_predicter = unmixer.make_predicter()
    pre_predicter.stuff(use_refs=False, PATH=MAIN_DIR + sample, remove_eels_background=False)
    predicter = pre_predicter.make_predicter()
    p_list = []

    for R in Rs:
        predicter = copy.copy(predicter)
        parameters = {"component_names": COMP_NAMES_ALL[:R],
                      "omegas": [None, 1, None],
                      "end_eels_cl": ("custom", 52,80)}
        predicter.parameters = parameters
        for learning_rate in learning_rates:
            for end_act_name, end_activation in end_activations:
                for layer_activation in layer_activations:
                    for ab_rate in ab_rates:
                        for neg_rate in neg_rates:
                            for layer_units in layer_unitss:
                                for dropout in dropouts:
                                    for batch_size in batch_sizes:

                                        data_pack = (MAIN_DIR + sample, PREPROCESS_MAIN_DIR, predicter, fusion_data_binned, save_name)
                                        process = Process(target=task, args=(N_curr, data_pack, R, learning_rate, (end_act_name, end_activation),
                                                            layer_activation, layer_units, ab_rate, neg_rate,
                                                            dropout, batch_size, epochs,))
                                        p_list.append(process)
                                        N_curr += len(epochs)
                                        #process.start()


    for p in p_list:
        while p_alive(p_list) >= 4:
            time.sleep(3)
        p.start()
        print("process " + p.name + " started")
        print(str(p_alive(p_list)) + " processes active")
        #print("process finished")