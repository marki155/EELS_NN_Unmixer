import os

import hyperspy.api as hs

import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
import pandas as pd
from scipy.optimize import minimize
import numpy as np

from conv_net_unmixer import ConvNetUnmixer
#from qmv_nmf import NMF_QMV




def nmf_qmv_reconstruction_error(X, W, H):
    return np.linalg.norm(X - W @ H, 'fro')

def remove_eels_background_with_fitting(s_cl, s_ll):
    print("remove background via EELS fitting")
    m = s_cl.create_model(ll=s_ll)
    m.enable_fine_structure() # wichtig!
    m.enable_background()
    m.enable_edges()
    m.smart_fit()
    comps = m.components
    s_back = m.as_signal(component_list=[comps.PowerLaw])
    s_cl.data = s_cl.data - s_back.data
    if False:
        s.plot()
        plt.show()
    return s_cl
    #s0 = m.as_signal()

def remove_eels_background_with_fitting_data(raw_cl_eels, data, elements):
    empty_spec = raw_cl_eels.sum()
    empty_spec.data = data
    sample = empty_spec

    sample.add_elements(elements)

    print("remove background via EELS fitting")
    m = sample.create_model()
    m.enable_fine_structure() # wichtig!
    m.enable_background()
    m.enable_edges()
    m.smart_fit()
    comps = m.components
    s_back = m.as_signal(component_list=[comps.PowerLaw])
    return sample.data - s_back.data
    #s0 = m.as_signal()



def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def make_datafusion(datas, os, KRON, sample_binning=None):
    splits_i = []
    all_datas = []
    kron_vector = []
    position = 0
    for i, data in enumerate(datas):
        if os[i] is not None:
            if sample_binning is not None:
                binned = []
                for ii in range(data.shape[2]):
                    binned.append(rebin(data[:,:,ii], sample_binning))
                data_binned = np.dstack(binned)
            else:
                data_binned = data

            flat_data = data_binned.reshape(-1, data.shape[-1])
            flat_data[np.where(flat_data < 0.0)] = 0.0


            splits_i.append(position + flat_data.shape[1])
            position += flat_data.shape[1]

            #theta = np.linalg.norm(flat_data) ** 2
            #theta = np.median(np.max(flat_data, axis=1))
            theta = np.max(flat_data)
            if KRON:
                all_datas.append(flat_data)
                kron_vector.append(os[i] / theta)
            else:
                all_datas.append((os[i] / theta) * flat_data)

    data_vector = np.concatenate(tuple(all_datas), axis=1)

    if KRON:
        data_vector = np.kron(kron_vector, data_vector)
        return data_vector, splits_i, kron_vector
    else:
        return data_vector, splits_i, None

    #flat_combination = np.multiply(kron_vector, data_vector)

def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        #print("The new directory is created!")

def unkron(kron_vec, data):
    divider = data.shape[1] // len(kron_vec)
    vecs = []

    for div in range(len(kron_vec)):
        vec = []
        for c in range(data.shape[0]):
            vec.append(data[c,div*divider:(div+1)*divider] / kron_vec[div])
        vecs.append(vec)
    return vecs

def get_spektra_back(flat_data, splits_i):
    indv_datas = []
    last_split = 0
    for ind in splits_i:
        indv = flat_data[last_split:ind]
        last_split = ind
        indv_datas.append(indv)
    return indv_datas



def get_energy_axis(sig, e_axis=2):
    sig.create_model()
    ext = sig.axes_manager.signal_extent
    return np.linspace(ext[0], ext[1], sig.data.shape[e_axis])


def calc_cross_abundances(maps):
    remaining = np.sum(maps, axis=1) - np.max(maps, axis=1)
    return np.mean(remaining)


class SpectralUnmixer:
    def __init__(self, elements, parameters):
        self.elements = elements
        self.parameters = parameters
        self.data_collection = None


    def init_data_collection(self, E_len):
        self.data_collection = np.empty((0, E_len))

    def add_data(self, PATH, remove_eels_background=False, KRON=False, use_refs=True, binning_factor=4):
        self.use_refs = use_refs
        self.path = PATH
        self.KRON = KRON

        if use_refs:
            self.si3n4_cl = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                          spectrum_type="coreloss")[1]
            self.si3n4_ll = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                          spectrum_type="lowloss")[0]

            self.sio2_cl = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                         spectrum_type="coreloss")[0]  # TODO ?
            self.sio2_ll = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                         spectrum_type="lowloss")[0]
            self.si_cl = hs.datasets.eelsdb(title="Silicon",
                                            spectrum_type="coreloss")[2]



        if os.path.isfile(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4'):
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss) (aligned).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4')
        else:
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss).dm4')
            self.raw_ll_eels.align_zero_loss_peak(also_align=[self.raw_cl_eels])
        self.raw_cl_eels.add_elements(self.elements)
        if remove_eels_background:
            self.raw_cl_eels = remove_eels_background_with_fitting(self.raw_cl_eels, self.raw_ll_eels)
        self.raw_cl_eels.fourier_ratio_deconvolution(ll=self.raw_ll_eels, extrapolate_lowloss=False,
                                                     extrapolate_coreloss=False)
        self.raw_eds = hs.load(PATH + "EDS Spectrum Image.dm4")

        ####

        _os = self.parameters["omegas"]
        end_eels_cl, start_eels_cl_energy, end_eels_cl_energy = self.parameters["end_eels_cl"]
        self.define_axis(_os, end_eels_cl, start_eels_cl_energy, end_eels_cl_energy, use_refs=self.use_refs)

        #fusion_data, splits_i, kron_vec = make_datafusion([self.eels_ll_data, self.eels_data,
        #                                                   self.edx_data], _os, self.KRON)


        n_x, n_y, _ = self.eels_data.shape
        if binning_factor is not None:
            b_x = (n_x // binning_factor) * binning_factor
            b_y = (n_y // binning_factor) * binning_factor
            s_binning = (b_x // binning_factor, b_y // binning_factor)
        else:
            s_binning = None
            b_x = n_x
            b_y = n_y

        fusion_data_binned, _, _ = make_datafusion([self.eels_ll_data[:b_x, :b_y, :], self.eels_data[:b_x, :b_y, :],
                                                    self.edx_data[:b_x, :b_y, :]], _os, self.KRON,
                                                   sample_binning=s_binning)

        fusion_data_binned = fusion_data_binned#[:, self.E_start:self.E_cut]

        if self.data_collection is None:
            self.init_data_collection(fusion_data_binned.shape[1])

        if fusion_data_binned.shape[1] > self.data_collection.shape[1]:
            self.E_cut = self.data_collection.shape[1]
            fusion_data_binned = fusion_data_binned[:, :self.E_cut]
            self.raw_cl_eels_cutted = self.raw_cl_eels_cutted.isig[:self.E_cut]

        self.data_collection = np.concatenate([self.data_collection, fusion_data_binned])






    def define_axis(self, os, end_cl_eels="as_ref", start_cl_eels_energy=None, end_cl_eels_energy=None, use_refs=True):

        real_energies_eels = get_energy_axis(self.raw_cl_eels)
        real_energies_ll_eels = get_energy_axis(self.raw_ll_eels)

        ll_start = 5  # eV

        if use_refs:
            ref_energies_ll_eels_sio2 = get_energy_axis(self.sio2_ll, 0)
            ref_energies_ll_eels_si3n4 = get_energy_axis(self.si3n4_ll, 0)
            real_i_end_ll_eels = np.argmax(real_energies_ll_eels >= self.sio2_ll.axes_manager.signal_extent[1])
            real_i_start_eels = np.argmax(real_energies_eels >= self.si3n4_cl.axes_manager.signal_extent[0])

            real_i_start_eels = np.argmax(real_energies_eels >= 70)
        else:

            real_i_end_ll_eels = len(real_energies_ll_eels)
            real_i_start_eels = 0
            real_i_start_eels = np.argmax(real_energies_eels >= 70)




        real_i_start_ll_eels = np.argmax(real_energies_ll_eels >= ll_start)
        real_energies_ll_eels = real_energies_ll_eels[real_i_start_ll_eels:real_i_end_ll_eels]
        if use_refs:
            ref_i_start_ll_eels_sio2 = np.argmax(ref_energies_ll_eels_sio2 >= ll_start)
            ref_energies_ll_eels_sio2 = ref_energies_ll_eels_sio2[ref_i_start_ll_eels_sio2:]

            ref_i_start_ll_eels_si3n4 = np.argmax(ref_energies_ll_eels_si3n4 >= ll_start)
            ref_energies_ll_eels_si3n4 = ref_energies_ll_eels_si3n4[ref_i_start_ll_eels_si3n4:]

            ref_energies_eels_sio2 = get_energy_axis(self.sio2_cl, 0)
            ref_energies_eels_si3n4 = get_energy_axis(self.si3n4_cl, 0)
            ref_energies_eels_si = get_energy_axis(self.si_cl, 0)

        if end_cl_eels == "all":
            real_i_end_eels = 2000#-1
        elif end_cl_eels == "as_ref":
            real_i_end_eels = np.argmax(real_energies_eels >= self.si3n4_cl.axes_manager.signal_extent[1])
        elif end_cl_eels == "custom":
            real_i_end_eels = np.argmax(real_energies_eels >= end_cl_eels_energy)
            real_i_start_eels = np.argmax(real_energies_eels >= start_cl_eels_energy)

        self.E_cut = real_i_end_eels
        self.E_start = real_i_start_eels




        real_energies_eels = real_energies_eels[real_i_start_eels:real_i_end_eels]
        real_energies_eds = get_energy_axis(self.raw_eds)
        real_i_end_eds = 300
        real_energies_eds = real_energies_eds[:real_i_end_eds]


        self.edx_data = self.raw_eds.data[:, :, :real_i_end_eds]

        self.eels_data = self.raw_cl_eels.data[:, :, real_i_start_eels:real_i_end_eels]
        self.raw_cl_eels_cutted = self.raw_cl_eels.isig[real_i_start_eels:real_i_end_eels]
        self.eels_ll_data = self.raw_ll_eels.data[:, :, real_i_start_ll_eels:real_i_end_ll_eels]




        self.real_E_axis = []
        if os[0] is not None:
            self.real_E_axis.append(real_energies_ll_eels)
        if os[1] is not None:
            self.real_E_axis.append(real_energies_eels)
        if os[2] is not None:
            self.real_E_axis.append(real_energies_eds)

        # s.set_microscope_parameters(beam_energy=80,
        #                            convergence_angle=0.2,
        #                            collection_angle=2.55)

        if use_refs:
            sio2_ll_data = self.sio2_ll.data[ref_i_start_ll_eels_sio2:]
            si3n4_ll_data = self.si3n4_ll.data[ref_i_start_ll_eels_si3n4:]
            self.ref_data = [self.si_cl.data, self.si3n4_cl.data, self.sio2_cl.data, None]
            self.ref_axis = [ref_energies_eels_si, ref_energies_eels_si3n4, ref_energies_eels_sio2, None]
            self.ref_data_ll = [si3n4_ll_data, sio2_ll_data, None]
            self.ref_axis_ll = [ref_energies_ll_eels_si3n4, ref_energies_ll_eels_sio2, None]
            self.ref_names = ["Silicon ref", "Si3N4 ref", "SiO2 ref", None]
            self.ref_colors = ["g", "k", "r", None]
        else:
            self.ref_data = None



    def shuffle(self):
        np.random.shuffle(self.data_collection)



    def unmixing(self, parameters, save_path):


        comp_names = parameters["component_names"]



        #endmembers, maps, err = self.calc_NMF(comp_names, fusion_data, beta_candidates, term, init_H=None)
        endmembers, err = self.train_nn(comp_names, self.data_collection, save_path + "checkpoints.ckpt")


        #comp_map, comp_dict, errs_indv, err = self.reconstruct_spectra(maps, endmembers, comp_names,
        #                                                               fusion_data, save_path,
        #                                                               splits_i, kron_vec, plot=True)

        plt.close("all")
        #df.plot(x="omega_2", y="NMF_error", logx=True)
        #plt.savefig(self.unmixing_save_name)
        #plt.savefig(save_name)
        return err

    def calc_NMF(self, comp_names, fusion_data, beta_candidates, term, init_H=None):
        num_components = len(comp_names)
        # Erstellen Sie eine Instanz der NMF-Klasse mit der gewünschten Anzahl der Komponenten

        if False:
            if init_H is not None:
                model = NMF(n_components=num_components, init="custom")
                W = model.fit_transform(fusion_data, H=init_H)
            else:
                model = NMF(n_components=num_components)
                W = model.fit_transform(fusion_data)

            H = model.components_
        else:
            beta_best, A, S, results_save = NMF_QMV(fusion_data, num_components, beta_candidates, term,
                                                                  'DRAWFIGS', 'no')
            H = A.T
            W = S.T

        # Führen Sie die NMF-Zerlegung auf den EELS-Daten durch

        #W : array-like, shape (n_samples, n_components)
        #If init=’custom’, it is used as initial guess for the solution.

        #H : array-like, shape (n_components, n_features)
        #If init=’custom’, it is used as initial guess for the solution.



        err = model.reconstruction_err_#





        return W, H, err


    def reconstruct_spectra(self, maps, endmembers, comp_names, fusion_data, save_path, overview_image_path, splits_i, kron_vec, plot=False):
        prediction = np.matmul(maps, endmembers)
        num_components = len(comp_names)
        H = endmembers
        W = maps

        if self.KRON:
            dekron_pred = unkron(kron_vec, prediction)
            prediction = np.mean(dekron_pred, axis=0)
            #ss = np.std(dekron_pred, axis=0)
            #print(np.mean(ss))
            dekron_train = unkron(kron_vec, fusion_data)
            fusion_data = np.mean(dekron_train, axis=0)
            ss_ = np.std(dekron_train, axis=0)
            #print(np.mean(ss_))

        # get single spectra back
        indv_preds_datas = []
        indv_fusion_datas = []
        last_split = 0
        for ind in splits_i:
            indv_pred = prediction[:,last_split:ind]
            indv_fusion = fusion_data[:,last_split:ind]
            last_split = ind
            indv_preds_datas.append(indv_pred)
            indv_fusion_datas.append(indv_fusion)

        errs_indv = []
        for ii in range(len(splits_i)):
            exp_var_score = explained_variance_score(indv_fusion_datas[ii], indv_preds_datas[ii])
            errs_indv.append(exp_var_score)

        err_total = explained_variance_score(fusion_data, prediction)
        err = err_total

        comp_dict = {}
        comp_map = {}
        ref_dict = {}

        make_dir(save_path)

        if self.KRON:
            H = np.mean(unkron(kron_vec, H), axis=0)
        for i, comp_name in enumerate(comp_names):
            E = H[i, :]
            indv_spectra = get_spektra_back(E, splits_i)

            indv_spectra[0] = remove_eels_background_with_fitting_data(self.raw_cl_eels_cutted, indv_spectra[0], self.elements)

            comp_dict[comp_name] = (self.real_E_axis, indv_spectra)

            W_img = W[:, i].reshape(self.eels_data.shape[:-1])
            comp_map[comp_name] = W_img
            np.save(save_path + comp_name, W_img)

        self.overview_plot(num_components, comp_dict, comp_map, overview_image_path)

        if True:
            for i, comp_n in enumerate(comp_names):
                if self.use_refs:
                    ref_data = (self.ref_axis, self.ref_data, self.ref_names, self.ref_colors)
                else:
                    ref_data = None
                comp_name = "comp" + str(i)
                self.single_comp_plot(comp_name, *comp_dict[comp_n], comp_map[comp_n],
                                      save_path + comp_name + ".png", ref_data)

        return comp_map, comp_dict, errs_indv, err

    def single_comp_plot(self, name, energy_axs, I_comps, W_map, save_name, ref_data):
        SPECTRA_NAMES = ["CL_EELS", "EDX"]


        fig, axs = plt.subplots(len(I_comps) + 1,1)
        fig.set_size_inches(18.5, 10.5)

        if ref_data is not None:
            ref_axis, ref_I, ref_names, ref_colors = ref_data
            for kk in range(0, len(self.ref_data)):

                if ref_I[kk] is not None:
                    factor = np.max(I_comps[0]) / np.max(ref_I[kk])
                    axs[0].plot(ref_axis[kk], ref_I[kk] * factor, c=ref_colors[kk]
                                   , label=ref_names[kk])

        for ss, spec in enumerate(I_comps):
            axs[ss].plot(energy_axs[ss], spec, c="b", label=name)
            axs[ss].legend()
            axs[ss].set_ylabel(SPECTRA_NAMES[ss])
            if ss < 1:
                axs[ss].set_xlabel("E [eV]")
            else:
                axs[ss].set_xlabel("E [keV]")
        axs[len(I_comps)].imshow(W_map)

        plt.legend()
        plt.savefig(save_name, dpi=600)


    def overview_plot(self, num_components, comp_dict, comp_map, save_name):


        # SPECTRA_NAMES = ["LL_EELS", "CL_EELS", "EDX"]
        SPECTRA_NAMES = ["CL_EELS", "EDX"]

        plt.close("all")
        _, indv_spectra_one = comp_dict[list(comp_dict.keys())[0]]
        fig, axs = plt.subplots(num_components, len(indv_spectra_one) + 1)
        fig.set_size_inches(18.5, 10.5)

        for i, comp_name in enumerate(list(comp_dict.keys())):

            real_E_axis, indv_spectra = comp_dict[comp_name]

            if self.ref_data is not None:
                for kk in range(0, len(self.ref_data)):

                    if self.ref_data[kk] is not None:
                        factor = np.max(indv_spectra[0]) / np.max(self.ref_data[kk])
                        #factor_ll = np.max(indv_spectra[0]) / np.max(self.ref_data_ll[i])
                        axs[i, 0].plot(self.ref_axis[kk], self.ref_data[kk] * factor, c=self.ref_colors[kk]
                                       , label=self.ref_names[kk])
                        #axs[i, 0].plot(self.ref_axis_ll[i], self.ref_data_ll[i] * factor_ll, c='g',
                        #               label=self.ref_names[i])


            for ss, spec in enumerate(indv_spectra):
                axs[i, ss].plot(real_E_axis[ss], spec, c="b", label="comp" + str(i))
                axs[i, ss].legend()
                axs[i, ss].set_ylabel(SPECTRA_NAMES[ss])
                if ss < 1:
                    axs[i, ss].set_xlabel("E [eV]")
                else:
                    axs[i, ss].set_xlabel("E [keV]")
            W_img = comp_map[comp_name]
            axs[i, len(indv_spectra)].imshow(W_img)
        plt.legend()
        plt.savefig(save_name, dpi=600)
        plt.savefig(save_name, dpi=600)


    def train_nn(self, comp_names, fusion_data_binned, save_name):
        unmixer = ConvNetUnmixer(fusion_data_binned, fusion_data_binned, save_name)
        unmixer.stepwise_training(len(comp_names), [(100, 10, 0.001, 0.1), (1000,300, 0.005, 0.2)])
        #unmixer.build_model(len(comp_names))
        #unmixer.load_last_training()
        endmembers = unmixer.get_endmembers()
        #maps = unmixer.get_maps()
        err = unmixer.get_error()
        del unmixer
        return endmembers, err

    def train_tuning(self, comp_names, fusion_data_binned, save_name, hyperparameter):
        R, batch_size, epochs, learning_rate, dropout, layer_activation, end_activation, layer_units = hyperparameter
        unmixer = ConvNetUnmixer(fusion_data_binned, fusion_data_binned, save_name)
        unmixer.build_model(R=R, g=layer_activation, end_act=end_activation, learning_rate=learning_rate,
                            dropout_rate=dropout, layer_units=layer_units)
        print("current loss:" + str(unmixer.loss()))
        unmixer.training(epochs=epochs, batch_size=batch_size)

    def predict(self, PATH, save_path, overview_image, remove_eels_background=False, KRON=False, use_refs=True):
        self.use_refs = use_refs
        self.path = PATH
        self.KRON = KRON

        if use_refs:
            self.si3n4_cl = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                               spectrum_type="coreloss")[1]
            self.si3n4_ll = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                               spectrum_type="lowloss")[0]

            self.sio2_cl = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                              spectrum_type="coreloss")[0]  # TODO ?
            self.sio2_ll = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                              spectrum_type="lowloss")[0]

            self.si_cl = hs.datasets.eelsdb(title="Silicon",
                                            spectrum_type="coreloss")[2]
            #self.si_ll = hs.datasets.eelsdb(title="Silicon",
            #                                spectrum_type="lowloss")

        if os.path.isfile(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4'):
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss) (aligned).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4')
        else:
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss).dm4')
            self.raw_ll_eels.align_zero_loss_peak(also_align=[self.raw_cl_eels])
        self.raw_cl_eels.add_elements(self.elements)
        if remove_eels_background:
            self.raw_cl_eels = remove_eels_background_with_fitting(self.raw_cl_eels, self.raw_ll_eels)
        self.raw_cl_eels.fourier_ratio_deconvolution(ll=self.raw_ll_eels, extrapolate_lowloss=False,
                                                     extrapolate_coreloss=False)
        self.raw_eds = hs.load(PATH + "EDS Spectrum Image.dm4")

        ####

        _os = self.parameters["omegas"]
        end_eels_cl, start_cl_eels_energy, end_eels_cl_energy = self.parameters["end_eels_cl"]
        self.define_axis(_os, end_eels_cl, start_cl_eels_energy, end_eels_cl_energy, use_refs=self.use_refs)

        fusion_data, splits_i, kron_vec = make_datafusion([self.eels_ll_data, self.eels_data,
                                                           self.edx_data], _os, self.KRON)



        comp_names = self.parameters["component_names"]

        #np.random.shuffle(self.data_collection)

        # endmembers, maps, err = self.calc_NMF(comp_names, fusion_data, beta_candidates, term, init_H=None)


        self.unmixer.val_data = fusion_data#[:,self.E_start:self.E_cut]
        endmembers = self.unmixer.get_endmembers()
        maps = self.unmixer.get_maps()
        loss, err = self.unmixer.get_error()
        #del self.unmixer

        comp_map, comp_dict, errs_indv, _ = self.reconstruct_spectra(maps, endmembers, comp_names,
                                                                       fusion_data, save_path, overview_image,
                                                                       splits_i, kron_vec, plot=True)
        
        cross_abundances = calc_cross_abundances(maps)

        plt.close("all")
        return comp_dict, comp_map, loss, err, cross_abundances

    def load_model(self, path, R, file_name=None):
        self.E_cut = 720
        self.E_start = 0
        if file_name is None:
            file_name = "checkpoints.ckpt"
        self.unmixer = ConvNetUnmixer(np.zeros((1,self.E_cut-self.E_start)), np.zeros((1,self.E_cut-self.E_start)), path + file_name)
        self.unmixer.build_model(R)
        self.unmixer.load_last_training()


    def save_df(self):
        self.df.to_pickle("err_TEM_withBack.pkl")

    def get_img_size(self, data):
        m = data.create_model()
        ext = m.axes_manager.signal_extent

        #mag = int(img.original_metadata.Instrument.Mag)
        #img_size = 200000000 / mag  # determined manually

        return (ext[1], ext[3])

    def make_predicter(self):
        return PrePredicter(self)


class PrePredicter:
    def __init__(self, large_unmixer: SpectralUnmixer):
        self.KRON = large_unmixer.KRON
        self.ref_data = None
        self.real_E_axis = large_unmixer.real_E_axis
        self.raw_cl_eels_cutted = large_unmixer.raw_cl_eels_cutted
        self.edx_data = large_unmixer.edx_data
        self.eels_data = large_unmixer.eels_data
        self.eels_ll_data = large_unmixer.eels_ll_data
        self.unmixer = None
        self.parameters = large_unmixer.parameters
        self.elements = large_unmixer.elements
        self.use_refs = False

    def stuff(self, use_refs, PATH, remove_eels_background):
        if use_refs:
            self.si3n4_cl = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                               spectrum_type="coreloss")[1]
            self.si3n4_ll = hs.datasets.eelsdb(title="Silicon Nitride Alpha",
                                               spectrum_type="lowloss")[0]

            self.sio2_cl = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                              spectrum_type="coreloss")[0]  # TODO ?
            self.sio2_ll = hs.datasets.eelsdb(title="Silicon Dioxide Amorphous",
                                              spectrum_type="lowloss")[0]

            self.si_cl = hs.datasets.eelsdb(title="Silicon",
                                            spectrum_type="coreloss")[2]
            # self.si_ll = hs.datasets.eelsdb(title="Silicon",
            #                                spectrum_type="lowloss")

        if os.path.isfile(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4'):
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss) (aligned).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss) (aligned).dm4')
        else:
            self.raw_cl_eels = hs.load(PATH + 'EELS Spectrum Image (high-loss).dm4')
            self.raw_ll_eels = hs.load(PATH + 'EELS Spectrum Image (low-loss).dm4')
            self.raw_ll_eels.align_zero_loss_peak(also_align=[self.raw_cl_eels])
        self.raw_cl_eels.add_elements(self.elements)
        if remove_eels_background:
            self.raw_cl_eels = remove_eels_background_with_fitting(self.raw_cl_eels, self.raw_ll_eels)
        self.raw_cl_eels.fourier_ratio_deconvolution(ll=self.raw_ll_eels, extrapolate_lowloss=False,
                                                     extrapolate_coreloss=False)
        self.raw_eds = hs.load(PATH + "EDS Spectrum Image.dm4")

        _os = self.parameters["omegas"]
        end_eels_cl, start_cl_eels_energy, end_eels_cl_energy = self.parameters["end_eels_cl"]
        self.define_axis(_os, end_eels_cl, start_cl_eels_energy, end_eels_cl_energy, use_refs=self.use_refs)

        self.fusion_data, self.splits_i, self.kron_vec = make_datafusion([self.eels_ll_data, self.eels_data,
                                                           self.edx_data], _os, self.KRON)

    def define_axis(self, os, end_cl_eels="as_ref", start_cl_eels_energy=None, end_cl_eels_energy=None, use_refs=True):

        real_energies_eels = get_energy_axis(self.raw_cl_eels)
        real_energies_ll_eels = get_energy_axis(self.raw_ll_eels)

        ll_start = 5  # eV

        if use_refs:
            ref_energies_ll_eels_sio2 = get_energy_axis(self.sio2_ll, 0)
            ref_energies_ll_eels_si3n4 = get_energy_axis(self.si3n4_ll, 0)
            real_i_end_ll_eels = np.argmax(real_energies_ll_eels >= self.sio2_ll.axes_manager.signal_extent[1])
            real_i_start_eels = np.argmax(real_energies_eels >= self.si3n4_cl.axes_manager.signal_extent[0])

            real_i_start_eels = np.argmax(real_energies_eels >= 70)
        else:

            real_i_end_ll_eels = len(real_energies_ll_eels)
            real_i_start_eels = 0
            real_i_start_eels = np.argmax(real_energies_eels >= 70)

        real_i_start_ll_eels = np.argmax(real_energies_ll_eels >= ll_start)
        real_energies_ll_eels = real_energies_ll_eels[real_i_start_ll_eels:real_i_end_ll_eels]
        if use_refs:
            ref_i_start_ll_eels_sio2 = np.argmax(ref_energies_ll_eels_sio2 >= ll_start)
            ref_energies_ll_eels_sio2 = ref_energies_ll_eels_sio2[ref_i_start_ll_eels_sio2:]

            ref_i_start_ll_eels_si3n4 = np.argmax(ref_energies_ll_eels_si3n4 >= ll_start)
            ref_energies_ll_eels_si3n4 = ref_energies_ll_eels_si3n4[ref_i_start_ll_eels_si3n4:]

            ref_energies_eels_sio2 = get_energy_axis(self.sio2_cl, 0)
            ref_energies_eels_si3n4 = get_energy_axis(self.si3n4_cl, 0)
            ref_energies_eels_si = get_energy_axis(self.si_cl, 0)

        if end_cl_eels == "all":
            real_i_end_eels = 2000  # -1
        elif end_cl_eels == "as_ref":
            real_i_end_eels = np.argmax(real_energies_eels >= self.si3n4_cl.axes_manager.signal_extent[1])
        elif end_cl_eels == "custom":
            real_i_end_eels = np.argmax(real_energies_eels >= end_cl_eels_energy)
            real_i_start_eels = np.argmax(real_energies_eels >= start_cl_eels_energy)

        self.E_cut = real_i_end_eels
        self.E_start = real_i_start_eels

        real_energies_eels = real_energies_eels[real_i_start_eels:real_i_end_eels]
        real_energies_eds = get_energy_axis(self.raw_eds)
        real_i_end_eds = 300
        real_energies_eds = real_energies_eds[:real_i_end_eds]

        self.edx_data = self.raw_eds.data[:, :, :real_i_end_eds]

        self.eels_data = self.raw_cl_eels.data[:, :, real_i_start_eels:real_i_end_eels]
        self.raw_cl_eels_cutted = self.raw_cl_eels.isig[real_i_start_eels:real_i_end_eels]
        self.eels_ll_data = self.raw_ll_eels.data[:, :, real_i_start_ll_eels:real_i_end_ll_eels]

        self.real_E_axis = []
        if os[0] is not None:
            self.real_E_axis.append(real_energies_ll_eels)
        if os[1] is not None:
            self.real_E_axis.append(real_energies_eels)
        if os[2] is not None:
            self.real_E_axis.append(real_energies_eds)

        # s.set_microscope_parameters(beam_energy=80,
        #                            convergence_angle=0.2,
        #                            collection_angle=2.55)

        if use_refs:
            sio2_ll_data = self.sio2_ll.data[ref_i_start_ll_eels_sio2:]
            si3n4_ll_data = self.si3n4_ll.data[ref_i_start_ll_eels_si3n4:]
            self.ref_data = [self.si_cl.data, self.si3n4_cl.data, self.sio2_cl.data, None]
            self.ref_axis = [ref_energies_eels_si, ref_energies_eels_si3n4, ref_energies_eels_sio2, None]
            self.ref_data_ll = [si3n4_ll_data, sio2_ll_data, None]
            self.ref_axis_ll = [ref_energies_ll_eels_si3n4, ref_energies_ll_eels_sio2, None]
            self.ref_names = ["Silicon ref", "Si3N4 ref", "SiO2 ref", None]
            self.ref_colors = ["g", "k", "r", None]
        else:
            self.ref_data = None

    def make_predicter(self):
        return Predicter(self)

class Predicter:
    def __init__(self, pre_predicter: PrePredicter):
        self.eels_data_shape = pre_predicter.eels_data.shape[:-1]
        self.real_E_axis = pre_predicter.real_E_axis
        self.kron_vec = pre_predicter.kron_vec
        self.splits_i = pre_predicter.splits_i
        self.fusion_data = pre_predicter.fusion_data
        self.unmixer = None
        self.parameters = pre_predicter.parameters
        self.ref_data = None
        self.elements = pre_predicter.elements

        energy_axis = pre_predicter.raw_cl_eels.axes_manager.signal_axes[0]
        self.energy_offset = energy_axis.offset
        self.energy_scale = energy_axis.scale
        self.energy_units = energy_axis.units
        self.beam_energy = pre_predicter.raw_cl_eels.metadata.Acquisition_instrument.TEM.beam_energy
        self.convergence_angle = pre_predicter.raw_cl_eels.metadata.Acquisition_instrument.TEM.convergence_angle
        self.collection_angle = pre_predicter.raw_cl_eels.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle

    def remove_eels_background_with_fitting(self, data):
        empty_spec = hs.signals.Signal1D(data)
        empty_spec.set_signal_type("EELS")
        empty_spec.set_microscope_parameters(beam_energy=self.beam_energy,
                                    convergence_angle=self.convergence_angle,
                                    collection_angle=self.collection_angle)

        #empty_spec.data = data
        empty_spec.axes_manager.signal_axes[0].scale = self.energy_scale
        empty_spec.axes_manager.signal_axes[0].units = self.energy_units
        empty_spec.axes_manager.signal_axes[0].offset = self.energy_offset
        sample = empty_spec

        sample.add_elements(self.elements)

        print("remove background via EELS fitting")
        m = sample.create_model()
        m.enable_fine_structure()  # wichtig!
        m.enable_background()
        m.enable_edges()
        m.smart_fit()
        comps = m.components
        s_back = m.as_signal(component_list=[comps.PowerLaw])
        return sample.data - s_back.data
        # s0 = m.as_signal()

    def predict(self, PATH, save_path, overview_image, remove_eels_background=False, KRON=False, use_refs=True):
        self.use_refs = use_refs
        self.path = PATH
        self.KRON = KRON



        ####

        _os = self.parameters["omegas"]


        comp_names = self.parameters["component_names"]

        # np.random.shuffle(self.data_collection)

        # endmembers, maps, err = self.calc_NMF(comp_names, fusion_data, beta_candidates, term, init_H=None)

        self.unmixer.val_data = self.fusion_data  # [:,self.E_start:self.E_cut]
        endmembers = self.unmixer.get_endmembers()
        maps = self.unmixer.get_maps()
        loss, err = self.unmixer.get_error()
        # del self.unmixer

        comp_map, comp_dict, errs_indv, _ = self.reconstruct_spectra(maps, endmembers, comp_names,
                                                                     self.fusion_data, save_path, overview_image,
                                                                     self.splits_i, self.kron_vec, plot=True)

        cross_abundances = calc_cross_abundances(maps)

        plt.close("all")
        return comp_dict, comp_map, loss, err, cross_abundances



    def reconstruct_spectra(self, maps, endmembers, comp_names, fusion_data, save_path, overview_image_path, splits_i,
                            kron_vec, remove_eels_background=False, plot=False):
        prediction = np.matmul(maps, endmembers)
        num_components = len(comp_names)
        H = endmembers
        W = maps

        if self.KRON:
            dekron_pred = unkron(kron_vec, prediction)
            prediction = np.mean(dekron_pred, axis=0)
            # ss = np.std(dekron_pred, axis=0)
            # print(np.mean(ss))
            dekron_train = unkron(kron_vec, fusion_data)
            fusion_data = np.mean(dekron_train, axis=0)
            ss_ = np.std(dekron_train, axis=0)
            # print(np.mean(ss_))

        # get single spectra back
        indv_preds_datas = []
        indv_fusion_datas = []
        last_split = 0
        for ind in splits_i:
            indv_pred = prediction[:, last_split:ind]
            indv_fusion = fusion_data[:, last_split:ind]
            last_split = ind
            indv_preds_datas.append(indv_pred)
            indv_fusion_datas.append(indv_fusion)

        errs_indv = []
        for ii in range(len(splits_i)):
            exp_var_score = explained_variance_score(indv_fusion_datas[ii], indv_preds_datas[ii])
            errs_indv.append(exp_var_score)

        err_total = explained_variance_score(fusion_data, prediction)
        err = err_total

        comp_dict = {}
        comp_map = {}
        ref_dict = {}

        make_dir(save_path)

        if self.KRON:
            H = np.mean(unkron(kron_vec, H), axis=0)
        for i, comp_name in enumerate(comp_names):
            E = H[i, :]
            indv_spectra = get_spektra_back(E, splits_i)
            if remove_eels_background:
                indv_spectra[0] = self.remove_eels_background_with_fitting(indv_spectra[0])

            comp_dict[comp_name] = (self.real_E_axis, indv_spectra)

            W_img = W[:, i].reshape(self.eels_data_shape)
            comp_map[comp_name] = W_img
            np.save(save_path + comp_name, W_img)

        self.overview_plot(num_components, comp_dict, comp_map, overview_image_path)

        if True:
            for i, comp_n in enumerate(comp_names):
                if self.use_refs:
                    ref_data = (self.ref_axis, self.ref_data, self.ref_names, self.ref_colors)
                else:
                    ref_data = None
                comp_name = "comp" + str(i)
                self.single_comp_plot(comp_name, *comp_dict[comp_n], comp_map[comp_n],
                                      save_path + comp_name + ".png", ref_data)

        return comp_map, comp_dict, errs_indv, err

    def single_comp_plot(self, name, energy_axs, I_comps, W_map, save_name, ref_data):
        SPECTRA_NAMES = ["CL_EELS", "EDX"]


        fig, axs = plt.subplots(len(I_comps) + 1,1)
        fig.set_size_inches(18.5, 10.5)

        if ref_data is not None:
            ref_axis, ref_I, ref_names, ref_colors = ref_data
            for kk in range(0, len(self.ref_data)):

                if ref_I[kk] is not None:
                    factor = np.max(I_comps[0]) / np.max(ref_I[kk])
                    axs[0].plot(ref_axis[kk], ref_I[kk] * factor, c=ref_colors[kk]
                                   , label=ref_names[kk])

        for ss, spec in enumerate(I_comps):
            axs[ss].plot(energy_axs[ss], spec, c="b", label=name)
            axs[ss].legend()
            axs[ss].set_ylabel(SPECTRA_NAMES[ss])
            if ss < 1:
                axs[ss].set_xlabel("E [eV]")
            else:
                axs[ss].set_xlabel("E [keV]")
        axs[len(I_comps)].imshow(W_map)

        plt.legend()
        plt.savefig(save_name, dpi=600)


    def overview_plot(self, num_components, comp_dict, comp_map, save_name):


        # SPECTRA_NAMES = ["LL_EELS", "CL_EELS", "EDX"]
        SPECTRA_NAMES = ["CL_EELS", "EDX"]

        plt.close("all")
        _, indv_spectra_one = comp_dict[list(comp_dict.keys())[0]]
        fig, axs = plt.subplots(num_components, len(indv_spectra_one) + 1)
        fig.set_size_inches(18.5, 10.5)

        for i, comp_name in enumerate(list(comp_dict.keys())):

            real_E_axis, indv_spectra = comp_dict[comp_name]

            if self.ref_data is not None:
                for kk in range(0, len(self.ref_data)):

                    if self.ref_data[kk] is not None:
                        factor = np.max(indv_spectra[0]) / np.max(self.ref_data[kk])
                        #factor_ll = np.max(indv_spectra[0]) / np.max(self.ref_data_ll[i])
                        axs[i, 0].plot(self.ref_axis[kk], self.ref_data[kk] * factor, c=self.ref_colors[kk]
                                       , label=self.ref_names[kk])
                        #axs[i, 0].plot(self.ref_axis_ll[i], self.ref_data_ll[i] * factor_ll, c='g',
                        #               label=self.ref_names[i])


            for ss, spec in enumerate(indv_spectra):
                axs[i, ss].plot(real_E_axis[ss], spec, c="b", label="comp" + str(i))
                axs[i, ss].legend()
                axs[i, ss].set_ylabel(SPECTRA_NAMES[ss])
                if ss < 1:
                    axs[i, ss].set_xlabel("E [eV]")
                else:
                    axs[i, ss].set_xlabel("E [keV]")
            W_img = comp_map[comp_name]
            axs[i, len(indv_spectra)].imshow(W_img)
        plt.legend()
        plt.savefig(save_name, dpi=600)
        plt.savefig(save_name, dpi=600)

