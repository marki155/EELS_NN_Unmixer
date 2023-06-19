import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf




def parameters_to_model(conc_arr, splits, empty_model):


    the_dict = empty_model.as_dictionary()
    comp_names = [x["name"] for x in the_dict["components"]]
    for name, start_p, end_p in splits:
        values = conc_arr[:, start_p:end_p]
        if values.shape[1] == 1:
            values = values.reshape(values.shape[0])

        comp_name, parameter_name = tuple(name.split(";"))
        comp_id = comp_names.index(comp_name)
        parameter_names = [x["name"] for x in the_dict["components"][comp_id]["parameters"]]
        param_id = parameter_names.index(parameter_name)
        the_dict["components"][comp_id]["parameters"][param_id]["map"]["values"] = values

    the_dict["components"] = the_dict["components"][1:]
    empty_model._load_dictionary(the_dict)


    return empty_model


def parameters_to_I(conc_arr, splits, empty_model):
    dd = {}
    ext = empty_model.axes_manager.signal_extent
    E = np.linspace(ext[0], ext[1], empty_model.signal.data.shape[2]).astype('float32')
    I = np.zeros_like(E, dtype='float32')

    comp_names = [x.name for x in empty_model.active_components]
    for gg in range(len(comp_names)):
        dd[gg] = {}
    for name, start_p, end_p in splits:
        values = conc_arr[:, start_p:end_p]

        comp_name, parameter_name = tuple(name.split(";"))
        comp_id = comp_names.index(comp_name)
        #parameter_names = [x.name for x in empty_model.active_components[comp_id].free_parameters]
        #param_id = parameter_names.index(parameter_name)
        #empty_model.active_components[comp_id].free_parameters[param_id]

        dd[comp_id][parameter_name] = values

    #dd[0]["E0"] = tf.zeros((conc_arr.shape[0], 1))



    main_params = []

    for i in dd.keys():
        if False:#i == 0:
            A = dd[i]["A"]
            r = dd[i]["r"]
            E0 = dd[i]["E0"]
            I_part = A * (E - E0) ** -r

        else:
            param_main = dd[i]["intensity"]
            parameters_fine = dd[i]["fine_structure_coeff"]
            main_params.append(param_main)
            I_part = empty_model.active_components[i].function(E, param_main, parameters_fine)
        I += I_part

    #empty_model.active_components[comp_id].function(param_main, parameters_fine)

    #the_dict["components"] = the_dict["components"][1:]
    #empty_model._load_dictionary(the_dict)


    return (I, tf.reduce_sum(tf.stack(main_params)))

def EELS_Edge_Model(Es, par):
    E = tf.convert_to_tensor(Es)

    E2 = E ** 2
    E3 = tf.exp(-E)
    E4 = tf.exp(E)
    E5 = tf.ones_like(E)
    E_axis = tf.stack([E, E2, E3, E4, E5])


    A = par[:,0], par[:,1]*E, par[:,2]*E**2
    return E_axis

def parameters_to_model_tf(conc_arr, splits, Es):
    par = conc_arr[:,20:]
    Si = EELS_Edge_Model(Es)
    N = EELS_Edge_Model(Es)
    O = EELS_Edge_Model(Es)
    C = EELS_Edge_Model(Es)
    E_axis = tf.concat([Si, N, O, C], 0)

    t = tf.matmul(conc_arr, E_axis)


    return t




def model_to_parameters(model):
    conc_arr = []
    splits = []
    position = 0

    for comp in list(model.active_components):
        for parameter in comp.free_parameters:
            arr = parameter.map["values"]

            split_name = comp.name + ";" + parameter.name
            if len(arr.shape) == 2:
                arr = arr.reshape(arr.shape[1] ,1)
                end_position = position + 1
            else:
                end_position = position + arr.shape[2]
                arr = arr.reshape(arr.shape[1], arr.shape[2])


            conc_arr.append(arr)
            splits.append((split_name, position, end_position))
            position = end_position
        if comp.name == "PowerLaw":
            split_name = comp.name + ";E0"
            arr = np.zeros((model.signal.data.shape[1],1))
            end_position = position + 1
            conc_arr.append(arr)
            splits.append((split_name, position, end_position))
            position = end_position

    return np.concatenate(conc_arr, axis=1), splits


if __name__ == "__main__":


    ax = {'size': 4090, 'name': 'EELS', 'units': 'eV', 'scale': 0.25, 'offset': 70.78768157958984}
    sample_tf = hs.signals.EELSSpectrum_TF(np.zeros((1,10,1000)))



    sample_tf.add_elements(["Si", "C", "O", "N"])

    sample_tf.set_microscope_parameters(beam_energy=200,
                                  convergence_angle=0.0,
                                  collection_angle=111.1)

    m_tf = sample_tf.create_model()
    m_tf.enable_fine_structure()  # wichtig!
    m_tf.enable_edges()
    m_tf.disable_background()
    #m.fit()

    if False:
        sample = hs.signals.EELSSpectrum(np.zeros((10, 1000)))

        sample.add_elements(["Si", "C", "O", "N"])

        sample.set_microscope_parameters(beam_energy=200,
                                            convergence_angle=0.0,
                                            collection_angle=111.1)

        m = sample.create_model()
        m.enable_fine_structure()  # wichtig!
        m.enable_edges()
        m.disable_background()

        m.active_components[0].function(np.linspace(0,1000,1000))

    parameter_ = m_tf.active_components[0].free_parameters[1].map["values"]
    parameter_fine = m_tf.active_components[0].free_parameters[0].map["values"]

    parameter_ = tf.convert_to_tensor(parameter_[0,0])
    parameter_fine = tf.convert_to_tensor(parameter_fine[0,0,:])
    conc_array, splits = model_to_parameters(m_tf)
    conc_array = tf.convert_to_tensor(conc_array, dtype='float32')
    part_I = parameters_to_I(conc_array, splits, empty_model=m_tf)
    I = m_tf.active_components[0].function(np.linspace(0,1000,1000), parameter_, parameter_fine)
    plt.plot(list(range(1000)), np.mean(m.as_signal().data, axis=0))
    plt.show()
    conc_arr, splits = model_to_parameters(m)
    conc_arr[:,:] = 0.1
    new_model = parameters_to_model(conc_arr, splits, empty_model=m)
    plt.plot(list(range(1000)), np.mean(new_model.as_signal().data, axis=0))
    plt.show()