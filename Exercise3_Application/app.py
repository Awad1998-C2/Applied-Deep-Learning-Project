import streamlit as st
import numpy as np
import re
import tensorflow as tf
import os

############################
# 0) Derive the absolute path for model.weights.h5
############################
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder containing app.py
weights_path = os.path.join(APP_DIR, "model.weights.h5")

############################
# 1) Element list & parser
############################
element_symbols = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]
NUM_ELEMENTS = len(element_symbols)

def parse_chemical_formula(formula: str) -> np.ndarray:
    element_counts = np.zeros(NUM_ELEMENTS, dtype=float)
    pattern = r'([A-Z][a-z]?)(\d*)'
    tokens = re.findall(pattern, formula)

    for (elem_symbol, count_str) in tokens:
        count = float(count_str) if count_str else 1.0
        if elem_symbol in element_symbols:
            idx = element_symbols.index(elem_symbol)
            element_counts[idx] += count
        else:
            st.warning(f"Unknown element '{elem_symbol}' in formula.")
    return element_counts

############################
# 2) Build & load model ONCE (cached)
############################
@st.cache_resource
def load_model(weights_path_absolute: str):
    """
    Build the model architecture once, load weights from the absolute path,
    and return the model. This is cached by Streamlit so it's only run once.
    """
    # # (Optional) Debug prints to confirm the path & local files
    # st.write("Using absolute path:", weights_path_absolute)
    # st.write("Files in this folder:", os.listdir(APP_DIR))

    INPUT_DIM = 119
    NUM_HIDDEN_LAYERS = 40

    model_layers = [tf.keras.layers.Input(shape=(INPUT_DIM,))]

    for _ in range(NUM_HIDDEN_LAYERS):
        model_layers.append(tf.keras.layers.Dense(256, use_bias=False, kernel_initializer="glorot_uniform"))
        model_layers.append(tf.keras.layers.BatchNormalization())
        model_layers.append(tf.keras.layers.Activation('silu'))

    model_layers.append(tf.keras.layers.Dense(4, activation='softmax', kernel_initializer='glorot_uniform'))

    model = tf.keras.Sequential(model_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    # Load weights from the absolute path
    model.load_weights(weights_path_absolute)
    return model

############################
# 3) Streamlit app logic
############################
st.title("Magnetic Ordering Prediction")
st.write("Enter a chemical formula and density to predict whether it's Ferromagnetic, Non-magnetic, Ferrimagnetic, or Antiferromagnetic.")

formula = st.text_input("Chemical Formula (e.g. Fe2O3)", "Fe2O3")
density = st.number_input("Density (g/cm3)", min_value=0.0, max_value=1000.0, value=7.87, step=0.01)

if st.button("Predict"):
    # 1) Load model from cache, specifying the absolute path
    model = load_model(weights_path)

    # 2) Build the input vector
    formula_vec = parse_chemical_formula(formula)
    input_vector = np.concatenate([formula_vec, [density]], axis=0)  # shape (119,)
    input_vector = input_vector.reshape(1, -1)  # shape (1,119)

    # 3) Predict
    predictions = model.predict(input_vector)
    pred_probabilities = predictions[0]
    predicted_index = np.argmax(pred_probabilities)

    ordering_classes = ["FM", "NM", "FiM", "AFM"]
    class_name_map = {
        "FM": "Ferromagnetic",
        "NM": "Non-magnetic",
        "FiM": "Ferrimagnetic",
        "AFM": "Antiferromagnetic",
    }

    predicted_class_short = ordering_classes[predicted_index]
    predicted_class_long  = class_name_map[predicted_class_short]
    confidence_pct = pred_probabilities[predicted_index] * 100.0

    # Display result
    st.subheader("Prediction Result")
    st.write(f"**Class**: {predicted_class_long}")
    st.write(f"**Model Confidence**: {confidence_pct:.2f}%")
