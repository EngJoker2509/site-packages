# https://app.datacamp.com/workspace/w/421ca202-c0b3-41cb-91e8-03ff99cc981b

# https://www.datacamp.com/tutorial/random-forests-classifier-python

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import tkinter as tk
import os
import xgboost as xgb
import seaborn as sns
# import datetime

from numpy.compat import basestring
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datetime import datetime, timedelta
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tabulate import tabulate

days_in_dates_first = True
save_pickle_name = "df.pkl"
save_csv_name = "data_rf"
save_csv_single_correl = "single_correlations.csv"

# Default tree parameters
params_trees = {'number of trees': 100,
                'max depth of tree': 10,
                'min samples to split node': 10,
                'min samples to be in leaf': 5,
                'share training set': 0.7,
                'confusion matrix cutoff': 0.5,
                'show feature importance': True,
                'show ROC': True,
                'draw and save trees': False,
                'save the data': True,
                'save the model': True}

# Default lg parameters
params_lg = {'number of iterations': 100,
             'share training set': 0.7,
             'collinearity threshold': 0.65,
             'insignificance threshold': 0.025,
             'confusion matrix cutoff': 0.2,
             'show ROC': True,
             'save the data': True,
             'save the model': True}

# Default kMeans parameters
params_kMeans = {'min groups': 2,
                 'max groups': 10,
                 'number random initiations': 10,
                 'show elbow chart': True,
                 'save the data': True,
                 'save the model': True}

rnd_state = 1
number_trees_pics = 5
trees_pics_depth = 2

# https://stackoverflow.com/questions/17098654/how-to-reversibly-store-and-load-a-pandas-dataframe-to-from-disk
save_data_pickle = False
load_model = False
get_data_from = "csv"  # "csv" or "pickle"

root = None
df_list = []  # an empty list of df objects
df_names = []  # an empty list of df object names
df = None
label_target_var = None
df_in_use = -1
operator_array = ["+", "-", "*", "/", "&", "="]
connector_array = ["=", "AND", "OR", "Cancel"]
overwrite_nan = -999999
unique_cutoff = 0.1
# if there are less than [x] unique values in a column, all will be shown
top_unique_value_threshold = 20
param_dict = []


def main_menu():
    global root

    # Check if the root window already exists
    if root is not None:
        # Clear the existing buttons from the window
        for child in root.winfo_children():
            child.destroy()
    else:
        # Create a new root window if one doesn't exist
        root = tk.Tk()
        root.title("Main Menu")
        root.geometry("400x600")

    # Create a list of dictionaries with text and command for each button
    button_dicts = [{"text": "Load data", "command": load_from_file}]

    # Add data frame buttons
    button_dicts = add_data_frame_buttons(button_dicts)

    button_dicts.append({"text": "Quit", "command": root.destroy})
    button_dicts.append({"text": "Test", "command": test})

    # Determine the width of the widest label
    max_width = max(max([len(button["text"]) for button in button_dicts]), 20)

    # Create the buttons using a loop
    for button_dict in button_dicts:
        button = tk.Button(
            root, text=button_dict["text"], width=max_width, command=button_dict["command"])
        button.pack(side='top', anchor='w')

    root.mainloop()


def add_data_frame_buttons(buttons):
    """
    Adds buttons to the main menu for operations that require a dataframe to be loaded
    :param buttons: a list of dictionaries with text and command for each button
    :return: a list of dictionaries with text and command for each button
    """
    if df is not None:
        buttons.append({"text": "Save csv", "command": save_to_csv})
        buttons.append({"text": "Data summary", "command": info_stats}),
        buttons.append({"text": "Sort data", "command": data_sort})
        buttons.append({"text": "Set target variable",
                       "command": set_target_var})
        buttons.append({"text": "Time transformation",
                       "command": transform_times})
        buttons.append({"text": "Create dummies", "command": create_dummies})
        buttons.append({"text": "Rolling averages",
                       "command": previous_averages})
        buttons.append({"text": "Delta Calculation",
                       "command": calc_delta_abs})
        buttons.append({"text": "Column operations", "command": column_calcs})
        buttons.append({"text": "Multi column operations",
                       "command": multi_column_calcs})
        buttons.append({"text": "Simple math transformations",
                       "command": simple_math_transformation})
        buttons.append({"text": "If...", "command": if_comparison3})
        buttons.append({"text": "Rename columns", "command": rename_columns})
        buttons.append({"text": "Delete columns", "command": remove_columns})
        buttons.append({"text": "Replace NaN", "command": replace_nan})
        buttons.append({"text": "Single Correlations",
                       "command": single_correlations})
        buttons.append({"text": "Correlation Matrix",
                       "command": correlation_matrix_chart})
        buttons.append({"text": "Change dataset", "command": change_df})
        buttons.append({"text": "Merge with...", "command": initiate_df_merge})
        buttons.append({"text": "Train model", "command": choose_algorithm})
    return buttons


def checkbox_menu(title_text, options):
    # create the top-level window for the selection dialog
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("400x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    # create a dictionary to hold the IntVar for each checkbox
    vars_dict = {}
    for option in options:
        vars_dict[option] = tk.IntVar()

    # create the checkboxes
    for option in options:
        tk.Checkbutton(frame, text=option,
                       variable=vars_dict[option]).pack(anchor="w")

    # create the ok button to return the selected items
    def ok():
        selected_items = [option for option,
                          var in vars_dict.items() if var.get() == 1]
        top.destroy()
        return selected_items

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="left", pady=10, padx=20, anchor="sw")

    # for more than one button
    # for button in buttons:
    #     tk.Button(frame, text=button, command=lambda text=button: ok(text), width=10).pack(
    #         side="left", pady=10, padx=20, anchor="sw")

    # wait for the selection dialog window to close
    top.wait_window()

    # return the selected items
    return ok()


def radio_menu(title_text):
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("800x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    scrollbar.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    options = df.columns.tolist()
    vars_dict = {}
    max_width = max(max([len(option) for option in options]), 20)

    for option in options:
        data_type = df[option].dtype
        unique_ratio = df[option].nunique() / len(df[option])
        if option == label_target_var:
            vars_dict[option] = tk.StringVar(value="ignore")
        elif option[:2].lower() == "id" or option[-2:].lower() == "id":
            vars_dict[option] = tk.StringVar(value="ignore")
        elif data_type == "int64" or data_type == "float64":
            vars_dict[option] = tk.StringVar(value="numerical")
        elif unique_ratio < unique_cutoff:
            vars_dict[option] = tk.StringVar(value="categorical")
        else:
            vars_dict[option] = tk.StringVar(value="ignore")

    for option in options:
        frame2 = tk.Frame(frame)
        frame2.pack(side="top", fill="x", padx=5, pady=5)
        label = tk.Label(frame2, text=option, width=max_width)
        label.pack(side="left")

        radio_frame = tk.Frame(frame2)
        radio_frame.pack(side="left", padx=5)

        for value in ["categorical", "numerical", "ignore"]:
            rb = tk.Radiobutton(radio_frame, text=value,
                                variable=vars_dict[option], value=value)
            rb.pack(side="left")

    def ok():
        selected_items = {option: var.get()
                          for option, var in vars_dict.items()}
        top.destroy()
        return selected_items

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="bottom", pady=10, padx=20)

    top.wait_window()

    return ok()


def button_menu(title_text, button_list, button_min_width=10, window_size="400x600", show_datetime=True):
    global df

    # Define function to be called when a button is clicked
    def button_clicked(sel_button):
        # Set the value of the selected column to the label_target_var
        button_selected.set(sel_button)
        # Close the window
        window.destroy()

    # Create a tkinter window
    window = tk.Toplevel(root)
    window.title(title_text)
    window.geometry(window_size)

    # create a canvas with a scrollbar
    canvas = tk.Canvas(window, height=500)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))

    # Create a StringVar to hold the selected column name
    button_selected = tk.StringVar()

    # Add a label to display the selected column name
    label = tk.Label(frame, textvariable=button_selected)
    label.pack()

    # Find the maximum width of the button labels
    max_width = max(
        max([len(button)
            for button in button_list if show_datetime or df[button].dtype == "datetime64[ns]"]),
        button_min_width)

    for button in button_list:
        if show_datetime or df[button].dtype == "datetime64[ns]":
            new_button = tk.Button(
                frame, text=button, command=lambda i=button: button_clicked(i), width=max_width)
            new_button.pack(anchor="w")

    # Wait for the window to be closed
    window.wait_window()

    # Return the selected column name
    return button_selected.get()


def math_op_menu(title_text):
    global df

    # Define function to be called when a button is clicked
    def button_clicked(operator):
        # Set the value of the selected operator
        chosen_operator.set(operator)
        # Close the window
        window.destroy()

    # Create a tkinter window
    window = tk.Toplevel(root)
    window.title(title_text)

    # Create a StringVar to hold the selected column name
    chosen_operator = tk.StringVar()

    # Add a label to display the selected column name
    label = tk.Label(window, textvariable=chosen_operator)
    label.pack()

    # Add a button for each column in the dataframe
    for column in operator_array:
        button = tk.Button(window, text=column, height=5, width=10,
                           command=lambda col=column: button_clicked(col))
        button.pack(side="left")

    # Wait for the window to be closed
    window.wait_window()

    # Return the selected operator
    return chosen_operator.get()


def math_comp_menu(title_text, var_summary):
    # create the top-level window for the selection dialog
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("400x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    stat_headers = ["Min: ", "Max:", "Mean: ", "Median: ", "Std: "]
    options = ["between", "not between", "=", "!=", "<", "<=", ">", ">="]

    for i in range(5):
        var = tk.StringVar()
        label = tk.Label(frame, textvariable=var)

        var.set(f"{stat_headers[i]} {var_summary[i]}")
        label.pack()

    combo_box = ttk.Combobox(frame, values=options)
    combo_box.pack()

    entry1 = ttk.Entry(frame)
    entry2 = ttk.Entry(frame)

    def on_combobox_select(event):
        selected_option = combo_box.get()
        entry1.pack()
        if selected_option in ["between", "not between"]:
            entry2.pack()
        else:
            entry2.pack_forget()

    def ok():
        # Retrieve the selected option and values
        selected_option = combo_box.get()
        value1 = entry1.get()
        value2 = entry2.get() if selected_option in [
            "between", "not between"] else None

        # Store the values as attributes of the window object
        top.selected_option = selected_option
        top.value1 = value1
        top.value2 = value2

        # Close the window
        top.destroy()

        # Return the selected option and values
        return selected_option, value1, value2

    combo_box.bind("<<ComboboxSelected>>", on_combobox_select)

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="left", pady=10, padx=20, anchor="sw")

    # wait for the selection dialog window to close
    top.wait_window()

    # return the selected items
    return top.selected_option, top.value1, top.value2


def choose_df(title_text):
    global df, df_in_use, df_names, df_list

    df_list[df_in_use] = df

    # Define function to be called when a button is clicked
    def button_clicked(df_names):
        # Set the value of the selected df
        chosen_df.set(df_names)
        # Close the window
        window.destroy()

    # Create a tkinter window
    window = tk.Toplevel(root)
    window.title(title_text)
    window.geometry("200x300")

    # Create a StringVar to hold the selected df name
    chosen_df = tk.StringVar()

    # Add a label to display the selected df name
    label = tk.Label(window, textvariable=chosen_df)
    label.pack()

    # Find the maximum width of the button labels
    max_width = max(max([len(name) for name in df_names]), 10)

    # Add a button for each dataframe
    for name in df_names:
        button = tk.Button(window, text=name, width=max_width,
                           command=lambda i=name: button_clicked(i))
        button.pack(anchor="w")

    # Wait for the window to be closed
    window.wait_window()

    return chosen_df.get()


def get_input(info_text, number_input=True):
    root = tk.Tk()
    root.withdraw()
    if number_input:
        user_input = simpledialog.askinteger("Input", f"Enter {info_text}:")
    else:
        user_input = simpledialog.askstring("Input", f"Enter {info_text}:")
    root.destroy()
    return user_input


def data_sort():
    global df
    rank1 = button_menu(
        "Choose the first rank variable for sorting.", df.columns)
    rank2 = button_menu(
        "Choose the second rank variable for sorting.", df.columns)
    rank3 = button_menu(
        "Choose the third rank variable for sorting.", df.columns)
    if not rank1:
        rank1 = rank2
        rank2 = rank3
        rank3 = None
    if not rank2:
        rank2 = rank3
        rank3 = None
    if rank1:
        sort_data(df, rank1, rank2, rank3)
        messagebox.showinfo("Data Sort", "Data sorted as specified.")
    else:
        messagebox.showinfo(
            "Data Sort", "Data not sorted as no columns were specified.")


def get_tree_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("Tree Model Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_trees.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['show feature importance', 'show ROC', 'draw and save trees', 'save the data', 'save the model']:

            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_trees[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    return param_values


def get_lr_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("Logistic Regression Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_lg.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['show ROC', 'save the data', 'save the model']:

            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_trees[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    return param_values


def get_kMeans_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("k-means Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_kMeans.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['show elbow chart', 'save the data', 'save the model']:

            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_kMeans[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    return param_values


def submit_params(window, param_values, entries):
    # Update the parameter values with the values from the entry widgets
    for key, entry in entries.items():
        if isinstance(entry, tk.Entry):
            # Convert numeric values to floats
            param_values[key] = float(entry.get())
        else:
            # Convert Boolean values to True or False
            param_values[key] = bool(entry.get())

    # Close the window
    window.destroy()


def set_target_var():
    global df, label_target_var
    label_target_var = button_menu("Choose the target variable.", df.columns)
    if label_target_var:
        messagebox.showinfo(
            "Target Variable", f"The target variable was set to '{label_target_var}'.")
    else:
        messagebox.showinfo("Target Variable", "No target variable selected.")


def change_df():
    global df, df_in_use, df_names, df_list
    df_in_use = df_names.index(button_menu("Select a dataset.", df_names))
    df = df_list[df_in_use]


def initiate_df_merge():
    global df, df_names
    df_to_add = df_list[df_names.index(button_menu(
        "Choose which dataset shall be added.", df_names))]
    identifier = button_menu("Choose the identifier.", df.columns)
    merge_dataframes(df, df_to_add, identifier, "left")
    messagebox.showinfo(
        "Data Merge", "Both datasets have been joined into a new dataset.")


def create_dummies():
    global df
    cols_to_dummy = checkbox_menu(
        "Choose which variables shall be binarized.", df.columns.tolist())
    min_dummy_number = get_input(
        "the minimum count of observations to qualify for binarization")
    delete_initial_column = button_menu("What shall happen with the initial column?", [
                                        "Keep initial column", "Remove initial column"])
    if min_dummy_number:
        for col in cols_to_dummy:
            dummy_variables(df, col, min_dummy_number)
            if delete_initial_column == "Remove initial column":
                df = df.drop(col, axis=1)

    if cols_to_dummy and min_dummy_number:
        messagebox.showinfo("Dummy Variables",
                            "Selected columns have been binarized.")
    else:
        messagebox.showinfo("Dummy Variables",
                            "No columns selected or no number provided.")


def simple_math_transformation():
    global df
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    cols_to_math = checkbox_menu(
        "Choose which variables shall be mathematically transformed.", numeric_cols)

    sel_math_op = button_menu("Select one option to be performed.", [
                              "+, -, * , / constant", "SQRT(x), 1/x, LN(x), x^2", "Rounding"])

    if sel_math_op == "Rounding":
        digits = get_input("How many digits?")
        for col in cols_to_math:
            df[col] = df[col].apply(lambda x: round(x, digits))
    elif sel_math_op == "SQRT(x), 1/x, LN(x), x^2":
        for col in cols_to_math:
            simple_math_trans_variables(df, col)
    else:
        operator = button_menu("Choose an operation.", operator_array[:4])
        const = get_input("Enter the number.")
        for col in cols_to_math:
            if operator == "+":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] + const)
            elif operator == "-":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] - const)
            elif operator == "*":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] * const)
            elif operator == "/":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] / const)

    messagebox.showinfo("Math transformation",
                        "The math transformations have been made.")


def transform_times():
    global df
    cols_to_time_trans = checkbox_menu(
        "Choose which variables shall be made a time variables.", df.columns.tolist())
    time_trans_kinds = button_menu("test", ["All", "Only days since 1900"])
    for col in cols_to_time_trans:
        time_transformation(df, col, time_trans_kinds)

    if cols_to_time_trans:
        messagebox.showinfo("Time Transformation",
                            "Selected columns have been made time variables.")
    else:
        messagebox.showinfo("Time Transformation", "No columns selected.")


def column_calcs():
    operator = "x"
    first_column = None
    new_col_label = ""
    while not (operator == "="):
        if operator == "x":
            first_column_df = df[[button_menu(
                "Perform a standard maths operation on two columns. Choose the first column.",
                df.select_dtypes(include=['number']).columns)]]  # double [[]] to return df object incl. label name
            new_col_label = first_column_df.columns[0]
            first_column = first_column_df[first_column_df.columns[0]]
        operator = button_menu("Perform a standard maths operation on two columns. Choose the operator.",
                               operator_array)
        if not (operator == "="):
            second_column_df = df[
                [button_menu("Perform a standard maths operation on two columns. Choose the second column.",
                             df.select_dtypes(include=['number']).columns)]]
            new_col_label = "(" + new_col_label + "_" + \
                operator + "_" + second_column_df.columns[0] + ")"
            second_column = second_column_df[second_column_df.columns[0]]
            first_column = column_operation(
                first_column, second_column, operator)
    df.insert(len(df.columns), new_col_label, first_column)
    #              array.pop(new_column_name))
    # if first_column and operator and second_column and operator == "=":
    messagebox.showinfo("Column Operations",
                        f"A new column {new_col_label} was created.")
    # else:
    #     messagebox.showinfo("Column Operations", "Something went wrong.")


def multi_column_calcs():
    second_columns = None
    first_columns = checkbox_menu("Perform a standard maths operation on two columns. Choose the first columns.",
                                  df.columns.tolist())
    operator = button_menu(
        "Perform a standard maths operation on two columns. Choose the operator.", operator_array)
    if not (operator == "="):
        second_columns = checkbox_menu("Perform a standard maths operation on two columns. Choose the second columns.",
                                       df.columns.tolist())
        multi_column_operation(df, first_columns, second_columns, operator)
    if first_columns and not (operator == "=") and second_columns:
        messagebox.showinfo("Multi column Operations",
                            "New columns have been created.")
    else:
        messagebox.showinfo("Multi column operations", "Something went wrong.")


def rename_columns():
    global df
    col_to_rename = button_menu(
        "Choose which variable shall be renamed.", df.columns)
    new_name = get_input("the new column name", False)
    df = df.rename(columns={col_to_rename: new_name})
    if col_to_rename:
        messagebox.showinfo(
            "Renamed Column", f"Old name: {col_to_rename}; new name: {new_name}")
    else:
        messagebox.showinfo("Renamed Column", "No column was renamed.")


def remove_columns():
    global df
    cols_to_del = checkbox_menu(
        "Choose which variables shall be removed.", df.columns.tolist())
    for col in cols_to_del:
        df = df.drop(col, axis=1)

    if cols_to_del:
        messagebox.showinfo("Removed Columns",
                            "Selected columns have been deleted.")
    else:
        messagebox.showinfo("Removed Columns", "No columns deleted.")


def previous_averages():
    global df
    # check if a time column exists
    if not any(df.dtypes == 'datetime64[ns]'):
        messagebox.showinfo("Error",
                            "This cannot be executed. First specify a time column using 'Time transformation'.")
    else:
        target_cols = checkbox_menu("Choose on which variables rolling counts, sums, averages and SDs be performed.",
                                    df.columns.tolist())
        time_ref_col = button_menu(
            "Choose to which time variable to refer.", df.columns, show_datetime=False)
        condition_col = button_menu(
            "Choose which variable contains the condition (typically an id).", df.columns)
        days_back = get_input("the days to look backwards")
        if days_back:
            for col in target_cols:
                rolling_window(df, col, time_ref_col, condition_col, days_back)

        if target_cols and days_back:
            messagebox.showinfo(
                "Rolling Window", "Rolling operations have been performed on selected columns.")
        else:
            messagebox.showinfo(
                "Rolling Window", "No columns selected or no number provided.")


def calc_delta_abs():
    global df
    target_cols = checkbox_menu(
        "Choose on which variables the differences shall be calculated.", df.columns.tolist())
    condition_col = button_menu(
        "Choose which variable contains the condition (typically an id).", df.columns)
    rows_up = get_input("the rows to look upwards (typically 1)")
    if rows_up:
        for col in target_cols:
            delta(df, col, condition_col, rows_up)

    if target_cols and rows_up:
        messagebox.showinfo(
            "Delta Calculation", "Differences on the selected columns have been calculated.")
    else:
        messagebox.showinfo("Delta Calculation",
                            "No columns selected or no number provided.")


def replace_nan():
    global df
    repl_value = get_input("a number to replace 'NaN' and 'Div/0' with")
    nan_replacement(df, repl_value)
    messagebox.showinfo(
        "NaN Replacement", f"All 'NaN' and 'Div/0' have been replaced with {repl_value}.")


def single_correlations():
    global df
    while not label_target_var:
        set_target_var()

    single_correl_selection = radio_menu(
        "Choose the datatype to get the tables.")
    min_count = get_input(
        "the minimum number of categorical observations to get shown")
    num_buckets = get_input(
        "into how many buckets continuous variables shall be split")
    part1 = single_correl_num(
        df, single_correl_selection, num_buckets, label_target_var)
    single_correl_cat(df, single_correl_selection, min_count,
                      num_buckets, label_target_var, part1)

    messagebox.showinfo("Single Correlations",
                        "A csv file showing the single target correlations was created.")


def correlation_matrix_chart():
    global df
    correl_matrix = df.corr(numeric_only=True)
    sns.set(rc={'figure.figsize': (20, 20)})
    sns.heatmap(correl_matrix, annot=True, cmap='YlOrRd')
    plt.show()


def choose_algorithm():
    which_model = button_menu("Which algorithm shall be used?",
                              ["All Tree Models", "Random Forest", "Extra Trees", "AdaBoost", "Gradient Boost", "XGBoost", "Logistic Regression", "k-means"])
    if which_model == "All Tree Models":
        tree_algorithms = [RandomForestClassifier, ExtraTreesClassifier,
                           AdaBoostClassifier, GradientBoostingClassifier, xgb.XGBClassifier]
        get_params = True
        for clf in tree_algorithms:
            train_tree_classifier(clf, get_params)
            get_params = False
    elif which_model == "Random Forest":
        train_tree_classifier(RandomForestClassifier)
    elif which_model == "Extra Trees":
        train_tree_classifier(ExtraTreesClassifier)
    elif which_model == "AdaBoost":
        train_tree_classifier(AdaBoostClassifier)
    elif which_model == "Gradient Boost":
        train_tree_classifier(GradientBoostingClassifier)
    elif which_model == "XGBoost":
        train_tree_classifier(xgb.XGBClassifier)
    elif which_model == "Logistic Regression":
        train_logistic_regression()
    elif which_model == "k-means":
        train_kMeans()
    else:
        return


def time_transformation(array, label_name, time_trans_kinds):
    """
    transforms columns with dates into new columns showing the year, the month, the day and the weekday
    :param array: the data frame object (df)
    :param label_name:
    :param time_trans_kinds:
    :return:
    """
    global df

    # careful - always check what is month and what is day and swap if necessary

    col_position = array.columns.get_loc(label_name)
    array[label_name] = pd.to_datetime(
        array[label_name], dayfirst=days_in_dates_first)
    array.insert(col_position, label_name + "_daysSince1900", array[label_name].astype(
        np.int64) // 10 ** 9 / 24 / 60 / 60 + 25569)  # converts the values to seconds since the Unix epoch (January 1, 1970) and then back into an Excel-like number_format

    if time_trans_kinds == "All":
        array.insert(col_position, label_name + "_hour",
                     pd.to_datetime(array[label_name]).dt.hour, True)
        array.insert(col_position, label_name + "_weekday", pd.to_datetime(array[label_name]).dt.weekday + 1,
                     True)  # + 1 because in Python Monday is 0 and Sunday is 6
        array.insert(col_position, label_name + "_day",
                     pd.to_datetime(array[label_name]).dt.day, True)
        array.insert(col_position, label_name + "_month",
                     pd.to_datetime(array[label_name]).dt.month, True)
        array.insert(col_position, label_name + "_year",
                     pd.to_datetime(array[label_name]).dt.year, True)
    else:
        array = array.drop(label_name, axis=1)

    df = array


def sort_data(array, first_rank, second_rank, third_rank):
    """
    sort data by first_rank, second_rank and third rank - enter "none" if less ranks needed
    :param array: the data frame object (df)
    :param first_rank:
    :param second_rank:
    :param third_rank:
    :return:
    """
    global df

    if not second_rank:
        array.sort_values(by=[first_rank], inplace=True)
    elif not third_rank:
        array.sort_values(by=[first_rank, second_rank], inplace=True)
    else:
        array.sort_values(
            by=[first_rank, second_rank, third_rank], inplace=True)
    df = array


def rolling_window(array, target_column, time_ref_column, condition_column, days_backwards):
    """
    creates previous time counts, sums, averages and SDs on a specified column and a specified time to look backwards
    :param array: the data frame object (df)
    :param target_column: the target_column on which the operations shall be performed
    :param time_ref_column: the column which contains the time stamp of the transaction or view
    :param condition_column: the column which is the =SUMIF(); =AVERAGEIF(), etc. condition, e.g customer_id
    :param days_backwards: the number of days looking backwards to sum, average, etc.
    :return:
    """
    global df
    rolling_column = \
        array.set_index(time_ref_column).groupby(condition_column).rolling(window=str(days_backwards) + 'D',
                                                                           closed="left")[
            target_column].agg(['count', 'sum', 'mean', 'max', 'min', 'std']).reset_index()

    array = pd.merge(array, rolling_column, on=[
                     time_ref_column, condition_column])

    df = array

    df.rename(columns={"count": target_column + "_count_prev" + str(days_backwards) + "d",
                       "sum": target_column + "_sum_prev" + str(days_backwards) + "d",
                       "mean": target_column + "_mean_prev" + str(days_backwards) + "d",
                       "max": target_column + "_max_prev" + str(days_backwards) + "d",
                       "min": target_column + "_min_prev" + str(days_backwards) + "d",
                       "std": target_column + "_std_prev" + str(days_backwards) + "d"}, inplace=True)

    # FutureWarning: Dropping of nuisance columns in rolling operations is deprecated; in a future version this will
    # raise TypeError. Select only valid columns before calling the operation. Dropped columns were Index(['dob'],
    # dtype='object')

    # careful, currently the merging seems to create some duplicates if there is more than one transaction with same condition and same time stamp

    # array[avg_column_name] = array[avg_column_name].fillna(-1)  # replace #N/A with -1


def column_operation(col1, col2, operator):
    """
    Mathematically connects two columns by a chosen operator - "+", "-", "*" or "/"
    :param array: the data frame object (df)
    :param label1: the first column
    :param label2: the second column
    :param operator: "+", "-", "*" or "/"
    :return:
    """
    # global df
    calc_column = None
    # new_column_name = "test"
    # calc_column = pd.Series(dtype=float)  # a new column to store the new data
    # array.insert(array.columns.get_loc(label2) + 1, new_column_name,
    #              array.pop(new_column_name))  # move the new column from the end just after the avg_target column
    if operator == "+":
        calc_column = col1 + col2
    elif operator == "-":
        calc_column = col1 - col2
    elif operator == "*":
        calc_column = col1 * col2
    elif operator == "/":
        calc_column = col1 / col2
    elif operator == "&":
        calc_column = col1.astype(str).str.cat(col2.astype(str), sep="_")
    else:
        return
    return calc_column


def multi_column_operation(array, labels1, labels2, operator):
    """
    Mathematically connects two columns by a chosen operator - "+", "-", "*" or "/". Can be performed on multiple columns simultaneously.
    :param array: the data frame object (df)
    :param labels1: all first columns
    :param labels2: all second columns
    :param operator: "+", "-", "*" or "/"
    :return:
    """
    global df

    for i1 in labels1:
        for i2 in labels2:
            new_column_name = i1 + "_" + operator + "_" + i2
            # a new column to store the new data
            array[new_column_name] = pd.Series(dtype=float)
            array.insert(array.columns.get_loc(i2) + 1, new_column_name,
                         array.pop(
                             new_column_name))  # move the new column from the end just after the avg_target column
            if operator == "+":
                array[new_column_name] = array[i1] + array[i2]
            elif operator == "-":
                array[new_column_name] = array[i1] - array[i2]
            elif operator == "*":
                array[new_column_name] = array[i1] * array[i2]
            elif operator == "/":
                array[new_column_name] = array[i1] / array[i2]
            elif operator == "&":
                array[new_column_name] = array[i1].astype(
                    str).str.cat(array[i2].astype(str), sep="_")
    df = array


def info_stats():
    global df
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)
    print("Data")
    print()
    print(df.head(10000))
    print()
    print('The shape of our data is (rows x columns):', df.shape)
    print()
    print("Percentiles, etc.")
    print(df.describe())
    print()
    print("Column names, non-null counts and data types per column")
    print(df.info())
    print()
    print("Number of unique values per column")
    print(df.nunique())
    print()
    main_menu()


def remove_unwanted_datatypes(array):
    """
    removes the target variable and unwanted datatypes which cannot be handled by random forest
    :param array: the data frame object (df)
    :return:
    """
    if label_target_var is not None:
        # removes the target variable from the feature list
        array = array.drop(label_target_var, axis=1)
    array = array.select_dtypes(exclude=['object', 'datetime64[ns]'])
    return array


def dummy_variables(array, column_to_dummy, min_number_values):
    """
    creates dummy variables [0, 1] of specified columns and inserts them at the end; removes the original column
    :param array:
    :param column_to_dummy:
    :param min_number_values: the minimum number of expressions to qualify for a single dummy column
    :return:
    """

    global df

    cats = array[column_to_dummy].value_counts(
    )[lambda x: x >= min_number_values].index

    df = pd.concat(
        [array, pd.get_dummies(pd.Categorical(
            array[column_to_dummy], categories=cats), prefix=column_to_dummy)],
        axis=1)


def simple_math_trans_variables(array, col_to_calc):
    """
    Calculates SQRT(col_to_calc), 1/col_to_calc, LN(col_to_calc) and col_to_calc^2 on specified columns and inserts them at the end; does not remove the original column
    :param array:
    :param col_to_calc:
    :return:
    """
    global df
    # think of adding 1 or more to avoid Div/0, LN(0), LN(neg), SQRT(neg)

    sqrt_col = np.sqrt(array[col_to_calc])
    inv_col = 1 / array[col_to_calc]
    ln_col = np.log(array[col_to_calc])
    square_col = array[col_to_calc] ** 2

    # insert the new columns after col_to_calc
    array.insert(loc=array.columns.get_loc(col_to_calc) + 1,
                 column='SQRT(' + col_to_calc + ')', value=sqrt_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 2,
                 column='1/' + col_to_calc, value=inv_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 3,
                 column='LN(' + col_to_calc + ')', value=ln_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 4,
                 column=col_to_calc + '^2', value=square_col)

    df = array


def if_comparison():
    global df
    sel_col_label = button_menu("Select a column.", df.columns.tolist())
    sel_col = df[sel_col_label]

    num_unique = sel_col.nunique()

    show_threshold = top_unique_value_threshold

    if sel_col.dtype == object or num_unique <= 5:

        if num_unique > show_threshold:
            show_threshold = get_input(
                "the number of top value counts to be shown")

        # Get the unique values and their counts in the column
        value_counts = sel_col.value_counts()

        # Get the top values based on count of appearance
        unique_values = value_counts.index.tolist()[:show_threshold]

        sel_expressions = checkbox_menu(
            "Select the expressions.", unique_values)
        logic = button_menu("Choose the logic.", ["OR", "NOT"])
        new_col_name = "multiple"
        if len(sel_expressions) < 5:
            new_col_name = '_'.join(str(expr) for expr in sel_expressions)

        if logic == "OR":
            df[sel_col_label + "_" + logic + "_" +
                new_col_name] = df[sel_col_label].isin(sel_expressions).astype(int)
        else:
            df[sel_col_label + "_" + logic + "_" + new_col_name] = (~df[sel_col_label].isin(sel_expressions)).astype(
                int)

        new_col_name = sel_col_label + "_" + logic + "_" + new_col_name

    else:
        # Get summary statistics for the numeric column
        summary_stats = [np.min(sel_col), np.max(sel_col), np.mean(sel_col), np.median(sel_col),
                         np.std(sel_col)]
        comp_choice, value1, value2 = math_comp_menu(
            "Choose the logic.", summary_stats)
        value1 = float(value1)
        if value2 is not None:
            value2 = float(value2)

            if value2 < value1:
                value_temp = value2
                value2 = value1
                value1 = value_temp

        if comp_choice == "between":
            df[f"{sel_col_label}_{comp_choice}_{value1}_{value2}"] = (
                (sel_col >= float(value1)) & (sel_col <= float(value2))).astype(int)
        elif comp_choice == "not between":
            df[f"{sel_col_label}_{comp_choice}_{value1}_{value2}"] = (
                ~(sel_col.between(float(value1), float(value2)))).astype(int)
        elif comp_choice == "=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col == float(value1)).astype(int)
        elif comp_choice == "!=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col != float(value1)).astype(int)
        elif comp_choice == "<":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col < float(value1)).astype(int)
        elif comp_choice == "<=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col <= float(value1)).astype(int)
        elif comp_choice == ">":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col > float(value1)).astype(int)
        elif comp_choice == ">=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col >= float(value1)).astype(int)

        new_col_name = sel_col_label + "_" + comp_choice + "_" + str(value1)
        if value2 is not None:
            new_col_name = new_col_name + "_" + str(value2)

    messagebox.showinfo(
        "New Column", f"Column {new_col_name} has been created.")

    return


def delta(array, ref_column, condition_column, look_upwards):
    """
    calculates the difference (delta) to the previous value of the same column
    :param array: the data frame object (df)
    :param ref_column: the column the delta shall be calculated on
    :param condition_column: the column indicating the condition, e.g. customer_id
    :param look_upwards: the number of rows to look up
    """
    global df

    # calculate absolute difference
    array[ref_column + "_delta_abs"] = array.groupby(
        condition_column)[ref_column].diff(periods=look_upwards)

    # calculate relative difference

    array[ref_column + "_delta_rel"] = (
        array.groupby(condition_column, group_keys=False)[ref_column]
        .apply(lambda x: x.pct_change(periods=look_upwards))
    )

    df = array


def load_from_file():
    global df, df_in_use, df_names

    root = tk.Tk()
    root.withdraw()

    # Set the starting directory to the current working directory
    start_dir = os.getcwd()

    # Show the file selection dialog box and get the selected file
    file_path = filedialog.askopenfilename(initialdir=start_dir, title="Select a file",
                                           filetypes=(("CSV files", "*.csv"),
                                                      ("Excel files", ("*.xls",
                                                       "*.xlsx", "*.xlsb")),
                                                      ("All files", "*.*")),
                                           initialfile="*.csv;*.xls;*.xlsx;*.xlsb")

    root.destroy()
    if not file_path:
        messagebox.showinfo("Information", "No file selected.")
    else:
        # Load the selected file as a pandas dataframe
        if df_in_use > -1:
            df_list[df_in_use] = df
        df_in_use += 1

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx', '.xlsb']:
            df = pd.read_excel(file_path)
        else:
            messagebox.showinfo("Information", "Unsupported file format.")
            return

        print("Selected file:", file_path)
        print("Dataframe shape:", df.shape)

        df_names.append(get_input("a name for the dataset", False))
        df_list.append(df)
        messagebox.showinfo("Information", "Data successfully loaded.")
    main_menu()


def save_to_csv():
    global df
    root = tk.Tk()
    root.withdraw()

    # Set the starting directory to the current working directory
    start_dir = os.getcwd()

    # Show the file save dialog box and get the selected file name and location
    file_path = filedialog.asksaveasfilename(initialdir=start_dir, title="Save as...",
                                             filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    root.destroy()
    if not file_path:
        messagebox.showinfo("Information", "Document not saved.")
    else:
        # Check if the file path has the .csv extension
        if not file_path.lower().endswith('.csv'):
            # Append .csv extension to the file path
            file_path += '.csv'
        # Save the dataframe as a CSV file at the selected location
        df.to_csv(file_path, index=False)
        messagebox.showinfo(
            "Information", f"Data successfully saved in '{file_path}'.")
        print("Saved file:", file_path)


def nan_replacement(array, replacement_value):
    global df
    array.replace(to_replace=np.nan, value=replacement_value, inplace=True)
    array.replace(to_replace=np.inf, value=replacement_value, inplace=True)
    array.replace(to_replace=-np.inf, value=replacement_value, inplace=True)
    df = array


def single_correl_num(df, radio_array, n, target_col):
    """Find the n boundaries for each column in the given pandas dataframe.

    Args:
        df: pandas DataFrame object with numerical and non-numerical columns
        n: number of buckets to split the data into (default 10)
        target_var: name of the column to count '1' values in (default None)
        output_file: path to CSV file to write the bucket information to (default None)

    Returns:
        A dictionary with column names as keys and a list of n split values as values.
    """
    result = {}
    bucket_info = []
    for col in df.columns:
        if df[col].dtype.kind in 'biufc' and radio_array.get(col, "") == "numerical" and col != target_col and df[
                col].nunique() > n:

            # overwrite blanks with a large negative value or so
            df[col] = df[col].fillna(overwrite_nan)

            boundaries = np.percentile(df[col], np.linspace(0, 100, n + 1))
            result[col] = list(boundaries)
            counts, _ = np.histogram(df[col], bins=boundaries)
            bucket_info.append([col, "Count", "Sum of 1s", "Average"])
            total_bucket_count = 0
            total_target_count = 0
            for i, bucket in enumerate(range(1, n + 1)):
                lower_bucket = result[col][i]
                upper_bucket = result[col][i + 1]
                bucket_count = counts[i]
                total_bucket_count += bucket_count

                if i < n - 1:
                    bucket_values = df[col][(df[col] >= lower_bucket) & (
                        df[col] < upper_bucket)]
                    upper_bucket_label = "<"
                else:
                    bucket_values = df[col][
                        (df[col] >= lower_bucket) & (df[col] <= upper_bucket)]  # the last bucket shall include <=
                    upper_bucket_label = "<="

                # can't handle dates

                target_count = len(
                    bucket_values[bucket_values.index.isin(df[df[target_col] == 1].index)])
                total_target_count += target_count
                if upper_bucket > lower_bucket:  # prevents incl. empty rows
                    if col[
                       -13:] != "_daysSince1900":  # shall ensure that dates are presented like dates in the else statement
                        bucket_info.append(
                            [f"{lower_bucket:.2f}_-_{upper_bucket_label}{upper_bucket:.2f}", bucket_count, target_count,
                             target_count / bucket_count])
                    else:
                        dt = datetime(1899, 12, 30) + \
                            timedelta(days=int(lower_bucket))
                        date_str_lower = dt.strftime('%d.%m.%Y')
                        dt = datetime(1899, 12, 30) + \
                            timedelta(days=int(upper_bucket))
                        date_str_upper = dt.strftime('%d.%m.%Y')
                        bucket_info.append(
                            [f"{date_str_lower}_-_{upper_bucket_label}{date_str_upper}", bucket_count, target_count,
                             target_count / bucket_count])

            bucket_info.append(
                ["Total", total_bucket_count, total_target_count, total_target_count / total_bucket_count])
            bucket_info.append(["", ""])
        else:
            result[col] = None  # Ignore non-numerical columns

    columns = ["Bucket Range", "Bucket Count", "Target Count", "Target Ratio"]
    df_bucket_info = pd.DataFrame(bucket_info, columns=columns)
    return df_bucket_info


def single_correl_cat(df, radio_array, min_count, n, target_col, df_bucket_info):
    result = []
    for col in df.columns:
        total_count = 0
        total_sum = 0
        if (radio_array.get(col, "") == "categorical" or (
                radio_array.get(col, "") == "numerical" and df[col].nunique() <= n)) and col != target_col:

            # to get <blank> shown separately
            df[col] = df[col].fillna("<blank>")

            grouped = df.groupby(col)[target_col].agg(['sum', 'count'])
            grouped['average'] = grouped['sum'] / grouped['count']
            grouped = grouped.reset_index().rename(
                columns={col: 'column', 'sum': 'sum_of_ones', 'count': 'total_count'})

            # Group expressions with count < min_count into "other" category
            if min_count > 0 and radio_array.get(col, "") == "categorical":
                other_group = grouped[grouped['total_count']
                                      < min_count].copy()
                other_group['column'] = 'other'
                other_group = other_group.groupby('column').sum().reset_index()
                grouped = pd.concat(
                    [grouped[grouped['total_count'] >= min_count], other_group], ignore_index=True)

            # Add a new column to sort by "Other" groups last
            grouped['sort_order'] = grouped['column'].apply(
                lambda x: 1 if x == 'other' else 0)

            # Sort by total count and sort order
            if radio_array.get(col, "") == "categorical":
                grouped = grouped.sort_values(
                    by=['sort_order', 'total_count'], ascending=[True, False])
            else:
                grouped = grouped.sort_values(
                    by=['sort_order', 'column'], ascending=[True, True])
            grouped = grouped.drop('sort_order', axis=1)

            # Append column name as new row
            result.append(
                pd.DataFrame([[col, '', '', '']], columns=['column', 'total_count', 'sum_of_ones', 'average']))

            # Append grouped dataframe
            result.append(
                grouped[['column', 'total_count', 'sum_of_ones', 'average']])

            # Append total row
            total_count += grouped['total_count'].sum()
            total_sum += grouped['sum_of_ones'].sum()
            total_average = total_sum / total_count

            result.append(
                pd.DataFrame([['Total', total_count, total_sum, total_average]],
                             columns=['column', 'total_count', 'sum_of_ones', 'average']))

            # Append a blank row
            result.append(
                pd.DataFrame([['', '', '', '']], columns=['column', 'total_count', 'sum_of_ones', 'average']))

    # Concatenate all results and save to CSV
    result_df = pd.concat(result, axis=0)
    result_df = pd.concat([result_df, df_bucket_info], ignore_index=True)
    result_df.to_csv(save_csv_single_correl, index=False)


def merge_dataframes(df1, df2, identifier, join_type="left"):
    global df, df_in_use, df_names, df_list
    # Inner join: only rows with matching values in both input DataFrames are included in the merged DataFrame.

    # Left join: all rows from the left input DataFrame and matching rows from the right input DataFrame are included
    # in the merged DataFrame. If there are no matching rows in the right input DataFrame, the columns from the right
    # input DataFrame are filled with NaN values.

    # Right join: all rows from the right input DataFrame and matching rows from the left input DataFrame are included
    # in the merged DataFrame. If there are no matching rows in the left input DataFrame, the columns from the left
    # input DataFrame are filled with NaN values.

    # Outer join: all ows from both input DataFrames are included in the merged DataFrame. If there are no matching
    # rows in one of the input DataFrames, the columns from that input DataFrame are filled with NaN values.

    df_names.append(df_names[df_in_use] + "_" +
                    join_type + "_join_" + df_names[df_in_use])
    df_list[df_in_use] = df
    df_in_use += 1
    df = pd.merge(df1, df2, on=identifier, how=join_type)

    # To merge on multiple columns, pass a list of column names to the on parameter: on=["column1", "column2"].

    df_list.append(df)


def train_tree_classifier(classifier, get_params=True):
    global df, label_target_var, param_dict

    # find the used algorithm

    if classifier == RandomForestClassifier:
        classifier_name = "Random_Forest"
    elif classifier == ExtraTreesClassifier:
        classifier_name = "Extra_Trees"
    elif classifier == AdaBoostClassifier:
        classifier_name = "AdaBoost"
    elif classifier == GradientBoostingClassifier:
        classifier_name = "Gradient_Boosting"
    elif classifier == xgb.XGBClassifier:
        classifier_name = "XGBoost"
    else:
        classifier_name = "unknown"

    print("Training", classifier_name, "classifier...")

    nan_replacement(df, -999999)
    while not label_target_var:
        set_target_var()

    y = df[label_target_var]
    x = remove_unwanted_datatypes(df)

    if get_params:
        param_dict = get_tree_params()

    number_of_trees = int(param_dict['number of trees'])
    max_depth_of_tree = int(param_dict['max depth of tree'])
    min_samples_to_split_node = int(param_dict['min samples to split node'])
    min_samples_to_be_in_leaf = int(param_dict['min samples to be in leaf'])
    train_split = param_dict['share training set']
    confusion_matrix_cutoff = param_dict['confusion matrix cutoff']
    show_feature_importance = param_dict['show feature importance']
    show_ROC = param_dict['show ROC']
    draw_and_save_trees = param_dict['draw and save trees']
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    if save_data:
        df.to_csv(save_csv_name, index=False)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_split, random_state=rnd_state)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame()

    # Add columns for number of observations, set (train or test), target variable, and model score
    results_df['Observation'] = list(range(len(x_train) + len(x_test)))
    results_df['Set'] = ['Train'] * len(x_train) + ['Test'] * len(x_test)
    results_df['Target'] = list(y_train) + list(y_test)
    results_df['Model Score'] = 0.0

    if classifier == AdaBoostClassifier:  # exclude parameters max_depth, min_samples_split, min_samples_leaf
        clf = classifier(random_state=rnd_state, n_estimators=number_of_trees)
    else:
        clf = classifier(random_state=rnd_state, n_estimators=number_of_trees, max_depth=max_depth_of_tree,
                         min_samples_split=min_samples_to_split_node, min_samples_leaf=min_samples_to_be_in_leaf)

    clf.fit(x_train, y_train)

    # Get the predicted probabilities for the positive class (index 1)
    # Probability of positive class for training set
    train_predicted_probs = clf.predict_proba(x_train)[:, 1]
    # Probability of positive class for test set
    test_predicted_probs = clf.predict_proba(x_test)[:, 1]

    # Assign the predicted values to the 'Model Score' column of the DataFrame
    results_df['Model Score'] = np.concatenate(
        (train_predicted_probs, test_predicted_probs))

    # Save the DataFrame to a CSV file
    results_df.to_csv('model_results.csv', index=False)

    if show_feature_importance:
        # Get feature importance and sort them in descending order
        importance = clf.feature_importances_
        indices = np.argsort(importance)[::-1]

        # Plot the feature importance
        plt.figure()
        plt.title("Feature importance")
        plt.bar(range(x.shape[1]), importance[indices],
                color="r", align="center")
        plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.show()

    y_pred = (clf.predict_proba(x_test)[
              :, 1] > confusion_matrix_cutoff).astype('float')
    cm = metrics.confusion_matrix(y_test, y_pred)
    cr = metrics.classification_report(y_test, y_pred)
    print(cr)
    print(cm)

    print("Accuracy Test Set", metrics.accuracy_score(y_test, y_pred))

    if show_ROC:
        y_pred_proba = clf.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="AUC Test Set = " + str(auc))
        plt.legend(loc=4)
        plt.show()

    if draw_and_save_trees:
        fig = plt.figure(figsize=(30, 20))

        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html

        for i in range(number_trees_pics):
            plot_tree(clf.estimators_[i],
                      feature_names=x_train.columns, max_depth=trees_pics_depth,
                      filled=True, impurity=True,
                      rounded=True)
            fig.savefig('figure_name' + str(i) + '.png')

    if save_model:
        joblib.dump(clf, "./" + classifier_name.lower() + ".joblib")


def train_logistic_regression():
    global df
    global label_target_var

    def drop_cols(data, target_col, collinear_threshold=0.65, insignificance_threshold=0.00):
        # compute the correlation matrix
        corr_matrix = data.corr(numeric_only=True)
        print(corr_matrix)

        # create a set to store the columns to drop
        cols_to_drop = set()

        # iterate over the columns
        for col in corr_matrix.columns:
            if col == target_col:
                continue
            # check if the column correlates strongly with another column
            strong_corr = (corr_matrix[col][:-1].abs()
                           > collinear_threshold).sum() > 1
            if strong_corr:
                # check if the column correlates more weakly with the target variable
                other_cols = corr_matrix[abs(
                    corr_matrix[col]) > collinear_threshold].index.tolist()
                other_cols.remove(col)
                max_corr = corr_matrix.loc[other_cols, target_col].abs().max()
                if max_corr >= abs(corr_matrix.loc[col, target_col]):
                    # if max_corr is equal to the correlation between the column and target variable
                    # choose the column with the higher index to drop
                    if abs(max_corr - abs(corr_matrix.loc[col, target_col])) <= 1e-10:
                        drop_col = max(
                            col, corr_matrix.loc[other_cols, target_col].abs().idxmax())
                    else:
                        drop_col = col
                    cols_to_drop.add(drop_col)
            else:
                # check if the column correlates weakly with the target variable
                corr_with_target = corr_matrix.loc[col, target_col]
                if abs(corr_with_target) < insignificance_threshold:
                    cols_to_drop.add(col)

        # drop the columns and return the updated DataFrame
        return data.drop(cols_to_drop, axis=1)

    nan_replacement(df, -999999)
    while not label_target_var:
        set_target_var()

    param_dict = get_lr_params()
    num_iter = int(param_dict['number of iterations'])
    collinearity = param_dict['collinearity threshold']
    insignificance = param_dict['insignificance threshold']
    train_split = param_dict['share training set']
    confusion_matrix_cutoff = param_dict['confusion matrix cutoff']
    show_ROC = param_dict['show ROC']
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    y = df[label_target_var]
    x = remove_unwanted_datatypes(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_split, random_state=rnd_state)

    # bring the target variable in for correl matrix
    train_data_incl_target = pd.concat([x_train, y_train], axis=1)

    x_train = drop_cols(train_data_incl_target, label_target_var, collinearity, insignificance).drop(label_target_var,
                                                                                                    axis=1)  # drop it again

    print(x_train)

    # drop the same columns from the test set, i.e. retain only those which are still in the training set
    common_cols_train_test = x_train.columns.intersection(x_test.columns)
    x_test = x_test[common_cols_train_test]

    # create a logistic regression model
    lr = LogisticRegression(max_iter=num_iter, random_state=rnd_state)

    print(x_train)
    print(y_train)

    # train the model on the training set
    lr.fit(x_train, y_train)

    # make predictions on the testing set
    y_pred = lr.predict(x_test)

    # evaluate the performance of the model

    print("Accuracy Test Set", metrics.accuracy_score(y_test, y_pred))

    # get the coefficients and intercept
    coef = lr.coef_
    intercept = lr.intercept_

    # create a list of tuples containing the feature names and coefficients
    coef_list = list(zip(x_train.columns, coef[0]))

    # insert the intercept at the beginning of the coef_list
    coef_list.insert(0, ("Intercept", intercept[0]))

    # print the coefficients and intercept in a table format
    headers = ["Feature", "Coefficient"]
    table = tabulate(coef_list, headers=headers, tablefmt="fancy_grid")
    print(table)

    y_pred = (lr.predict_proba(x_test)[:, 1] >
              confusion_matrix_cutoff).astype('float')
    cmrfc = metrics.confusion_matrix(y_test, y_pred)
    crrfc = metrics.classification_report(y_test, y_pred)
    print(crrfc)
    print(cmrfc)

    if show_ROC:
        y_pred_proba = lr.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="AUC Test Set = " + str(auc))
        plt.legend(loc=4)
        plt.show()


def train_kMeans():
    df_kMeans = df
    headers = df_kMeans.columns

    param_dict = get_kMeans_params()

    calc_elbow = param_dict['show elbow chart']
    min_groups = int(param_dict['min groups'])
    max_groups = int(param_dict['max groups'])
    number_init = int(param_dict['number random initiations'])
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    df_kMeans = remove_unwanted_datatypes(df_kMeans)

    # Normalize the continuous variables >> perhaps make this optional
    scaler = StandardScaler()
    # unfortunately this converts the df object into a numpy array
    df_kMeans = scaler.fit_transform(df_kMeans)

    # Fit KMeans model to the data
    if calc_elbow:
        inertias = []
        for num_clusters in range(min_groups, max_groups + 1):
            kmeans = KMeans(n_clusters=num_clusters,
                            n_init=number_init, random_state=1).fit(df_kMeans)
            inertias.append(kmeans.inertia_)

        # Plot elbow chart
        plt.plot(range(min_groups, max_groups + 1), inertias)
        plt.title('Elbow chart')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        # Choose the number of clusters you want to create
        num_clusters = get_input("the number of clusters")
    else:
        num_clusters = min_groups

    # Create a k-means object
    kmeans = KMeans(n_clusters=num_clusters, n_init=number_init,
                    random_state=1).fit(df_kMeans)

    # Add the cluster labels to the numpy array
    labels = (kmeans.labels_ + 1).reshape(-1, 1)
    result = np.hstack((df_kMeans, labels))
    df_kMeans = pd.DataFrame(result, columns=list(headers) + ['cluster'])

    # measures how well each point fits into its assigned cluster, based on the distance between the point and the
    # neighboring clusters. A higher silhouette score indicates that the point is well-clustered, while a lower score
    # suggests that the point may belong to a different cluster. >> between -1 and 1
    siL_score = metrics.silhouette_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    # measures the ratio of between-cluster variance to within-cluster variance. A higher Calinski-Harabasz index
    # indicates better-defined clusters.
    c_h_score = metrics.calinski_harabasz_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    # average similarity between each cluster and its most similar
    # cluster, relative to the average dissimilarity between each cluster and its least similar cluster
    d_b_score = metrics.davies_bouldin_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    print(f"Silhouette Score: {siL_score}")  # >0.5
    print(f"Calinski-Harabasz Score: {c_h_score}")  # >100
    print(f"Davies-Bouldin Score: {d_b_score}")  # <1.0

    # Retrieve the cluster centers
    centers = kmeans.cluster_centers_

    df_centers = pd.DataFrame(centers, columns=list(headers))
    df_centers['center'] = ['Center{}'.format(
        i) for i in range(1, kmeans.n_clusters + 1)]

    # Add the centers to the dataframe with their respective cluster names
    df_kMeans = pd.concat([df_kMeans, df_centers], axis=0, ignore_index=True)

    df_kMeans = df_kMeans.sort_values(by='center', ascending=True)

    # Replace the empty "cluster" values for the centers with the corresponding "center" value
    df_kMeans['cluster'].fillna(df_kMeans['center'], inplace=True)

    df_kMeans = df_kMeans.drop('center', axis=1)

    messagebox.showinfo("k-means", "Model trained.")

    # This code calculates the cosine similarity between the i-th and j-th rows of the X dataframe. The resulting
    # cos_sim value will range between -1 and 1, with higher values indicating greater similarity. and i and j are
    # the indices of the two observations
    # cos_sim = cosine_similarity(df_kMeans.iloc[[5]], df_kMeans.iloc[[6]])[0][0]
    # print(f"cosine_similarity: {cos_sim}")

    # This code calculates the Euclidean distance between the i-th and j-th rows of the X dataframe. The resulting
    # euclidean_dist value will be non-negative, with lower values indicating greater similarity.
    # euclidean_dist = euclidean_distances(df_kMeans.iloc[[5]], df_kMeans.iloc[[6]])[0][0]
    # print(f"euclidean_dist: {euclidean_dist}")

    if save_data:
        df_kMeans.to_csv("kMeans_modeltest2.csv", index=False)
        messagebox.showinfo("k-means", "Data saved.")

    if save_model:
        joblib.dump(kmeans, 'kmeans_model.sav')

    # load the saved model from disk
    #     # load the saved k-means model and scaler object from disk
    #     kmeans_model = joblib.load('kmeans_model.sav')
    #     scaler = joblib.load('scaler.sav') >> to be saved
    #
    #     # load the new data
    #     new_data = pd.read_csv('new_data.csv')
    #
    #     # apply the same scaling used for the original data
    #     scaled_data = scaler.transform(new_data)
    #
    #     # use the k-means model to predict cluster labels for the new data
    #     predicted_labels = kmeans_model.predict(scaled_data)


def test():
    return


def if_comparison3():
    global df
    connector = "x"
    new_column_name = ""

    while not (connector == "="):
        if connector == "x":
            first_column, new_column_name = if_comparison2()

        connector = button_menu("Choose the connector.", connector_array)

        if not (connector == "="):
            second_column, second_col_name = if_comparison2()
            new_column_name = new_column_name + "__" + connector + "__" + second_col_name
            if connector == "AND":
                first_column = first_column * second_column
            elif connector == "OR":
                first_column = min(first_column + second_column, 1)

    df.insert(len(df.columns), new_column_name, first_column)

    messagebox.showinfo("Column Operations",
                        f"A new column {new_column_name} was created.")


def if_comparison2():
    global df
    sel_col_label = button_menu("Select a column.", df.columns.tolist())
    sel_col = df[sel_col_label]

    num_unique = sel_col.nunique()

    show_threshold = top_unique_value_threshold

    if sel_col.dtype == object or num_unique <= 5:

        if num_unique > show_threshold:
            show_threshold = get_input(
                "the number of top value counts to be shown")

        # Get the unique values and their counts in the column
        value_counts = sel_col.value_counts()

        # Get the top values based on count of appearance
        unique_values = value_counts.index.tolist()[:show_threshold]

        sel_expressions = checkbox_menu(
            "Select the expressions.", unique_values)
        logic = button_menu("Choose the logic.", ["OR", "NOT"])
        new_col_name = "multiple"
        if len(sel_expressions) < 5:
            new_col_name = '_'.join(str(expr) for expr in sel_expressions)

        if logic == "OR":
            new_column = df[sel_col_label].isin(sel_expressions).astype(int)
        else:
            new_column = (~df[sel_col_label].isin(sel_expressions)).astype(int)

        new_col_name = sel_col_label + "_" + logic + "_" + new_col_name

    else:
        # Get summary statistics for the numeric column
        summary_stats = [np.min(sel_col), np.max(sel_col), np.mean(sel_col), np.median(sel_col),
                         np.std(sel_col)]
        comp_choice, value1, value2 = math_comp_menu(
            "Choose the logic.", summary_stats)
        value1 = float(value1)
        if value2 is not None:
            value2 = float(value2)

            if value2 < value1:
                value_temp = value2
                value2 = value1
                value1 = value_temp

        if comp_choice == "between":
            new_column = ((sel_col >= float(value1)) & (
                sel_col <= float(value2))).astype(int)
        elif comp_choice == "not between":
            new_column = (
                ~(sel_col.between(float(value1), float(value2)))).astype(int)
        elif comp_choice == "=":
            new_column = (sel_col == float(value1)).astype(int)
        elif comp_choice == "!=":
            new_column = (sel_col != float(value1)).astype(int)
        elif comp_choice == "<":
            new_column = (sel_col < float(value1)).astype(int)
        elif comp_choice == "<=":
            new_column = (sel_col <= float(value1)).astype(int)
        elif comp_choice == ">":
            new_column = (sel_col > float(value1)).astype(int)
        elif comp_choice == ">=":
            new_column = (sel_col >= float(value1)).astype(int)

        new_col_name = sel_col_label + "_" + comp_choice + "_" + str(value1)
        if value2 is not None:
            new_col_name = new_col_name + "_" + str(value2)

        # messagebox.showinfo("New Column", f"Column {new_col_name} has been created.")

    return new_column, new_col_name


main_menu()

if load_model:
    loaded_rf = joblib.load("./random_forest.joblib")

    # predict new observation - here from file "newObs.csv"

    new_obs_pred = pd.read_csv(testing_file)

    new_obs_pred = time_transformation(new_obs_pred, time_stamp_column)
    new_obs_pred = time_transformation(new_obs_pred, "dob")

    new_obs_pred = sort_data(
        new_obs_pred, "customer_id", time_stamp_column, "none")

    # create a new column to store the rolling average

    # new_obs_pred = rolling_window(new_obs_pred, "amt", time_stamp_column, "customer_id", 2)

    y = new_obs_pred[label_target_var]
    x = remove_unwanted_datatypes(new_obs_pred)

    info_stats(x)

    # s = (
    #         X.dtypes == 'object')  # the ordinal numbering won't work here, because it's numbered inconsistently from the training data
    # object_cols = list(s[s].index)
    # ordinal_encoder = OrdinalEncoder()
    # X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

    # print("Data2 after time modifications and ordinal encoding")
    # print(X.head(19))
    # print()

    print(loaded_rf.predict(x))
