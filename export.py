"""
Code to facilitate the export of data resulting from the benchmarking experiments.
"""

import pandas as pd
import numpy as np
import os
import json

def find_benchmarking_files(root_dir):
    target_files = []
    for target in os.listdir(root_dir):
        target_path = os.path.join(root_dir, target)

        if os.path.isfile(target_path):
            if target_path.endswith(".json"):
                target_files.append(target_path)

    return target_files

def load_benchmarking_file_dict(file_path):
    
    file_dict = None
    with open(file_path, "r") as fp:
        file_dict = json.load(fp)

    return file_dict

def get_benchmark_experiment_names(benchmark_dict):
    # We want to find every experiment we want to potentially display in the excel file
    names = []

    for image_name, image_dict in benchmark_dict["images"].items():
        for experiment_name, experiment_dict in image_dict["benchmarks"].items():
            names.append(experiment_name)
    
    return names

def get_benchmark_image_sizes(benchmark_dict):
    sizes = []

    for _, image_dict in benchmark_dict["images"].items():
        size_x = image_dict["metadata"]["size_x"]
        size_y = image_dict["metadata"]["size_y"]
        shape = size_x, size_y
        if shape not in sizes:
            sizes.append(shape)
    
    return sizes


def extract_experiment_data(experiment_names, benchmarking_dicts, chosen_img_shape=None):
    # For each experiment name, get the average performance for each method
    data = {}
    valid_methods = []
    for exp_name in experiment_names:
        mse = []
        psnr = []
        ssim = []
        precision = []
        recall = []
        emb_time = []
        for benchmark_dict in benchmarking_dicts:
            # Results for the individual experiments
            exp_mse = []
            exp_psnr = []
            exp_ssim = []
            exp_precision = []
            exp_recall = []
            exp_emb_time = []
            for image in benchmark_dict["images"]:
                if chosen_img_shape is not None:
                    img_size_x = benchmark_dict["images"][image]["metadata"]["size_x"]
                    img_size_y = benchmark_dict["images"][image]["metadata"]["size_y"]
                    if (img_size_x, img_size_y) != chosen_img_shape:
                        continue

                exp_mse.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["MSE"])
                exp_psnr.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["PSNR"])
                exp_ssim.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["SSIM"])
                exp_recall.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["recall"])
                exp_precision.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["precision"])
                exp_emb_time.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["time"])

            # Average the lists if they are not empty
            if len(exp_mse) > 0:
                mse.append(float(np.round(np.average(exp_mse), decimals=4)))
                psnr.append(float(np.round(np.average(exp_psnr), decimals=4)))
                ssim.append(float(np.round(np.average(exp_ssim), decimals=4)))
                precision.append(float(np.round(np.average(exp_precision), decimals=4)))
                recall.append(float(np.round(np.average(exp_recall), decimals=4)))
                emb_time.append(float(np.round(np.average(exp_emb_time), decimals=4)))
    
        data[exp_name] = {"precision": precision, "recall": recall, 
                          "mse": mse, "psnr": psnr, "ssim": ssim, "emb_time": emb_time}
    
    return data


def extract_experiment_data_finegrained(experiment_names, benchmarking_dicts, chosen_img_shape=None):
    # For each experiment name, get the average performance for each method
    data = {}
    valid_methods = []
    for exp_name in experiment_names:
        mse = {}
        psnr = {}
        ssim = {}
        precision = {}
        recall = {}
        emb_time = {}
        for benchmark_dict in benchmarking_dicts:
            # Results for the individual experiments
            exp_mse = []
            exp_psnr = []
            exp_ssim = []
            exp_precision = []
            exp_recall = []
            exp_emb_time = []
            for image in benchmark_dict["images"]:
                if chosen_img_shape is not None:
                    img_size_x = benchmark_dict["images"][image]["metadata"]["size_x"]
                    img_size_y = benchmark_dict["images"][image]["metadata"]["size_y"]
                    if (img_size_x, img_size_y) != chosen_img_shape:
                        continue
                
                if exp_name not in benchmark_dict["images"][image]["benchmarks"].keys():
                    continue
                
                exp_mse.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["MSE"])
                exp_psnr.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["PSNR"])
                exp_ssim.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["SSIM"])
                exp_recall.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["recall"])
                exp_precision.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["precision"])
                exp_emb_time.append(benchmark_dict["images"][image]["benchmarks"][exp_name]["time"])

            # Average the lists if they are not empty
            if len(exp_mse) > 0:
                mse[benchmark_dict["method"]] = float(np.round(np.average(exp_mse), decimals=4))
                psnr[benchmark_dict["method"]] = float(np.round(np.average(exp_psnr), decimals=4))
                ssim[benchmark_dict["method"]] = float(np.round(np.average(exp_ssim), decimals=4))
                precision[benchmark_dict["method"]] = float(np.round(np.average(exp_precision), decimals=4))
                recall[benchmark_dict["method"]] = float(np.round(np.average(exp_recall), decimals=4))
                emb_time[benchmark_dict["method"]] = float(np.round(np.average(exp_emb_time), decimals=4))
    
        data[exp_name] = {"precision": precision, "recall": recall, 
                          "mse": mse, "psnr": psnr, "ssim": ssim, "emb_time": emb_time}
    
    return data


def index_to_xls(idx):
    rem = chr(ord('A') + (idx % 26))

    if idx > 25:
        # Forwarding part
        forward = ((idx - (idx % 26)) // 26) - 1
        return index_to_xls(forward) + rem
    else:
        return rem

def get_xls_range_string(col_start, col_stop, row_start, row_stop):
    """
    col_stop: Exclusive
    """
    return "{}{}:{}{}".format(index_to_xls(col_start), row_start, index_to_xls(col_stop - 1), row_stop)


def generate_xlsx(root_dir):
    target_file_paths = find_benchmarking_files(root_dir=root_dir)

    # Load all dictionaries
    benchmarking_dicts = list(map(lambda x: load_benchmarking_file_dict(x), target_file_paths))

    # Determine the order in which the results will always be given
    # For now just based on the directory order
    method_order = [os.path.split(file_path)[-1][:-5] for file_path in target_file_paths]

    # Determine the names of the experiments which have been used for the various methods
    # e.g. "text_attack" or "center_crop_10%"
    exp_names = []

    for benchmark_dict in benchmarking_dicts:
        applicable_experiments = get_benchmark_experiment_names(benchmark_dict=benchmark_dict)

        for name in applicable_experiments:
            if name not in exp_names:
                exp_names.append(name)
    
    img_sizes = []
    methods_per_shape = {}
    # Find all image shapes that are found in the benchmark dicts
    for benchmark_dict in benchmarking_dicts:
        image_sizes = get_benchmark_image_sizes(benchmark_dict=benchmark_dict)
        method = benchmark_dict["method"]
        for shape in image_sizes:
            shape_str = "{}x{}".format(shape[0], shape[1])
            if not shape in img_sizes:
                img_sizes.append(shape)
                methods_per_shape[shape_str] = [method]
            else:
                methods_per_shape[shape_str].append(method)

    
    # Now compute the performance for each method on each experiment
    # Only aggregate performance on images with the same size
    data_per_shape_dict = {}
    for shape in img_sizes:
        shape_str = "{}x{}".format(shape[0], shape[1])
        data = extract_experiment_data(experiment_names=exp_names, benchmarking_dicts=benchmarking_dicts, chosen_img_shape=shape)
        
        applicable_method_order = []
        for method in method_order:
            if method in methods_per_shape[shape_str]:
                applicable_method_order.append(method)

        # Turn everything into a pandas dataframe
        data_cols = {"names": applicable_method_order}
        print(shape, applicable_method_order)

        for name in exp_names:
            data_cols[name + "_mse"] = data[name]["mse"]
            data_cols[name + "_psnr"] = data[name]["psnr"]
            data_cols[name + "_ssim"] = data[name]["ssim"]
            data_cols[name + "_precision"] = data[name]["precision"]
            data_cols[name + "_recall"] = data[name]["recall"]
            if name == "embedding":
                data_cols[name + "_time"] = data[name]["emb_time"]
            if name == "tamper_detection":
                data_cols[name + "_det_time"] = data[name]["emb_time"]

        data_df = pd.DataFrame(data_cols)
        data_per_shape_dict[shape_str] = data_df
    
    writer = pd.ExcelWriter("benchmarking/benchmark_results.xlsx", engine="xlsxwriter")
    
    for shape_str, data_df in data_per_shape_dict.items():

        # Write the dataframe data to XlsxWriter. Turn off the default header and
        # index and skip a couple of rows to allow us to insert a user defined header.
        startrow = 3
        data_df.to_excel(writer, sheet_name=shape_str, startrow=startrow, header=False, index=False)
        
        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets[shape_str]

        # Get the dimensions of the dataframe.
        (max_row, max_col) = data_df.shape

        # Create a list of column headers, to use in add_table().
        column_settings = [{"header": column} for column in data_df.columns]

        # Add the Excel table structure.
        # We leave whitespace for user-defined columns
        worksheet.add_table(startrow - 1, 0, max_row, max_col - 1, {"columns": column_settings})

        # Make the columns wider for clarity.
        worksheet.set_column(0, max_col - 1, 12)

        # Add merged cells for clarity
        merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "white",
        }
    )

        # 0 = A, 1 = B, ...
        offset = 1
        # Excel rows start at one
        row_idx = 1

        # First at the unique embedding case
        embed_header_cells = 6
        worksheet.merge_range(get_xls_range_string(col_start=offset, col_stop=offset + embed_header_cells, row_start=row_idx, row_stop=row_idx), "Embedding", merge_format)
        offset += embed_header_cells
        # Then do the other cases
        other_header_cells = 5
        for column in exp_names[1:]:
            worksheet.merge_range(get_xls_range_string(col_start=offset, col_stop=offset + other_header_cells, row_start=row_idx, row_stop=row_idx), column, merge_format)
            offset += other_header_cells

        # Also again add the metrics for clarity
        metric_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "white",
        })

        row_idx = 2
        col_offset = 0
        worksheet.write("A2", "Method", metric_format)
        for column in data_df.columns:
            if "mse" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "mse", metric_format)
            elif "psnr" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "psnr", metric_format)
            elif "ssim" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "ssim", metric_format)
            elif "precision" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "precision", metric_format)
            elif "recall" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "recall", metric_format)
            elif "time" in column:
                worksheet.write("{}{}".format(index_to_xls(col_offset), row_idx), "time (s)", metric_format)
            
            col_offset += 1
        
        # Hide the row using pandas column names to remove clutter
        worksheet.set_row(2, None, None, {'hidden': True})
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()



if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    root_dir = os.path.join(current_directory, 'benchmarking')
    generate_xlsx(root_dir=root_dir)