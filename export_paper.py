"""
Code to facilitate the export of data resulting from the benchmarking experiments, specifically for the paper.
"""

import os
from export import find_benchmarking_files, load_benchmarking_file_dict, get_benchmark_experiment_names, extract_experiment_data, get_benchmark_image_sizes, extract_experiment_data_finegrained

def method_to_cite(method: str):
    if method == "double_dwt_fast":
        return "\\cite{Bouarroudj2023_1}"
    elif method == "hi_lo_freq":
        return "Frequency"
    elif method == "hi_lo_spatial":
        return "\\hiloshort"
    elif method == "median":
        return "\\cite{Rajput2020}"
    elif method == "molina_garcia":
        return "\\cite{MolinaGarcia2020}"
    elif method == "sisaudia":
        return "\\cite{Sisaudia2024}"
    elif method == "svd":
        return "\\cite{Shebab2018}"
    elif method == "grayscale":
        return "\\grayshort"

def metric_keyword_to_name(kw):
    if kw == "mse":
        return "\\gls{mse}"
    elif kw == "psnr":
        return "\\gls{psnr}"
    elif kw == "ssim":
        return "\\gls{ssim}"
    elif kw == "emb_time":
        return "Time (s)"

def get_best_function(metric: str):
    if metric == "mse":
        return min
    elif metric == "psnr":
        return max
    elif metric == "ssim":
        return max
    elif metric == "emb_time":
        return min

def check_if_best(values, index, metric):
    best_function = get_best_function(metric=metric)

    return values[index] == best_function(values)

def generate_embedding_table(root_dir):
    target_file_paths = find_benchmarking_files(root_dir=root_dir)

    # Load all dictionaries
    benchmarking_dicts = list(map(lambda x: load_benchmarking_file_dict(x), target_file_paths))

    exp_names = []

    for benchmark_dict in benchmarking_dicts:
        applicable_experiments = get_benchmark_experiment_names(benchmark_dict=benchmark_dict)

        for name in applicable_experiments:
            if name not in exp_names:
                exp_names.append(name)

    # Now compute the performance for each method on each experiment
    data = extract_experiment_data(experiment_names=exp_names, benchmarking_dicts=benchmarking_dicts)
    method_order = [os.path.split(file_path)[-1][:-5] for file_path in target_file_paths]

    data_cols = {"names": method_order}

    for name in exp_names:
        data_cols[name + "_mse"] = data[name]["mse"]
        data_cols[name + "_psnr"] = data[name]["psnr"]
        data_cols[name + "_ssim"] = data[name]["ssim"]
        if name == "embedding":
            data_cols[name + "_time"] = data[name]["emb_time"]
    
    table_string = ""
    n_methods = len(method_order)
    metrics = ["mse", "psnr", "ssim", "emb_time"]
    
    # Make latex structure
    table_string += "\\begin{table}\n"
    table_string += "   \\centering\n"
    table_string += "   \\caption{Watermark embedding impact with regards to average image integrity and computation time.}\n"
    table_string += "   \\label{tab:experiments:embedding}\n"
    table_string += "   \\begin{tabular}{c" + "|c" * len(metrics) + "}\n"

    # Make header row
    metrics_names = "& ".join([metric_keyword_to_name(kw) for kw in metrics])
    header_row = "      Method & {} \\\\\n".format(metrics_names)
    header_row += "     \\hline\n"
    
    table_string += header_row
    
    # Make data rows:
    for i in range(n_methods):
        row = "     {}".format(method_to_cite(method_order[i]))
        
        for metric in metrics:
            value = data["embedding"][metric][i]
            is_best = check_if_best(values=data["embedding"][metric], index=i, metric=metric)
            value_str = "{}".format(value)

            row += " & {}".format(value_str)
        
        row += " \\\\\n"

        table_string += row

    # Table trailer

    table_string += "   \\end{tabular}\n"
    table_string += "\\end{table}"
    
    return table_string

def get_aggregated_data(root_dir):
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
        data = extract_experiment_data_finegrained(experiment_names=exp_names, benchmarking_dicts=benchmarking_dicts, chosen_img_shape=shape)

        data_per_shape_dict[shape_str] = data
    
    return data_per_shape_dict, method_order, img_sizes


def generate_cost_table(root_dir, decimals=2):
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)

    # print(json.dumps(data_per_shape_dict))
    
    # Now we can iterate over this data to make a table
    
    # For a cost table, we are only interested in (for every shape):
    # Embedding SSIM, PSNR, MSE
    # Embedding time
    # Tamper detection time (in the case of no tampering)
    metrics = ["MSE", "PSNR", "SSIM", "$T_e$ (s)", "$T_r$ (s)"]
    n_metrics = len(metrics)
    n_shapes = len(img_sizes)

    preamble = "\\begin{tabular}{l" + n_shapes * ("|" + "c" * n_metrics) + "}\n"
    header1 = " & " + "& ".join(["\\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_metrics, "Shape: {}x{}".format(shape[0], shape[1])) for shape in img_sizes]) + "\\\\\n"
    header2 = "Approach & " + " & ".join(metrics * n_shapes) + "\\\\\n"
    
    table = preamble + header1 + header2 + "\\hline \n"
    # print(header2)
    # Now we iterate over all methods:
    for method in method_order:
        # Compose a line:
        line = method_to_cite(method=method)

        for shape in img_sizes:
            shape_str = "{}x{}".format(shape[0], shape[1])
            # Check if we have experiments for this method
            if method in data_per_shape_dict[shape_str]["embedding"]["mse"].keys():
                mse = data_per_shape_dict[shape_str]["embedding"]["mse"][method]
                psnr = data_per_shape_dict[shape_str]["embedding"]["psnr"][method]
                ssim = data_per_shape_dict[shape_str]["embedding"]["ssim"][method]
                t_e = data_per_shape_dict[shape_str]["embedding"]["emb_time"][method]
                t_r = data_per_shape_dict[shape_str]["tamper_detection"]["emb_time"][method]

                line += " & {mse:.{decimals}f} & {psnr:.{decimals}f} & {ssim:.{decimals}f} & {t_e:.{decimals}f} & {t_r:.{decimals}f}".format(
                    mse=mse, psnr=psnr, ssim=ssim, t_e=t_e, t_r=t_r, decimals=decimals)
            else:
                line += " &  &  &  &  & "
        
        line += "\\\\\n"

        table += line
    
    table += "\\end{tabular}"
    print(table)


def generate_cost_table_2(root_dir, decimals=2, exclude_shapes=None):
    # Difference: We now only give MSE/PSNR/SSIM for multi-image tests
    # In the case there is only one image to test on, we only report processing times
    
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)
    
    if exclude_shapes is not None:
        new_img_sizes = []
        for shape in img_sizes:
            if shape not in exclude_shapes:
                new_img_sizes.append(shape)
        
        img_sizes = new_img_sizes

    # For a cost table, we are only interested in (for every shape):
    # Embedding time
    # Tamper detection time (in the case of no tampering)
    # For 512 x 512 images we are also interested in:
    # Embedding SSIM, PSNR, MSE

    metrics_long = ["MSE", "PSNR", "SSIM", "$T_e$ (s)", "$T_d$ (s)"]
    metrics_short = ["$T_e$ (s)", "$T_d$ (s)"]
    n_metrics_l = len(metrics_long)
    n_metrics_s = len(metrics_short)
    n_shapes = len(img_sizes)

    preamble = "\\begin{tabular}{l" + ("|" + "c" * n_metrics_l) + (n_shapes - 1) * ("|" + "c" * n_metrics_s) + "}\n"
    header1 = " & \\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_metrics_l, "Shape: {}x{}".format(img_sizes[0][0], img_sizes[0][1]))
    header1 += " & " + "& ".join(["\\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_metrics_s, "{}x{}".format(shape[0], shape[1])) for shape in img_sizes[1:]]) + "\\\\\n"
    header2 = "Approach & " + " & ".join(metrics_long) + " & "+ " & ".join(metrics_short * (n_shapes - 1)) + "\\\\\n"
    
    table = preamble + header1 + header2 + "\\hline \n"

    # Now we iterate over all methods:
    for method in method_order:
        # Compose a line:
        line = method_to_cite(method=method)

        for shape in img_sizes:
            shape_str = "{}x{}".format(shape[0], shape[1])
            if shape_str == "512x512":
                mse = data_per_shape_dict[shape_str]["embedding"]["mse"][method]
                psnr = data_per_shape_dict[shape_str]["embedding"]["psnr"][method]
                ssim = data_per_shape_dict[shape_str]["embedding"]["ssim"][method]
                t_e = data_per_shape_dict[shape_str]["embedding"]["emb_time"][method]
                t_d = data_per_shape_dict[shape_str]["tamper_detection"]["emb_time"][method]

                line += " & {mse:.{decimals}f} & {psnr:.{decimals}f} & {ssim:.{decimals}f} & {t_e:.{decimals}f} & {t_d:.{decimals}f}".format(
                    mse=mse, psnr=psnr, ssim=ssim, t_e=t_e, t_d=t_d, decimals=decimals)
            else:
                # Check if we have experiments for this method
                if method in data_per_shape_dict[shape_str]["embedding"]["emb_time"].keys():
                    t_e = data_per_shape_dict[shape_str]["embedding"]["emb_time"][method]
                    t_d = data_per_shape_dict[shape_str]["tamper_detection"]["emb_time"][method]
                    line += " & {t_e:.{decimals}f} & {t_d:.{decimals}f}".format(t_e=t_e, t_d=t_d, decimals=decimals)
                else:
                    line += " &  & "
        
        line += "\\\\\n"

        table += line
    
    table += "\\end{tabular}"

    # Add table to clipboard
    print(table)


def generate_tamper_detection_table(root_dir, include_recall=True, include_precision=False, include_f1=False, decimals=2):
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)
    # For a recovery table, we are interested in (for 512x512):
    # TPR for every relevant experiment
    # Interesting experiments:
    overarch_exp = ["Center crop random", "Center crop zero"]
    n_over = len(overarch_exp)
    percentages = ["10\\%", "25\\%", "50\\%", "75\\%"]
    n_percent = len(percentages)
    experiments = ["center_crop_random_10.0%", "center_crop_random_25.0%", "center_crop_random_50.0%", "center_crop_random_75.0%"]
    experiments += ["center_crop_0_10.0%", "center_crop_0_25.0%", "center_crop_0_50.0%", "center_crop_0_75.0%"]
    n_exp = len(experiments)

    # Metrics per experiment
    metrics = []
    if include_precision:
        metrics += ["P"]
    if include_recall:
        metrics += ["R"]
    if include_f1:
        metrics += ["F1"]

    n_metrics = len(metrics)
    print(metrics, n_percent, n_metrics, n_exp)
    preamble = "\\begin{tabular}{l" + n_exp * ("|" + "c" * n_metrics) + "}\n"
    header0 = " & " + "& ".join(["\\multicolumn{{{count}}}{{c}}{{{label}}}".format(count=n_percent * n_metrics, label=over) for over in overarch_exp]) + "\\\\\n"
    header1 = " & " + "& ".join(n_over * ["\\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_metrics, p) for p in percentages]) + "\\\\\n"
    header2 = "Approach & " + " & ".join(metrics * n_exp) + "\\\\\n"
    
    table = preamble + header0 + header1 + header2 + "\\hline \n"
    # print(header2)
    # Now we iterate over all methods:
    shape_str = "512x512"
    for method in method_order:
        # Compose a line:
        line = method_to_cite(method=method)

        for exp in experiments:

            if include_precision:
                precision = data_per_shape_dict[shape_str][exp]["precision"][method]
                line += " & {precision:.{decimals}f}".format(precision=precision, decimals=decimals)
            if include_recall:
                recall = data_per_shape_dict[shape_str][exp]["recall"][method]
                line += " & {recall:.{decimals}f}".format(recall=recall, decimals=decimals)
            if include_f1:
                precision = data_per_shape_dict[shape_str][exp]["precision"][method]
                recall = data_per_shape_dict[shape_str][exp]["recall"][method]
                f1 = (2 * precision * recall) / (precision + recall)
                line += " & {f1:.{decimals}f}".format(f1=f1, decimals=decimals)
        
        line += "\\\\\n"

        table += line
    
    table += "\\end{tabular}"
    print(table)


def generate_recovery_graph(root_dir, include_mse=True, include_psnr=True, include_ssim=True, decimals=2):
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)
    # For a recovery table, we are interested in (for 512x512):
    # SSIM, PSNR, MSE for every relevant experiment
    # Interesting experiments:
    overarch_exp = ["Center crop random", "Center crop zero"]
    n_over = len(overarch_exp)
    percentages = ["10\\%", "25\\%", "50\\%", "75\\%"]
    n_percent = len(percentages)
    experiments = ["center_crop_random_10.0%", "center_crop_random_25.0%", "center_crop_random_50.0%", "center_crop_random_75.0%"]
    experiments += ["center_crop_0_10.0%", "center_crop_0_25.0%", "center_crop_0_50.0%", "center_crop_0_75.0%"]
    n_exp = len(experiments)


def generate_vertical_recover_table(root_dir, include_mse=True, include_psnr=True, include_ssim=True, decimals=2):
    
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)
    overarch_exp = ["Center crop random", "Center crop zero"]
    n_over = len(overarch_exp)
    
    # Metrics per experiment
    metrics = []
    if include_mse:
        metrics += ["MSE"]
    if include_psnr:
        metrics += ["PSNR"]
    if include_ssim:
        metrics += ["SSIM"]

    percentages = ["10\\%", "25\\%", "50\\%", "75\\%"]
    n_metrics = len(metrics)
    n_percent = len(percentages)

    preamble = "\\begin{tabular}{l" + n_percent * ("|" + "c" * n_metrics) + "}\n"
    table = preamble

    for theme in overarch_exp:
        if theme != overarch_exp[0]:
            table += "\\hline \n"

        header0 = " & \\multicolumn{{{count}}}{{c}}{{{label}}} \\\\\n".format(count=n_percent * n_metrics, label=theme)
        header1 = " & " + "& ".join(["\\multicolumn{{{0}}}{{c|}}{{{1}}}".format(n_metrics, p) for p in percentages]) + "\\\\\n"
        header2 = "Approach & " + " & ".join(metrics * n_percent) + "\\\\\n"
        table += (header0 + header1 + header2 + "\\hline \n")
        
        base_string = theme.lower().replace(" ", "_").replace("zero", "0")

        shape_str = "512x512"
        for method in method_order:
            # Compose a line:
            line = method_to_cite(method=method)

            for percentage in percentages:
                base_percentage_string = percentage.replace("\\", ".0")
                exp = "{}_{}".format(base_string, base_percentage_string)
                if include_mse:
                    mse = data_per_shape_dict[shape_str][exp]["mse"][method]
                    line += " & {mse:.{decimals}f}".format(mse=mse, decimals=decimals)
                if include_psnr:
                    psnr = data_per_shape_dict[shape_str][exp]["psnr"][method]
                    line += " & {psnr:.{decimals}f}".format(psnr=psnr, decimals=decimals)
                if include_ssim:
                    ssim = data_per_shape_dict[shape_str][exp]["ssim"][method]
                    line += " & {ssim:.{decimals}f}".format(ssim=ssim, decimals=decimals)
            
            line += "\\\\\n"

            table += line

    table += "\\end{tabular}"
    print(table)


def generate_recovery_table(root_dir, include_mse=True, include_psnr=True, include_ssim=True, decimals=2):
    data_per_shape_dict, method_order, img_sizes = get_aggregated_data(root_dir)

    # For a recovery table, we are interested in (for 512x512):
    # SSIM, PSNR, MSE for every relevant experiment
    # Interesting experiments:
    overarch_exp = ["Center crop random", "Center crop zero"]
    n_over = len(overarch_exp)
    percentages = ["10\\%", "25\\%", "50\\%", "75\\%"]
    n_percent = len(percentages)
    experiments = ["center_crop_random_10.0%", "center_crop_random_25.0%", "center_crop_random_50.0%", "center_crop_random_75.0%"]
    experiments += ["center_crop_0_10.0%", "center_crop_0_25.0%", "center_crop_0_50.0%", "center_crop_0_75.0%"]
    n_exp = len(experiments)

    # Metrics per experiment
    metrics = []
    if include_mse:
        metrics += ["MSE"]
    if include_psnr:
        metrics += ["PSNR"]
    if include_ssim:
        metrics += ["SSIM"]

    n_metrics = len(metrics)
    print(metrics, n_percent, n_metrics, n_exp)
    preamble = "\\begin{tabular}{l" + n_exp * ("|" + "c" * n_metrics) + "}\n"
    header0 = " & " + "& ".join(["\\multicolumn{{{count}}}{{c}}{{{label}}}".format(count=n_percent * n_metrics, label=over) for over in overarch_exp]) + "\\\\\n"
    header1 = " & " + "& ".join(n_over * ["\\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_metrics, p) for p in percentages]) + "\\\\\n"
    header2 = "Approach & " + " & ".join(metrics * n_exp) + "\\\\\n"
    
    table = preamble + header0 + header1 + header2 + "\\hline \n"
    # print(header2)
    # Now we iterate over all methods:
    shape_str = "512x512"
    for method in method_order:
        # Compose a line:
        line = method_to_cite(method=method)

        for exp in experiments:

            if include_mse:
                mse = data_per_shape_dict[shape_str][exp]["mse"][method]
                line += " & {mse:.{decimals}f}".format(mse=mse, decimals=decimals)
            if include_psnr:
                psnr = data_per_shape_dict[shape_str][exp]["psnr"][method]
                line += " & {psnr:.{decimals}f}".format(psnr=psnr, decimals=decimals)
            if include_ssim:
                ssim = data_per_shape_dict[shape_str][exp]["ssim"][method]
                line += " & {ssim:.{decimals}f}".format(ssim=ssim, decimals=decimals)

            # line += " & {} & {} & {}".format(mse, psnr, ssim)
        
        line += "\\\\\n"

        table += line
    
    table += "\\end{tabular}"
    print(table)

if __name__ == "__main__":

    current_directory = os.path.dirname(__file__)
    root_dir = os.path.join(current_directory, 'benchmarking')
    #tb_string = generate_embedding_table(root_dir=root_dir)

    # print(tb_string) 
    # generate_cost_table(root_dir)
    generate_cost_table_2(root_dir, exclude_shapes=[(2048, 2048)])
    # generate_tamper_detection_table(root_dir, include_recall=True, include_precision=False, include_f1=False, decimals=4)
    # generate_vertical_recover_table(root_dir, include_mse=True, include_psnr=True, include_ssim=True, decimals=2)
    # generate_recovery_table(root_dir, include_mse=False, include_psnr=True, include_ssim=True, decimals=2)