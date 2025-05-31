import pandas as pd
import gradio as gr
import numpy as np


def generate_dataframes(
    annotator_metrics: dict[str, dict],
    overall_metrics: dict[str, dict],
    metric_name: str = "strength",
    sort_by: str = None,
    sort_ascending: bool = False,
):
    overall_metrics_df = get_overall_table_df(overall_metrics)

    annotator_table_df = get_annotator_table_df(
        annotator_metrics,
        metric_name=metric_name,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
    )

    return {
        "overall_metrics": overall_metrics_df,
        "annotator": annotator_table_df,
    }


def get_overall_table_df(overall_metrics: dict[str, dict]) -> gr.Dataframe:
    overall_df = pd.DataFrame(overall_metrics)
    overall_df.insert(0, "Metric", overall_df.index)  # insert metric name as col

    # In this table, each row has the same type (e.g. int or float)
    example_dict = overall_metrics[list(overall_metrics.keys())[0]]
    row_types = [type(value) for value in example_dict.values()]

    # Convert DataFrame to numpy array for styling
    shown_values = overall_df.to_numpy()

    def get_display_value(values):
        display_values = []
        for i, row in enumerate(values):
            display_row = []
            row_type = row_types[i]

            for col in row:
                # check if value is a string (indicating metric name)
                if not isinstance(col, str):
                    if row_type == int:
                        display_row.append(f"{int(col)}")
                    elif row_type == np.float64:
                        if abs(col) >= 100:
                            display_row.append(f"{col:.1f}")
                        else:
                            display_row.append(f"{col:.2f}")
                else:
                    display_row.append(col)
            display_values.append(display_row)
        return display_values

    display_value = get_display_value(shown_values)

    value = {
        "data": shown_values,
        "headers": overall_df.columns.tolist(),
        "metadata": {
            "display_value": display_value,
        },
    }

    return gr.Dataframe(value, interactive=False)


def get_annotator_table_df(
    annotator_metrics: dict[str, dict],
    metric_name: str = "strength",
    sort_by: str = None,
    sort_ascending: bool = False,
    color_scale: str = "berlin",
    neutral_value: float = 0.0,
) -> pd.DataFrame:

    initial_dataset = list(annotator_metrics.keys())[0]
    metric = metric_name  # Use the provided metric_name instead of hardcoded "strength"
    all_annotator_keys = set()
    for _, dataset_dict in annotator_metrics.items():
        all_annotator_keys.update(list(dataset_dict["metrics"][metric].keys()))
    all_annotator_keys = list(all_annotator_keys)

    metric_columns = {
        dataset_name: [
            dataset_dict["metrics"][metric].get(annotator_key, None)
            for annotator_key in all_annotator_keys
        ]
        for dataset_name, dataset_dict in annotator_metrics.items()
    }

    headers = ["Annotator"] + list(metric_columns.keys())

    # sanity check
    # for dataset_name, dataset_dict in annotator_metrics.items():
    #    assert list(dataset_dict["metrics"][metric].keys()) == annotator_keys, (
    #        f"Annotator keys mismatch for dataset {dataset_name}. "
    #        f"Expected: {annotator_keys}, "
    #        f"Found: {list(dataset_dict['metrics'][metric].keys())}"
    #        f"Dataset: {dataset_name}"
    #        f"Difference: {set(dataset_dict['metrics'][metric].keys()) - set(annotator_keys)}"
    #    )

    shown_df = pd.DataFrame(
        {
            "annotator": all_annotator_keys,
            **metric_columns,
        }
    )
    if len(metric_columns) > 1:
        shown_df["Max diff"] = abs(
            shown_df.iloc[:, 1:].max(axis=1) - shown_df.iloc[:, 1:].min(axis=1)
        )
        headers.append("Max diff")
    else:
        sort_by = list(metric_columns.keys())[0]

    # get max and min numerical value in the dataframe (ignoring non-numeric values)
    max_value = shown_df.select_dtypes(include=[np.number]).max().max()
    min_value = shown_df.select_dtypes(include=[np.number]).min().min()

    # sort by
    if sort_by is None:
        sort_by = list(metric_columns.keys())[0]

    shown_df = shown_df.sort_values(by=sort_by, ascending=sort_ascending)

    shown_values = shown_df.to_numpy()

    def get_styling(values):
        display_values = []
        positive_color = "#9eb0ff"  # matplotlib.colors.rgb2hex(cmap(0.0))
        negative_color = "#ffadad"  # matplotlib.colors.rgb2hex(cmap(1.0))
        for row in values:
            display_row = []
            for col in row:
                if isinstance(col, float):
                    if col > neutral_value:
                        denominator = max_value - neutral_value
                        if denominator != 0:
                            normalized_val = (
                                (col - neutral_value) / denominator
                            ) * 0.5 + 0.5
                        else:
                            normalized_val = 0.5
                    else:
                        denominator = neutral_value - min_value
                        if denominator != 0:
                            normalized_val = ((col - min_value) / denominator) * 0.5
                        else:
                            normalized_val = 0.5

                    # Calculate opacity based on normalized value (0 to 1)
                    opacity = abs(
                        normalized_val * 2 - 1
                    )  # Convert 0.5-centered scale to 0-1 scale
                    # Determine which color to use (positive or negative)
                    color_to_use = (
                        positive_color if normalized_val > 0.5 else negative_color
                    )
                    # Apply the color with opacity
                    display_row.append(
                        f"background-color: rgba({int(color_to_use[1:3], 16)}, {int(color_to_use[3:5], 16)}, {int(color_to_use[5:7], 16)}, {opacity}); color: var(--body-text-color);"
                        # f"background-color: color-mix(in srgb, {color_to_use} {opacity * 100}%, var(--body-background-fill)); color: var(--body-text-color);" # alternative with no line alternating color
                    )
                else:
                    display_row.append("")
            display_values.append(display_row)
        return display_values

    def get_display_value(values):
        display_values = []
        for row in values:
            display_row = []
            for col in row:
                if isinstance(col, float):
                    display_row.append(f"{col:.2f}")
                else:
                    display_row.append(col)
            display_values.append(display_row)
        return display_values

    styling = get_styling(shown_values)
    display_value = get_display_value(shown_values)

    value = {
        "data": shown_values,
        "headers": headers,
        "metadata": {
            "styling": styling,
            "display_value": display_value,
        },
    }

    column_widths = ["100px"] + ["60px"] * (len(headers) - 1)

    return gr.Dataframe(
        value=value,
        headers=headers,
        datatype=["str"] + ["number"] * len(metric_columns),
        # show_search="filter", # TODO: reactivate once sorting issue by Gradio is fixed, doesn't work in default Gradio version in HF spaces (5.20.1, works with some higher versions (tested 5.32.0))
        interactive=False,
        pinned_columns=1,
        column_widths=column_widths,
        wrap=True,
    )
