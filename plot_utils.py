from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource
from bokeh.models import HoverTool, BasicTicker
from bokeh.plotting import figure, output_file, save
from bokeh.transform import transform


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

from logger import Logger


def create_confusion_matrix(results_df, pred_column, logger):
  
  y_true = results_df['Orig_Label'].values
  crt_y_pred = results_df[pred_column].values

  num_classes = int(logger.config_dict['NUM_CLASSES'])
  cmat = confusion_matrix(y_true, crt_y_pred)

  df = pd.DataFrame(cmat,
    columns=[str(i) for i in range(num_classes)],
    index=[str(i) for i in range(num_classes)])

  df.index.name = 'Label'
  df.columns.name = 'Prediction'
  df = df.stack().rename("value").reset_index()

  colors = [matplotlib.colors.rgb2hex(c) for c in sns.color_palette("Blues")]

  mapper = LinearColorMapper(palette=colors, low=df.value.min(),
    high=df.value.max())
  p = figure(
    plot_width=800,
    plot_height=800,
    title="Confusion matrix for {} test data".format(pred_column),
    x_range=list(df.Label.drop_duplicates()),
    y_range=list(df.Prediction.drop_duplicates()),
    tools = "wheel_zoom,box_zoom,save, hover")
  p.rect(
    x="Label",
    y="Prediction",
    width=1,
    height=1,
    source=ColumnDataSource(df),
    line_color=None,
    fill_color=transform('value', mapper))
  color_bar = ColorBar(
    color_mapper=mapper,
    location=(0, 0),
    ticker=BasicTicker(desired_num_ticks=len(colors)))
  
  hover = p.select(dict(type=HoverTool))
  hover.tooltips = [
    ("Value", "@value"),
  ]
  p.add_layout(color_bar, 'right')

  output_file(logger.get_output_file("{}_cmat.html".format(pred_column)))
  save(p)


if __name__ == '__main__':

  logger = Logger(show = True, html_output = True, config_file = "config.txt",
    data_folder = "drive")

  results_df = pd.read_csv(logger.get_output_file(
    logger.config_dict['RESULTS_FILE']))

  create_confusion_matrix(results_df, "Orig_Label", logger)

  blurred_cols = ["Pred_Blurred_" + str(i) + "_Label" for i in range(5, 20, 5)]
  for col in blurred_cols:
    create_confusion_matrix(results_df, col, logger)

  dark_cols = ["Pred_Dark_" + str(i) + "_Label" for i in range(1, 4)]
  for col in dark_cols:
    create_confusion_matrix(results_df, col, logger)

  bright_cols = ["Pred_Bright_" + str(i) + "_Label" for i in range(1, 8)]
  for col in bright_cols:
    create_confusion_matrix(results_df, col, logger)

  occl_cols = ["Pred_Occl_" + str(i) + "_Label" for i in range(5, 30, 5)]
  for col in occl_cols:
    create_confusion_matrix(results_df, col, logger)