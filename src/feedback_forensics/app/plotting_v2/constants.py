"""All constants defining the plot setup, all px """

### margins ###
MARGIN_LEFT = 10
MARGIN_RIGHT = 10
MARGIN_TOP = 10
MARGIN_BOTTOM = 10

### row constants ###
ROW_HEIGHT = 20  # standard height for each row in tables
ROW_Y_ANCHOR = "bottom"

### column constants ###
COLUMN_WIDTH = 100  # standard width for each column in tables
COLUMN_X_ANCHOR = "right"
COLUMN_X_ALIGN = "right"
# First column with metric names and principles
NAME_COLUMN_WIDTH = 150
NAME_COLUMN_X_ANCHOR = "right"
NAME_COLUMN_X_ALIGN = "right"

### gap constants ###
GAP_BETWEEN_TABLES = 50
GAP_BETWEEN_TABLE_VALUES_AND_HEADINGS = 10

### font constants ###
FONT_SIZE = 14
FONT_COLOR = "black"
FONT_WEIGHT = "normal"
NAME_FONT_WEIGHT = "bold"

INFO_ANNOTATION_DESCRIPTION = """
<b>Metrics Information</b><br>
We test what principles are implicitly encoded in the annotations<br>
(e.g. "a response in list format is preferred"). How much a principle is<br>
encoded in the annotations is measured by giving the principle to a<br>
principle-following AI annotator and checking how much of the original<br>
annotations it can reconstruct. Principles are typically generated using<br>
<i>Inverse Constitutional AI</i> (ICAI).<br><br>
For each principle, we calculate the following metrics:<br><br>
<b>Relevance <i>(rel)</i></b>: Proportion of datapoints that AI annotators deemed <br>
the principle relevant to. Ranges from 0 to 1.<br><br>
<b>Accuracy <i>(acc)</i></b>: Accuracy of principle-following AI annotator <br>
reconstructing the original annotations, when datapoints are deemed relevant.<br>
Ranges from 0 to 1.<br><br>
<b>Performance <i>(perf)</i></b>: Combines accuracy and relevance of a principle,<br>
ranges from -1 to 1. Calculated as perf=(acc-0.5)×rel×2. A value of 0 indicates<br>
no predictive performance (either due to random prediction or low relevance),<br>
values below 0 indicate principle-following AI annotator is worse than random<br>
annotator, and values above 0 indicate principle-following AI annotator is better<br>
than random annotator.
"""
