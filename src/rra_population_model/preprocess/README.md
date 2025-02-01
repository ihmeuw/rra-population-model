# People per Structure Model Preprocessing

The set of pipelines in this subpackage are used to preprocess data for the people per
structure model. There are required steps (e.g. caching building density, preparing
the training data set and features) as well as optional steps (e.g. making
diagostic plots).  The preprocessing steps are run using the `ppsrun preprocess <step>`
command and individual steps can be run using the `pmtask preprocess <task>` command.
Where reasonable, the tasks have the same names as the steps, though some steps may
be broken into multiple kinds of tasks.  Consult `pmrun preprocess --help` and
`pmtask preprocess --help` for more information.

## Preprocessing Steps

1. `pmrun preprocess modeling_frame`:
2. `pmrun preprocess features`: This step generates raster features at the
   pixel level and writes them out by block. `features` has the single task:
   `pmtask preprocess features`.
3. `pmrun preprocess census_data`: Prepare database of linked census data for
   processing into training data.
4. `pmrun preprocess training_data`: This step generates the training data
   for the people per structure model including the features and the target variable.
5. (Optional) `pmrun preprocess summarize_training_data`: This step creates
   tile/scene-level summaries of the training data in tabular form. This is useful
   for understanding the distribution of the target variable and how it changes in time.
6. (Optional) `pmrun preprocess plot_training_data`: This step creates diagnostic
   plots from the summaries created in the previous step.
